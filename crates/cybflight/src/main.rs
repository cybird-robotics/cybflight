#![no_std]
#![no_main]

use cybflight::hal::interrupt;
use embassy_executor::InterruptExecutor;
// `panic_probe` is the default panic handler for `probe-rs run` —
// emits a defmt log then `udf` so the debugger breaks on panic. The
// post-mortem subsystem provides its own `#[panic_handler]` that
// commits state to BKPSRAM before sys_reset; only one
// `#[panic_handler]` symbol may be linked per binary, so the two are
// mutually exclusive.
#[cfg(not(feature = "postmortem"))]
use panic_probe as _;
// When `postmortem` is on, the `#[panic_handler]` lives inside
// `cybflight::postmortem::fault`. The lib's `pub mod postmortem;`
// (gated on the same feature) pulls it in via `lib.rs`, so we don't
// need a separate import here — just gate out `panic_probe`.

// ─────────────────────────────────────────────────────────────────────
// Three-tier cooperative-preemptive executor architecture.
//
// Cortex-M NVIC priorities (lower number = higher priority):
//
//   P0–P5   HAL-managed (DMA completion, SPI, I2C, EXTI, USART, USB)
//   P6      EXECUTOR_HIGH — DShot motor output (8 kHz, safety-critical)
//   P10     EXECUTOR_CTRL — IMU readers, INDI, failsafe
//   Thread  Main executor — ESKF, MPC outer loop, RC, USB, telemetry,
//           mission planner, sensors (mag/baro/GPS)
//
// DShot preempts everything. The inner control chain on P10 (IMU → INDI
// → motors) preempts the thread executor, so compute-heavy thread tasks —
// the MPC SQP solver (~4 ms), the BFGS trajectory planner (~100 ms), or
// the ESKF predict/update — cannot stall the 8 kHz INDI loop. Cross-tier
// communication uses lock-free Signals and PubSub channels
// (CriticalSectionRawMutex).
//
// Interrupt-to-executor bindings repurpose NVIC slots the firmware does
// not otherwise use: CRS (clock recovery) and FDCAN_CAL (CAN calibration).
// ─────────────────────────────────────────────────────────────────────

/// High-priority interrupt executor (P6) — motor output.
///
/// DShot runs here so its ~8 kHz frame rate is maintained regardless of
/// any other task load. Preempts P10 and the thread executor.
static EXECUTOR_HIGH: InterruptExecutor = InterruptExecutor::new();

/// Mid-priority interrupt executor (P10) — inner control chain.
///
/// Hosts IMU readers, INDI controller, and failsafe. Preempts the thread
/// executor, so compute-heavy thread tasks (MPC SQP, ESKF, BFGS planner)
/// cannot starve the 8 kHz INDI loop. Preempted by DShot (P6) so motor
/// output remains the top-priority real-time path.
static EXECUTOR_CTRL: InterruptExecutor = InterruptExecutor::new();

/// CRS interrupt handler — drives the high-priority executor.
///
/// CRS (Clock Recovery System) is unused by this firmware, so we repurpose
/// its NVIC slot as the executor's wake interrupt.
#[interrupt]
unsafe fn CRS() {
    unsafe { EXECUTOR_HIGH.on_interrupt() }
}

/// FDCAN_CAL interrupt handler — drives the mid-priority executor.
///
/// The FDCAN calibration unit is unused by this firmware (CAN not wired
/// on any supported board), so its NVIC slot hosts the control executor.
#[interrupt]
unsafe fn FDCAN_CAL() {
    unsafe { EXECUTOR_CTRL.on_interrupt() }
}

#[embassy_executor::main]
async fn main(spawner: embassy_executor::Spawner) {
    cybflight::platform::enable_icache();
    let (board, defmt_uart) = cybflight::bsp::init();
    cybflight::serial_logger::init(defmt_uart);
    defmt::info!("cybflight: {} starting", cybflight::bsp::BOARD_NAME);
    // Confirm the BSP's declared clocks match what the RCC configuration
    // actually produced. DShot and WS2812 bit timings are computed from
    // those constants, so a mismatch mistimes both; reported, not fatal,
    // because a boot loop is the worse failure.
    cybflight::clocks::verify_clocks();
    // STM32H743 96-bit unique device ID (UID register @ 0x1FF1_E800),
    // read via embassy-stm32's `uid` module. Logged once at boot so each
    // board reports an identity for field bring-up and log correlation.
    defmt::info!("device UID: {}", cybflight::hal::uid::uid_hex());
    // Why did the previous session end? Captured from RCC.RSR in
    // `pre_init` (before any HAL code could clear the latch). IWDG =
    // firmware hung (panic/hardfault/starved executor); Brownout/BOR =
    // power dipped; Pin/POR = normal power-up or NRST. Also readable
    // later over USB via the `resetcause` shell verb.
    {
        let raw = cybflight::reset_cause::read();
        defmt::info!(
            "reset cause: {:?} (raw=0x{=u32:08x})",
            cybflight::reset_cause::classify(raw),
            raw,
        );
    }
    cybflight::status::STATUS
        .sender()
        .send(cybflight::status::SystemStatus::Booting);

    // --- Start the two interrupt executors ---
    {
        use cybflight::hal::interrupt::{self, InterruptExt, Priority};
        interrupt::CRS.set_priority(Priority::P6);
        interrupt::FDCAN_CAL.set_priority(Priority::P10);
    }
    let high_spawner = EXECUTOR_HIGH.start(cybflight::hal::interrupt::CRS);
    let ctrl_spawner = EXECUTOR_CTRL.start(cybflight::hal::interrupt::FDCAN_CAL);

    // --- Board-specific init ---
    //
    // board_init spawns:
    //   * DShot task on `high_spawner`
    //   * IMU reader tasks (icm_reader_task, etc.) on `ctrl_spawner` —
    //     these must preempt the planner so the control chain never
    //     starves for fresh gyro/accel data.
    //   * Everything else (LED, USB, RC, mag, baro, GPS, ESP bridge)
    //     on `spawner` (thread).
    cybflight::board_init::init(&spawner, &ctrl_spawner, &high_spawner, board).await;

    // --- Post-mortem: BKPSRAM enable + boot recovery ---
    //
    // Order matters: `bkpsram::enable` must run before any reader (so
    // the AHB4 clock is up before we touch 0x3880_0000), and
    // `boot_recovery` reads the prior boot's record + initializes
    // this boot's record. After this, the post-mortem task and the
    // fault handlers (panic / HardFault / PVD) all share a coherent
    // BKPSRAM state.
    //
    // `enable_pvd_brownout` arms the PVD interrupt — must be after
    // bsp::init configures clocks (PWR clock comes from APB1).
    #[cfg(feature = "postmortem")]
    {
        cybflight::postmortem::bkpsram::enable();
        cybflight::postmortem::recovery::boot_recovery();
        cybflight::postmortem::fault::enable_pvd_brownout();
        spawner
            .spawn(cybflight::postmortem::task::postmortem_task())
            .unwrap_or_else(|_| defmt::panic!("failed to spawn postmortem task"));
    }

    // Snapshot the reboot-flagged safety/RC params into their task-local
    // caches before the consumers spawn. Params are already loaded by
    // `board_init` (`params::init_from_flash`); until these run, both
    // caches serve their pre-parameter fallbacks.
    cybflight::sensors::rc::init_arm_config();
    cybflight::control::failsafe::init_fs_config();

    // --- IWDG: system-level safety net ---
    cybflight::watchdog::init();
    spawner
        .spawn(cybflight::watchdog::iwdg_feed_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn IWDG feed task"));
    spawner
        .spawn(cybflight::control::rc_interpreter::rc_interpreter_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn RC interpreter task"));

    // Failsafe on the control executor (P10): the RC parsers it depends
    // on also live on P10 now, and its controller-watchdog stamps
    // (`LAST_CONTROLLER_PUBLISH`) come from INDI/MPC on P10. Keeping
    // failsafe on the same tier as its inputs means stage-1 RC-loss
    // detection and controller-silence detection both respond within
    // a couple of milliseconds, even while the thread executor is
    // busy with a planner solve.
    ctrl_spawner
        .spawn(cybflight::control::failsafe::failsafe_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn failsafe task"));
    // ESKF: odometry estimator. Runs on the thread executor — its ~50 μs
    // predict (1 kHz) and ~120 μs mocap update (100 Hz) are small, but
    // co-locating it with MPC keeps the position-estimation → MPC pipeline
    // on the same tier and avoids any risk of blocking the 8 kHz INDI loop.
    // ESKF also publishes `VEHICLE_ATTITUDE` for telemetry consumers
    // (CRSF, ESP bridge, USB stream, blackbox). It goes out alongside
    // `VEHICLE_ODOMETRY` under the same `ODOM_DECIMATION`, so both are
    // 1 kHz — in *every* build, since the predict is decimated to
    // ~1 kHz whether the IMU runs at 1 kHz or 8 kHz.
    // Cross-executor communication (ESKF_GYRO_BIAS, ESKF_ACCEL_BIAS Signals and
    // VEHICLE_ODOMETRY PubSub) is lock-free by construction.
    #[cfg(feature = "est_pos_mocap")]
    spawner
        .spawn(cybflight::estimation::eskf_imu_mocap::estimation_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn estimation task"));
    #[cfg(feature = "est_pos_gps")]
    spawner
        .spawn(cybflight::estimation::eskf_imu_gps::estimation_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn estimation task"));

    // Mahony attitude filter: always-on, logging/telemetry only (feeds
    // the blackbox `/attitude` topic; on no-ESKF builds it is also the
    // sole VEHICLE_ATTITUDE publisher). ~1 kHz update on the thread
    // executor, negligible CPU — see estimation/mahony_task.rs.
    spawner
        .spawn(cybflight::estimation::mahony_task::mahony_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn Mahony task"));

    // spawner
    //     .spawn(cybflight::control::nmpc_driver::nmpc_task())
    //     .unwrap_or_else(|_| defmt::panic!("failed to spawn NMPC driver task"));

    spawner
        .spawn(cybflight::usb_serial::imu1_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn IMU1 stream task"));
    #[cfg(not(feature = "board_sakurah743"))]
    spawner
        .spawn(cybflight::usb_serial::imu2_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn IMU2 stream task"));
    // ── Inner loop controller (INDI, 8 kHz on P10) ────────────────
    ctrl_spawner
        .spawn(cybflight::control::indi_task::indi_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn INDI task"));

    // ── Outer loop controllers (thread executor) ────────────────────
    // Cascade position→attitude→geometric controller (100 Hz).
    #[cfg(feature = "outer_geometric")]
    spawner
        .spawn(cybflight::control::cascade_task::cascade_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn cascade task"));
    // MPC SQP outer loop (50 Hz). ~4 ms per solve — thread executor
    // so P10 INDI is never blocked.
    #[cfg(feature = "outer_mpc")]
    spawner
        .spawn(cybflight::control::outer_loop::control_loop_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn MPC outer loop task"));
    // Mission planner (offline MINCO schedule by default, or the BFGS
    // trajectory solver under `plan_online`; thread executor).
    #[cfg(feature = "outer_mpc")]
    spawner
        .spawn(cybflight::control::mission_planner::mission_planner_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn mission planner task"));
    spawner
        .spawn(cybflight::usb_serial::att_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn attitude stream task"));
    spawner
        .spawn(cybflight::usb_serial::ocp_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn OCP stream task"));
    spawner
        .spawn(cybflight::usb_serial::rc_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn RC stream task"));
    spawner
        .spawn(cybflight::usb_serial::rc_link_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn RC link stream task"));
    spawner
        .spawn(cybflight::usb_serial::dshot_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn DShot stream task"));
    spawner
        .spawn(cybflight::usb_serial::power_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn power stream task"));
    spawner
        .spawn(cybflight::usb_serial::attitude_control_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn attitude control stream task"));
    spawner
        .spawn(cybflight::usb_serial::gps_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn GPS stream task"));
    spawner
        .spawn(cybflight::usb_serial::gpsrtk_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn GPS RTK stream task"));
    #[cfg(not(feature = "board_sakurah743"))]
    spawner
        .spawn(cybflight::usb_serial::magext_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn mag ext stream task"));
    #[cfg(not(feature = "board_sakurah743"))]
    spawner
        .spawn(cybflight::usb_serial::magint_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn mag int stream task"));
    #[cfg(not(feature = "board_sakurah743"))]
    spawner
        .spawn(cybflight::usb_serial::baro1_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn baro1 stream task"));
    #[cfg(not(feature = "board_sakurah743"))]
    spawner
        .spawn(cybflight::usb_serial::baro2_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn baro2 stream task"));
    spawner
        .spawn(cybflight::usb_serial::vicon_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn Vicon stream task"));
    spawner
        .spawn(cybflight::usb_serial::timesync_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn time sync stream task"));
    spawner
        .spawn(cybflight::usb_serial::estimator_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn estimator stream task"));

    defmt::info!("all init done");
}
