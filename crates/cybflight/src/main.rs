#![no_std]
#![no_main]

use cybflight::hal::interrupt;
use embassy_executor::InterruptExecutor;
use panic_probe as _;

/// High-priority interrupt executor for safety-critical motor output.
///
/// DShot runs here so it preempts the thread executor (IMU, CRSF, USB, LED)
/// and maintains its ~8 kHz frame rate regardless of other task load.
static EXECUTOR_HIGH: InterruptExecutor = InterruptExecutor::new();

/// CRS interrupt handler — drives the high-priority executor.
///
/// CRS (Clock Recovery System) is unused by this firmware, so we repurpose
/// its NVIC slot as the executor's wake interrupt.
#[interrupt]
unsafe fn CRS() {
    unsafe { EXECUTOR_HIGH.on_interrupt() }
}

#[embassy_executor::main]
async fn main(spawner: embassy_executor::Spawner) {
    cybflight::platform::enable_icache();
    let (board, defmt_uart) = cybflight::bsp::init();
    cybflight::serial_logger::init(defmt_uart);
    defmt::info!("cybflight: {} starting", cybflight::bsp::BOARD_NAME);
    cybflight::status::STATUS
        .sender()
        .send(cybflight::status::SystemStatus::Booting);

    // --- Start high-priority interrupt executor for DShot ---
    //
    // Priority P6 (of P0..P15, lower = higher priority):
    //   P0–P5 : DMA completion, SPI, I2C, EXTI (HAL-managed)
    //   P6    : DShot executor — preempts thread mode, yields to DMA
    //   Thread: Main executor (IMU readers, CRSF, USB, LED)
    {
        use cybflight::hal::interrupt::{self, InterruptExt, Priority};
        let irq = interrupt::CRS;
        irq.set_priority(Priority::P6);
    }
    let high_spawner = EXECUTOR_HIGH.start(cybflight::hal::interrupt::CRS);

    // --- Board-specific init (spawns DShot on high_spawner, rest on spawner) ---
    cybflight::board_init::init(&spawner, &high_spawner, board).await;

    // --- IWDG: system-level safety net ---
    cybflight::watchdog::init();
    spawner
        .spawn(cybflight::watchdog::iwdg_feed_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn IWDG feed task"));
    // spawner
    //     .spawn(cybflight::sensors::attitude::mahony_task())
    //     .unwrap_or_else(|_| defmt::panic!("failed to spawn attitude task"));
    //
    // spawner
    //     .spawn(cybflight::control::nmpc_driver::nmpc_task())
    //     .unwrap_or_else(|_| defmt::panic!("failed to spawn NMPC driver task"));
    //

    // spawner
    //     .spawn(cybflight::estimate::feedthrough_estimate::feedthrough_estimate())
    //     .unwrap_or_else(|_| defmt::panic!("failed to spawn feedthrough estimate task"));

    spawner
        .spawn(cybflight::control::inner_loop::inner_loop_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn inner loop task"));


    // TODO: create a outer_loop task that runs around 100Hz
    // TODO: create a mission_plan_task that runs around 10 Hz

    spawner
        .spawn(cybflight::control::rc_interpreter::rc_interpreter_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn RC interpreter task"));

    spawner
        .spawn(cybflight::control::failsafe::failsafe_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn failsafe task"));
    // spawner
    //     .spawn(cybflight::sensors::attitude::mahony_task())
    //     .unwrap_or_else(|_| defmt::panic!("failed to spawn attitude task"));

    spawner
        .spawn(cybflight::estimation::eskf_imu_mocap::estimation_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn estimation task"));

    // spawner
    //     .spawn(cybflight::control::nmpc_driver::nmpc_task())
    //     .unwrap_or_else(|_| defmt::panic!("failed to spawn NMPC driver task"));

    spawner
        .spawn(cybflight::usb_serial::imu1_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn IMU1 stream task"));
    spawner
        .spawn(cybflight::usb_serial::imu2_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn IMU2 stream task"));
    // inner_loop_task replaces attitude_control as the sole controller.
    // spawner
    //     .spawn(cybflight::control::attitude_control::attitude_control_task())
    //     .unwrap_or_else(|_| defmt::panic!("failed to spawn attitude control task"));
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
        .spawn(cybflight::usb_serial::attitude_control_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn attitude control stream task"));
    spawner
        .spawn(cybflight::usb_serial::gps_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn GPS stream task"));
    spawner
        .spawn(cybflight::usb_serial::magext_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn mag ext stream task"));
    spawner
        .spawn(cybflight::usb_serial::magint_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn mag int stream task"));
    spawner
        .spawn(cybflight::usb_serial::baro1_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn baro1 stream task"));
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

    // spawner.spawn(cybflight::usb_serial::feedthrough_stream_task())
    //     .unwrap_or_else(|_| defmt::panic!("failed to spawn feedthrough stream task"));

    defmt::info!("all init done");
}
