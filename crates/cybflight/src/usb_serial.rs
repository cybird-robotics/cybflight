use core::fmt::Write;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::bsp;
use crate::comm;
use crate::control::ATTITUDE_CONTROL_SETPOINT;
use crate::control::OCP_SOLVER_OUTPUT;
use crate::estimation::{EstimatorPhase, ESTIMATOR_STATUS};
use crate::hal;
use crate::motors::ACTUATOR_MOTORS;
use crate::msgs;
use crate::platform;
use crate::sensors::gps::{
    carr_soln_str, GPS_HEALTH, LATEST_GPS_HEADING, LATEST_NAV_PVT, NAV_PVT_MAX_INTERVAL_RECENT_US,
    NAV_PVT_MEAN_INTERVAL_RECENT_US,
};
use crate::sensors::{
    BARO_1, BARO_2, DSHOT_TELEMETRY, GPS_FIX, IMU_1, IMU_2, MAG_EXT, MAG_INT, POWER_STATUS,
    RC_INPUT, RC_LINK_STATUS, VEHICLE_ATTITUDE, VICON_POSE,
};
use crate::shell::format::ShellMsg;
use crate::shell::write_all;
use crate::shell::ShellLine;
use crate::shell::WriteBuf;
use crate::shell::SHELL_OUT;
use embassy_futures::join::join;
use embassy_futures::select::{select, Either};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::pubsub::{PubSubChannel, Subscriber, WaitResult};
use embassy_time::{with_timeout, Duration, Instant, Timer};
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::driver::EndpointError;
use embassy_usb::Builder;
use hal::usb::Driver;
use nalgebra::Vector3;

type UsbDriver<'d> = Driver<'d, hal::peripherals::USB_OTG_FS>;

const PROMPT: &[u8] = b"> ";
const HELP_TEXT: &[u8] = b"\
  imu1                 one-shot IMU 1 snapshot\r\n\
  imu2                 one-shot IMU 2 snapshot\r\n\
  imurate              measure the actual IMU sample rate vs the build's\r\n\
  indistat             INDI step cost avg/max us since last call\r\n\
  att                  one-shot attitude snapshot\r\n\
  ocp                  one-shot OCP solver output\r\n\
  rc                   one-shot RC channel values\r\n\
  rcstats              one-shot RC link status\r\n\
  dshot                one-shot DShot telemetry\r\n\
  power                one-shot power status\r\n\
  gps                  one-shot GPS fix\r\n\
  gpshealth            one-shot GPS init/fix health\r\n\
  gpsrtk               one-shot RTK / NAV-PVT detail (corrections, accuracy)\r\n\
  fleetstate           one-shot neighbour fleet state (swarm pos/vel)\r\n\
  magext               one-shot external compass\r\n\
  magint               one-shot internal compass\r\n\
  baro1                one-shot barometer 1\r\n\
  baro2                one-shot barometer 2\r\n\
  vicon                one-shot Vicon pose\r\n\
  timesync             one-shot time sync status\r\n\
  eskf                 one-shot estimator status\r\n\
  health               aggregated arming/health snapshot\r\n\
  resetcause           why the board last reset (IWDG / brownout / pin / POR)\r\n\
  stream <topic> on    stream data on <topic>\r\n\
  stream <topic> off   stop data stream on <topic>\r\n\
                       (topics: imu1 imu2 att ocp rc rcstats dshot gps gpsrtk\r\n\
                        magext magint baro1 baro2 attcontrol vicon timesync eskf)\r\n\
  motor <1-4> <0-100>  set motor throttle (test mode)\r\n\
  param list           list all vehicle parameters\r\n\
  param get <name>     get a parameter value\r\n\
  param set <name> <v> set a parameter (in-memory)\r\n\
  param save [--prune] write params to flash (--prune: drop stale overrides)\r\n\
  param defaults       reset to compile-time defaults\r\n\
  mission list                            list available offline trajectories\r\n\
  mission get                             show the active mission: schedule, head/end, yaw/lookahead, flatness map, state\r\n\
  mission set <env> <variant> <speed>     select a trajectory (in-memory; 'param save' to persist)\r\n\
  led on               enable arm LEDs (red top / blue bottom, brighter when armed)\r\n\
  led off              disable arm LEDs\r\n\
  blackbox record on   manually start a flight log (flight_NNNN.mcap)\r\n\
  blackbox record off  stop the manual flight log\r\n\
  blackbox status      show current recording state and what's triggering it\r\n\
  blackbox set <tier>  set the record-set tier (none|small|mid|large|sysid); next session\r\n\
                       none=disabled, small=events+rc,\r\n\
                       mid=small+imu1/attitude/motors/health (default flight-debug tier),\r\n\
                       large=mid+ESKF odometry/mpc/tracking_error/control_setpoint/estimator_state,\r\n\
                       sysid=small+imu1_raw/motors/motor_state/odometry with INDI mirrors at >=500 Hz\r\n\
                       (sysid is NOT a superset of large; see 'blackbox_rate_div' for card bandwidth)\r\n\
  blackbox ls          list files in the FAT root (rejected while recording)\r\n\
  blackbox get <f> [<off> [<len>]]  download a file: 'OK <size>' + raw bytes + 'CRC <hex8>'\r\n\
                       (binary! use 'just blackbox-pull', not a bare terminal)\r\n\
  blackbox rm <f>      delete a file from the FAT root (rejected while recording)\r\n\
  blackbox clean       delete ALL flight_NNNN.mcap logs (other files untouched; rejected while recording)\r\n\
                       (recorder also auto-starts on real ARM_STATE arm: RC switch / failsafe path)\r\n\
                       (file numbering picks up after the highest existing flight_NNNN; survives reboots)\r\n\
  postmortem show      print the prior-boot crash record (reset cause, fatal kind, last events)\r\n\
  postmortem clear     wipe the BKPSRAM post-mortem slot\r\n\
  reboot               software reset\r\n\
  reboot --dfu         reset into USB DFU bootloader\r\n\
  help                 show this message\r\n\
  <Tab>                complete a command, stream topic, param or mission name\r\n\
";

// ---------------------------------------------------------------------------
// Stream enable flags — written by the shell, read by the stream tasks
// ---------------------------------------------------------------------------

pub static STREAM_IMU1: AtomicBool = AtomicBool::new(false);
pub static STREAM_IMU2: AtomicBool = AtomicBool::new(false);
pub static STREAM_ATT: AtomicBool = AtomicBool::new(false);
pub static STREAM_OCP: AtomicBool = AtomicBool::new(false);
pub static STREAM_RC: AtomicBool = AtomicBool::new(false);
pub static STREAM_RC_LINK: AtomicBool = AtomicBool::new(false);
pub static STREAM_DSHOT: AtomicBool = AtomicBool::new(false);
pub static STREAM_POWER: AtomicBool = AtomicBool::new(false);
pub static STREAM_GPS: AtomicBool = AtomicBool::new(false);
pub static STREAM_GPSRTK: AtomicBool = AtomicBool::new(false);
pub static STREAM_MAGEXT: AtomicBool = AtomicBool::new(false);
pub static STREAM_MAGINT: AtomicBool = AtomicBool::new(false);
pub static STREAM_BARO1: AtomicBool = AtomicBool::new(false);
pub static STREAM_BARO2: AtomicBool = AtomicBool::new(false);
pub static STREAM_ATTITUDE_CONTROL: AtomicBool = AtomicBool::new(false);
pub static STREAM_VICON: AtomicBool = AtomicBool::new(false);
pub static STREAM_TIMESYNC: AtomicBool = AtomicBool::new(false);
pub static STREAM_ESKF: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Generic stream bridge + concrete embassy task wrappers
// ---------------------------------------------------------------------------

/// Forwards messages from a pub/sub channel to the `SHELL_OUT` queue while
/// `enabled` is set.  One instance runs per topic, permanently.
pub async fn msg_stream_task<
    M: msgs::Message,
    const CAP: usize,
    const SUBS: usize,
    const PUBS: usize,
>(
    sub: &'static PubSubChannel<CriticalSectionRawMutex, M, CAP, SUBS, PUBS>,
    enabled: &'static AtomicBool,
) -> !
where
    for<'m> ShellMsg<'m, M>: core::fmt::Display,
{
    let mut sub = sub
        .subscriber()
        .expect("Programmer error: subscriber slots exhausted");
    loop {
        match sub.next_message().await {
            WaitResult::Message(msg) => {
                if enabled.load(Ordering::Relaxed) {
                    let mut line = ShellLine::new();
                    line.format(|w| {
                        write!(w, "{}\r\n", ShellMsg(&msg)).ok();
                    });
                    SHELL_OUT.try_send(line).ok();
                }
            }
            WaitResult::Lagged(_) => {} // subscriber behind publisher — expected for high-rate topics
        }
    }
}

#[embassy_executor::task]
pub async fn imu1_stream_task() {
    msg_stream_task(&IMU_1, &STREAM_IMU1).await
}

#[embassy_executor::task]
pub async fn imu2_stream_task() {
    msg_stream_task(&IMU_2, &STREAM_IMU2).await
}

#[embassy_executor::task]
pub async fn att_stream_task() {
    msg_stream_task(&VEHICLE_ATTITUDE, &STREAM_ATT).await
}

#[embassy_executor::task]
pub async fn ocp_stream_task() {
    msg_stream_task(&OCP_SOLVER_OUTPUT, &STREAM_OCP).await
}

#[embassy_executor::task]
pub async fn rc_stream_task() {
    msg_stream_task(&RC_INPUT, &STREAM_RC).await
}

#[embassy_executor::task]
pub async fn rc_link_stream_task() {
    msg_stream_task(&RC_LINK_STATUS, &STREAM_RC_LINK).await
}

#[embassy_executor::task]
pub async fn dshot_stream_task() {
    msg_stream_task(&DSHOT_TELEMETRY, &STREAM_DSHOT).await
}

#[embassy_executor::task]
pub async fn power_stream_task() {
    msg_stream_task(&POWER_STATUS, &STREAM_POWER).await
}

#[embassy_executor::task]
pub async fn gps_stream_task() {
    msg_stream_task(&GPS_FIX, &STREAM_GPS).await
}

/// Periodically writes a NAV-PVT/RTK detail line to `SHELL_OUT` at 2 Hz while
/// `STREAM_GPSRTK` is set. Uses the `LATEST_NAV_PVT` snapshot rather than the
/// `GPS_NAV_PVT` Signal so the ESKF consumer is never starved.
#[embassy_executor::task]
pub async fn gpsrtk_stream_task() {
    use embassy_time::Timer;
    loop {
        Timer::after(Duration::from_millis(500)).await;
        if !STREAM_GPSRTK.load(Ordering::Relaxed) {
            continue;
        }
        let snap = LATEST_NAV_PVT.lock(|c| c.get());
        let hsnap = LATEST_GPS_HEADING.lock(|c| c.get());
        let mut line = ShellLine::new();
        line.format(|w| match snap {
            None => {
                write!(w, "GPSRTK no NAV-PVT yet\r\n").ok();
            }
            Some(p) => {
                let age_ms = Instant::now().duration_since(p.timestamp).as_millis();
                write!(
                    w,
                    "GPSRTK fix={} sv={} rtk={}({}) dgps={} h={}mm v={}mm s={}mm/s \
                     vNED={},{},{} pos={:.7},{:.7} alt={}mm age={}ms\r\n",
                    p.fix_type,
                    p.num_sv,
                    carr_soln_str(p.carr_soln),
                    p.carr_soln,
                    if p.diff_soln { "yes" } else { "no" },
                    p.h_acc_mm,
                    p.v_acc_mm,
                    p.s_acc_mm_s,
                    p.vel_north_mm_s,
                    p.vel_east_mm_s,
                    p.vel_down_mm_s,
                    p.lat_deg,
                    p.lon_deg,
                    p.alt_msl_mm,
                    age_ms,
                )
                .ok();
            }
        });
        SHELL_OUT.try_send(line).ok();
        // Dual-antenna heading on its own line (UM982 only; skipped when absent
        // so u-blox streams stay quiet). Separate ShellLine keeps each within
        // one buffer.
        if let Some(hd) = hsnap {
            let mut hline = ShellLine::new();
            hline.format(|w| {
                const RAD2DEG: f32 = 180.0 / core::f32::consts::PI;
                let h_age = Instant::now().duration_since(hd.timestamp).as_millis();
                let (sh, ch) = libm::sincosf(hd.heading_rad);
                let (sp, cp) = libm::sincosf(hd.pitch_rad);
                let b_world = Vector3::new(cp * sh, cp * ch, sp); // ENU (E,N,U), heading direction in the ENU frame
                write!(
                    w,
                    "GPSRTK HDG hdg={:.1}deg pitch={:.1}deg (sh={:.2}deg sp={:.2}deg) rtk={}({}) bENU=[{:.3},{:.3},{:.3}] age={}ms\r\n",
                    hd.heading_rad * RAD2DEG,
                    hd.pitch_rad * RAD2DEG,
                    hd.heading_sigma_rad * RAD2DEG,
                    hd.pitch_sigma_rad * RAD2DEG,
                    carr_soln_str(hd.carr_soln),
                    hd.carr_soln,
                    b_world[0],
                    b_world[1],
                    b_world[2],
                    h_age,
                )
                .ok();
            });
            SHELL_OUT.try_send(hline).ok();
        }
    }
}

#[embassy_executor::task]
pub async fn magext_stream_task() {
    msg_stream_task(&MAG_EXT, &STREAM_MAGEXT).await
}

#[embassy_executor::task]
pub async fn magint_stream_task() {
    msg_stream_task(&MAG_INT, &STREAM_MAGINT).await
}

#[embassy_executor::task]
pub async fn baro1_stream_task() {
    msg_stream_task(&BARO_1, &STREAM_BARO1).await
}

#[embassy_executor::task]
pub async fn baro2_stream_task() {
    msg_stream_task(&BARO_2, &STREAM_BARO2).await
}

#[embassy_executor::task]
pub async fn attitude_control_stream_task() {
    msg_stream_task(&ATTITUDE_CONTROL_SETPOINT, &STREAM_ATTITUDE_CONTROL).await
}

#[embassy_executor::task]
pub async fn vicon_stream_task() {
    msg_stream_task(&VICON_POSE, &STREAM_VICON).await
}

#[embassy_executor::task]
pub async fn timesync_stream_task() {
    loop {
        embassy_time::Timer::after_millis(1000).await;
        if STREAM_TIMESYNC.load(Ordering::Relaxed) {
            let s = crate::comm::time_sync::status();
            let mut line = ShellLine::new();
            line.format(|w| {
                write!(
                    w,
                    "TimeSync(synced={}, offset={} us, ping_rtt={} us, clock_err={} us)\r\n",
                    s.synced, s.offset_us, s.ping_rtt_us, s.ping_clock_err_us
                )
                .ok();
            });
            SHELL_OUT.try_send(line).ok();
        }
    }
}

/// Periodically writes the estimator status to `SHELL_OUT` at 2 Hz while
/// `STREAM_ESKF` is set.  Runs permanently; cheap when stream is off.
#[embassy_executor::task]
pub async fn estimator_stream_task() {
    use embassy_time::Timer;
    loop {
        Timer::after(embassy_time::Duration::from_millis(500)).await;
        if STREAM_ESKF.load(Ordering::Relaxed) {
            let phase = ESTIMATOR_STATUS.lock(|c| c.get());
            let mut line = ShellLine::new();
            line.format(|w| {
                write_eskf_phase(w, &phase).ok();
                write!(w, "\r\n").ok();
            });
            SHELL_OUT.try_send(line).ok();
        }
    }
}

/// Format the estimator phase into `w`.
fn write_eskf_phase(w: &mut WriteBuf<'_>, phase: &EstimatorPhase) -> core::fmt::Result {
    #[cfg(feature = "est_pos_gps")]
    const AWAITING_LABEL: &str = "ESKF awaiting first RTK-fixed PVT (carr_soln==2)";
    #[cfg(feature = "est_pos_mocap")]
    const AWAITING_LABEL: &str = "ESKF awaiting mocap";
    #[cfg(not(any(feature = "est_pos_mocap", feature = "est_pos_gps")))]
    const AWAITING_LABEL: &str = "ESKF awaiting exteroceptive";

    match phase {
        EstimatorPhase::AwaitingExteroceptive => write!(w, "{}", AWAITING_LABEL),
        EstimatorPhase::Converging {
            roll_deg,
            pitch_deg,
            yaw_deg,
            pos,
            vel,
            gyro_bias,
            accel_bias,
            pos_reject_total,
            att_reject_total,
            pos_inflated_total,
            att_inflated_total,
            jump_total,
            carr_soln,
            num_sv,
            h_acc_mm,
        } => write!(
            w,
            "ESKF[Converging] roll={:.1} pitch={:.1} yaw={:.1}  \
             pos=[{:.2},{:.2},{:.2}]m  vel=[{:.2},{:.2},{:.2}]m/s  \
             gyro_bias=[{:.4},{:.4},{:.4}]rad/s  accel_bias=[{:.3},{:.3},{:.3}]m/s2  \
             rej(pos/att)={}/{} infl(pos/att)={}/{} jumps={}  \
             rtk(soln/sv/h_mm)={}/{}/{}",
            roll_deg,
            pitch_deg,
            yaw_deg,
            pos[0],
            pos[1],
            pos[2],
            vel[0],
            vel[1],
            vel[2],
            gyro_bias[0],
            gyro_bias[1],
            gyro_bias[2],
            accel_bias[0],
            accel_bias[1],
            accel_bias[2],
            pos_reject_total,
            att_reject_total,
            pos_inflated_total,
            att_inflated_total,
            jump_total,
            carr_soln,
            num_sv,
            h_acc_mm,
        ),
        EstimatorPhase::Running {
            roll_deg,
            pitch_deg,
            yaw_deg,
            pos,
            vel,
            gyro_bias,
            accel_bias,
            pos_reject_total,
            att_reject_total,
            pos_inflated_total,
            att_inflated_total,
            jump_total,
            carr_soln,
            num_sv,
            h_acc_mm,
        } => write!(
            w,
            "ESKF[Running] roll={:.1} pitch={:.1} yaw={:.1}  \
             pos=[{:.2},{:.2},{:.2}]m  vel=[{:.2},{:.2},{:.2}]m/s  \
             gyro_bias=[{:.4},{:.4},{:.4}]rad/s  accel_bias=[{:.3},{:.3},{:.3}]m/s2  \
             rej(pos/att)={}/{} infl(pos/att)={}/{} jumps={}  \
             rtk(soln/sv/h_mm)={}/{}/{}",
            roll_deg,
            pitch_deg,
            yaw_deg,
            pos[0],
            pos[1],
            pos[2],
            vel[0],
            vel[1],
            vel[2],
            gyro_bias[0],
            gyro_bias[1],
            gyro_bias[2],
            accel_bias[0],
            accel_bias[1],
            accel_bias[2],
            pos_reject_total,
            att_reject_total,
            pos_inflated_total,
            att_inflated_total,
            jump_total,
            carr_soln,
            num_sv,
            h_acc_mm,
        ),
    }
}

// ---------------------------------------------------------------------------
// Task entry points
// ---------------------------------------------------------------------------

#[embassy_executor::task]
pub async fn task(
    usb_otg: hal::Peri<'static, hal::peripherals::USB_OTG_FS>,
    dp: hal::Peri<'static, hal::peripherals::PA12>,
    dm: hal::Peri<'static, hal::peripherals::PA11>,
) {
    run(usb_otg, dp, dm).await
}

pub async fn run(
    usb_otg: hal::Peri<'static, hal::peripherals::USB_OTG_FS>,
    dp: hal::Peri<'static, hal::peripherals::PA12>,
    dm: hal::Peri<'static, hal::peripherals::PA11>,
) {
    let mut ep_out_buffer = [0u8; 256];
    let mut usb_config = hal::usb::Config::default();
    usb_config.vbus_detection = false;

    let driver = Driver::new_fs(
        usb_otg,
        bsp::UsbIrqs,
        dp,
        dm,
        &mut ep_out_buffer,
        usb_config,
    );

    let mut config = embassy_usb::Config::new(0x0483, 0x5740);
    config.manufacturer = Some("cybflight");
    config.product = Some("cybflight-shell");
    config.serial_number = Some("001");

    let mut config_descriptor = [0u8; 256];
    let mut bos_descriptor = [0u8; 256];
    let mut control_buf = [0u8; 64];
    let mut state = State::new();

    let mut builder = Builder::new(
        driver,
        config,
        &mut config_descriptor,
        &mut bos_descriptor,
        &mut [],
        &mut control_buf,
    );
    let mut class = CdcAcmClass::new(&mut builder, &mut state, 64);
    let mut usb = builder.build();

    join(usb.run(), async {
        loop {
            class.wait_connection().await;
            defmt::info!("USB shell connected");
            shell_loop(&mut class).await;
            defmt::info!("USB shell disconnected");
        }
    })
    .await;
}

// ---------------------------------------------------------------------------
// Shell loop — binary select: keyboard input vs. SHELL_OUT queue
// ---------------------------------------------------------------------------

async fn shell_loop<'d>(class: &mut CdcAcmClass<'d, UsbDriver<'d>>) {
    // Reset stream flags so a fresh connection starts silent.
    STREAM_IMU1.store(false, Ordering::Relaxed);
    STREAM_IMU2.store(false, Ordering::Relaxed);
    STREAM_ATT.store(false, Ordering::Relaxed);
    STREAM_OCP.store(false, Ordering::Relaxed);
    STREAM_RC.store(false, Ordering::Relaxed);
    STREAM_RC_LINK.store(false, Ordering::Relaxed);
    STREAM_DSHOT.store(false, Ordering::Relaxed);
    STREAM_POWER.store(false, Ordering::Relaxed);
    STREAM_GPS.store(false, Ordering::Relaxed);
    STREAM_GPSRTK.store(false, Ordering::Relaxed);
    STREAM_MAGEXT.store(false, Ordering::Relaxed);
    STREAM_MAGINT.store(false, Ordering::Relaxed);
    STREAM_BARO1.store(false, Ordering::Relaxed);
    STREAM_BARO2.store(false, Ordering::Relaxed);
    STREAM_ATTITUDE_CONTROL.store(false, Ordering::Relaxed);
    STREAM_VICON.store(false, Ordering::Relaxed);
    STREAM_TIMESYNC.store(false, Ordering::Relaxed);
    STREAM_ESKF.store(false, Ordering::Relaxed);

    // 96 (not 64): must fit the longest legal command line —
    // `blackbox get <32-char name> <u32 offset> <u32 len>` is 67.
    let mut line_buf = [0u8; 96];
    let mut line_len = 0usize;
    let mut rx_buf = [0u8; 64];

    let mut banner_buf = [0u8; 160];
    let mut w = WriteBuf::new(&mut banner_buf);
    let _ = write!(
        w,
        "cybflight v{} ({}) {} [{}] vehicle={}\r\ntype 'help' for commands\r\n\r\n> ",
        crate::BUILD_VERSION,
        crate::GIT_HASH,
        crate::BUILD_TIMESTAMP,
        crate::bsp::BOARD_NAME,
        crate::vehicle::BAKED_VEHICLE,
    );

    if write_all(class, w.as_slice()).await.is_err() {
        return;
    }

    loop {
        match select(class.read_packet(&mut rx_buf), SHELL_OUT.receive()).await {
            Either::First(Err(_)) => return,

            Either::First(Ok(n)) => {
                for &b in &rx_buf[..n] {
                    match b {
                        b'\r' | b'\n' => {
                            if write_all(class, b"\r\n").await.is_err() {
                                return;
                            }
                            let line = core::str::from_utf8(&line_buf[..line_len])
                                .unwrap_or("")
                                .trim();
                            if dispatch(class, line).await.is_err() {
                                return;
                            }
                            line_len = 0;
                            if write_all(class, PROMPT).await.is_err() {
                                return;
                            }
                        }
                        // DEL or backspace
                        0x7f | 0x08 => {
                            if line_len > 0 {
                                line_len -= 1;
                                if write_all(class, b"\x08 \x08").await.is_err() {
                                    return;
                                }
                            }
                        }
                        b'\t' => {
                            if crate::shell::complete::tab_complete(
                                class,
                                &mut line_buf,
                                &mut line_len,
                                PROMPT,
                            )
                            .await
                            .is_err()
                            {
                                return;
                            }
                        }
                        b if b >= 0x20 => {
                            if line_len < line_buf.len() {
                                line_buf[line_len] = b;
                                line_len += 1;
                                if write_all(class, &[b]).await.is_err() {
                                    return;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            Either::Second(line) => {
                if write_all(class, line.as_bytes()).await.is_err() {
                    return;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Command dispatch
// ---------------------------------------------------------------------------

async fn dispatch<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
    line: &str,
) -> Result<(), EndpointError> {
    match line {
        "" => {}
        "help" => {
            write_all(class, HELP_TEXT).await?;
        }
        "imu1" => {
            let mut sub = match IMU_1.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "imu2" => {
            let mut sub = match IMU_2.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "imurate" => {
            dispatch_imurate(class).await?;
        }
        "indistat" => {
            use crate::control::indi_task::step_stats;
            let (count, sum, max, period_max) = step_stats::take();
            let mut buf = [0u8; 160];
            let mut w = WriteBuf::new(&mut buf);
            if count == 0 {
                write!(w, "indi: no iterations since last call\r\n").ok();
            } else {
                let us = |c: u32| c as f32 * (1.0e6 / step_stats::SYSCLK_HZ as f32);
                write!(
                    w,
                    "indi: {} iters, step avg {:.1} us, max {:.1} us, loop period max {:.1} us\r\n",
                    count,
                    us(sum / count),
                    us(max),
                    us(period_max),
                )
                .ok();
            }
            write_all(class, w.as_slice()).await?;
        }
        "resetcause" => {
            dispatch_resetcause(class).await?;
        }
        "att" => {
            let mut sub = match VEHICLE_ATTITUDE.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "ocp" => {
            let mut sub = match OCP_SOLVER_OUTPUT.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "rc" => {
            let mut sub = match RC_INPUT.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "rcstats" => {
            let mut sub = match RC_LINK_STATUS.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 128).await?;
        }
        "attcontrol" => {
            let mut sub = match ATTITUDE_CONTROL_SETPOINT.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "stream imu1 on" => {
            STREAM_IMU1.store(true, Ordering::Relaxed);
            write_all(class, b"IMU1 stream on\r\n").await?;
        }
        "stream imu1 off" => {
            STREAM_IMU1.store(false, Ordering::Relaxed);
            write_all(class, b"IMU1 stream off\r\n").await?;
        }
        "stream imu2 on" => {
            STREAM_IMU2.store(true, Ordering::Relaxed);
            write_all(class, b"IMU2 stream on\r\n").await?;
        }
        "stream imu2 off" => {
            STREAM_IMU2.store(false, Ordering::Relaxed);
            write_all(class, b"IMU2 stream off\r\n").await?;
        }
        "stream att on" => {
            STREAM_ATT.store(true, Ordering::Relaxed);
            write_all(class, b"attitude stream on\r\n").await?;
        }
        "stream att off" => {
            STREAM_ATT.store(false, Ordering::Relaxed);
            write_all(class, b"attitude stream off\r\n").await?;
        }
        "stream ocp on" => {
            STREAM_OCP.store(true, Ordering::Relaxed);
            write_all(class, b"OCP solver output stream on\r\n").await?;
        }
        "stream ocp off" => {
            STREAM_OCP.store(false, Ordering::Relaxed);
            write_all(class, b"OCP solver output stream off\r\n").await?;
        }
        "stream rc on" => {
            STREAM_RC.store(true, Ordering::Relaxed);
            write_all(class, b"RC stream on\r\n").await?;
        }
        "stream rc off" => {
            STREAM_RC.store(false, Ordering::Relaxed);
            write_all(class, b"RC stream off\r\n").await?;
        }
        "stream rcstats on" => {
            STREAM_RC_LINK.store(true, Ordering::Relaxed);
            write_all(class, b"RC link status stream on\r\n").await?;
        }
        "stream rcstats off" => {
            STREAM_RC_LINK.store(false, Ordering::Relaxed);
            write_all(class, b"RC link status stream off\r\n").await?;
        }
        "dshot" => {
            let mut sub = match DSHOT_TELEMETRY.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "stream dshot on" => {
            STREAM_DSHOT.store(true, Ordering::Relaxed);
            write_all(class, b"DShot telemetry stream on\r\n").await?;
        }
        "stream dshot off" => {
            STREAM_DSHOT.store(false, Ordering::Relaxed);
            write_all(class, b"DShot telemetry stream off\r\n").await?;
        }
        "power" => {
            let mut sub = match POWER_STATUS.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 128).await?;
        }
        "stream power on" => {
            STREAM_POWER.store(true, Ordering::Relaxed);
            write_all(class, b"power stream on\r\n").await?;
        }
        "stream power off" => {
            STREAM_POWER.store(false, Ordering::Relaxed);
            write_all(class, b"power stream off\r\n").await?;
        }
        "gps" => {
            let mut sub = match GPS_FIX.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "gpshealth" => {
            let health = GPS_HEALTH.lock(|c| c.get());
            let max_iv = NAV_PVT_MAX_INTERVAL_RECENT_US.load(Ordering::Relaxed);
            let mean_iv = NAV_PVT_MEAN_INTERVAL_RECENT_US.load(Ordering::Relaxed);
            let mut buf = [0u8; 320];
            let mut w = WriteBuf::new(&mut buf);
            write!(w, "{}\r\n", health).ok();
            if max_iv == u32::MAX || mean_iv == u32::MAX {
                write!(w, "  cadence: <2 PVTs seen>\r\n").ok();
            } else {
                // Integer Hz×100 → "X.YY Hz" without floating point.
                // 0 mean defends against /0 if span_us collapsed.
                let hz_x100: u32 = if mean_iv == 0 {
                    0
                } else {
                    (100_000_000u64 / mean_iv as u64).min(u32::MAX as u64) as u32
                };
                write!(
                    w,
                    "  cadence (last 16 PVTs): mean={}us ({}.{:02} Hz), max={}us\r\n",
                    mean_iv,
                    hz_x100 / 100,
                    hz_x100 % 100,
                    max_iv
                )
                .ok();
            }
            write_all(class, w.as_slice()).await?;
        }
        "gpsrtk" => {
            let snap = LATEST_NAV_PVT.lock(|c| c.get());
            let hsnap = LATEST_GPS_HEADING.lock(|c| c.get());
            let mut buf = [0u8; 512];
            let mut w = WriteBuf::new(&mut buf);
            match snap {
                None => {
                    write!(w, "GPS: no NAV-PVT received yet\r\n").ok();
                }
                Some(p) => {
                    let age_ms = Instant::now().duration_since(p.timestamp).as_millis();
                    let fix_str = match p.fix_type {
                        0 => "no-fix",
                        1 => "dead-reckoning",
                        2 => "2D",
                        3 => "3D",
                        4 => "GNSS+DR",
                        5 => "time-only",
                        _ => "?",
                    };
                    write!(
                        w,
                        "GPS NAV-PVT (age={}ms):\r\n  fix      = {} ({}), gnss_fix_ok={}, num_sv={}\r\n  rtk      = {} (carr_soln={}), dgps={}\r\n  acc      = h={}mm v={}mm s={}mm/s\r\n  vel NED  = {} {} {} mm/s\r\n  pos      = {:.7}, {:.7}  alt_msl={}mm\r\n",
                        age_ms,
                        p.fix_type,
                        fix_str,
                        if p.gnss_fix_ok { "yes" } else { "no" },
                        p.num_sv,
                        carr_soln_str(p.carr_soln),
                        p.carr_soln,
                        if p.diff_soln { "yes" } else { "no" },
                        p.h_acc_mm,
                        p.v_acc_mm,
                        p.s_acc_mm_s,
                        p.vel_north_mm_s,
                        p.vel_east_mm_s,
                        p.vel_down_mm_s,
                        p.lat_deg,
                        p.lon_deg,
                        p.alt_msl_mm,
                    )
                    .ok();
                }
            }
            const RAD2DEG: f32 = 180.0 / core::f32::consts::PI;
            match hsnap {
                None => {
                    write!(w, "  heading  = n/a (no UNIHEADING / single-antenna receiver)\r\n")
                        .ok();
                }
                Some(hd) => {
                    let h_age = Instant::now().duration_since(hd.timestamp).as_millis();
                    let (sh, ch) = libm::sincosf(hd.heading_rad);
                    let (sp, cp) = libm::sincosf(hd.pitch_rad);
                    let b_world = Vector3::new(cp * sh, cp * ch, sp); // ENU (E,N,U), heading direction in the ENU frame
                    write!(
                        w,
                        "  heading  = {:.1}deg, pitch={:.1}deg (sh={:.2}deg sp={:.2}deg), rtk={} (carr_soln={}), age={}ms\r\n  bENU     = [{:.3}, {:.3}, {:.3}]  (E,N,U unit dir)\r\n",
                        hd.heading_rad * RAD2DEG,
                        hd.pitch_rad * RAD2DEG,
                        hd.heading_sigma_rad * RAD2DEG,
                        hd.pitch_sigma_rad * RAD2DEG,
                        carr_soln_str(hd.carr_soln),
                        hd.carr_soln,
                        h_age,
                        b_world[0],
                        b_world[1],
                        b_world[2],
                    )
                    .ok();
                }
            }
            write_all(class, w.as_slice()).await?;
        }
        "fleetstate" => {
            // Neighbour fleet state from the GS swarm broadcast. Demonstrates
            // safe, stale-aware access: whole-frame staleness from the message
            // timestamp, plus per-entry `valid`/`age`.
            let mut buf = [0u8; 384];
            let mut w = WriteBuf::new(&mut buf);
            match crate::sensors::NEIGHBOR_STATES.try_get() {
                None => {
                    write!(w, "fleet: no fleet-state received yet\r\n").ok();
                }
                Some(ns) => {
                    let now = Instant::now();
                    let msg_age = ns.message_age(now).as_millis();
                    let stale = ns.is_stale(now, Duration::from_millis(500));
                    write!(
                        w,
                        "fleet (msg_age={}ms{}): self_idx={} count={}\r\n",
                        msg_age,
                        if stale { " STALE" } else { "" },
                        ns.self_idx,
                        ns.count,
                    )
                    .ok();
                    for i in 0..(ns.count as usize).min(ns.states.len()) {
                        let s = ns.states[i];
                        let me = if i == ns.self_idx as usize { " (self)" } else { "" };
                        if s.valid {
                            write!(
                                w,
                                "  [{}]{} pos=[{} {} {}] vel=[{} {} {}] age={}ms\r\n",
                                i,
                                me,
                                s.position.x,
                                s.position.y,
                                s.position.z,
                                s.velocity.x,
                                s.velocity.y,
                                s.velocity.z,
                                s.age.as_millis(),
                            )
                            .ok();
                        } else {
                            write!(w, "  [{}]{} <no data>\r\n", i, me).ok();
                        }
                    }
                }
            }
            write_all(class, w.as_slice()).await?;
        }
        "vicon" => {
            let mut sub = match VICON_POSE.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "magext" => {
            let mut sub = match MAG_EXT.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "stream gps on" => {
            STREAM_GPS.store(true, Ordering::Relaxed);
            write_all(class, b"GPS stream on\r\n").await?;
        }
        "stream gps off" => {
            STREAM_GPS.store(false, Ordering::Relaxed);
            write_all(class, b"GPS stream off\r\n").await?;
        }
        "stream gpsrtk on" => {
            STREAM_GPSRTK.store(true, Ordering::Relaxed);
            write_all(class, b"GPS RTK stream on\r\n").await?;
        }
        "stream gpsrtk off" => {
            STREAM_GPSRTK.store(false, Ordering::Relaxed);
            write_all(class, b"GPS RTK stream off\r\n").await?;
        }
        "stream magext on" => {
            STREAM_MAGEXT.store(true, Ordering::Relaxed);
            write_all(class, b"mag ext stream on\r\n").await?;
        }
        "stream magext off" => {
            STREAM_MAGEXT.store(false, Ordering::Relaxed);
            write_all(class, b"mag ext stream off\r\n").await?;
        }
        "magint" => {
            let mut sub = match MAG_INT.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "stream magint on" => {
            STREAM_MAGINT.store(true, Ordering::Relaxed);
            write_all(class, b"mag int stream on\r\n").await?;
        }
        "stream magint off" => {
            STREAM_MAGINT.store(false, Ordering::Relaxed);
            write_all(class, b"mag int stream off\r\n").await?;
        }
        "baro1" => {
            let mut sub = match BARO_1.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "stream baro1 on" => {
            STREAM_BARO1.store(true, Ordering::Relaxed);
            write_all(class, b"baro1 stream on\r\n").await?;
        }
        "stream baro1 off" => {
            STREAM_BARO1.store(false, Ordering::Relaxed);
            write_all(class, b"baro1 stream off\r\n").await?;
        }
        "baro2" => {
            let mut sub = match BARO_2.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
        }
        "stream baro2 on" => {
            STREAM_BARO2.store(true, Ordering::Relaxed);
            write_all(class, b"baro2 stream on\r\n").await?;
        }
        "stream baro2 off" => {
            STREAM_BARO2.store(false, Ordering::Relaxed);
            write_all(class, b"baro2 stream off\r\n").await?;
        }
        "stream attcontrol on" => {
            STREAM_ATTITUDE_CONTROL.store(true, Ordering::Relaxed);
            write_all(class, b"attitude control setpoint stream on\r\n").await?;
        }
        "stream attcontrol off" => {
            STREAM_ATTITUDE_CONTROL.store(false, Ordering::Relaxed);
            write_all(class, b"attitude control setpoint stream off\r\n").await?;
        }
        "stream vicon on" => {
            STREAM_VICON.store(true, Ordering::Relaxed);
            write_all(class, b"Vicon pose stream on\r\n").await?;
        }
        "stream vicon off" => {
            STREAM_VICON.store(false, Ordering::Relaxed);
            write_all(class, b"Vicon pose stream off\r\n").await?;
        }
        "timesync" => {
            let s = comm::time_sync::status();
            let mut line = ShellLine::new();
            line.format(|w| {
                write!(
                    w,
                    "TimeSync(synced={}, offset={} us, ping_rtt={} us, clock_err={} us)\r\n",
                    s.synced, s.offset_us, s.ping_rtt_us, s.ping_clock_err_us
                )
                .ok();
            });
            write_all(class, line.as_bytes()).await?;
        }
        "stream timesync on" => {
            STREAM_TIMESYNC.store(true, Ordering::Relaxed);
            write_all(class, b"Time sync stream on\r\n").await?;
        }
        "stream timesync off" => {
            STREAM_TIMESYNC.store(false, Ordering::Relaxed);
            write_all(class, b"Time sync stream off\r\n").await?;
        }
        "eskf" => {
            let phase = ESTIMATOR_STATUS.lock(|c| c.get());
            let mut buf = [0u8; 256];
            let mut w = WriteBuf::new(&mut buf);
            write_eskf_phase(&mut w, &phase).ok();
            write!(w, "\r\n").ok();
            write_all(class, w.as_slice()).await?;
        }
        "stream eskf on" => {
            STREAM_ESKF.store(true, Ordering::Relaxed);
            write_all(class, b"ESKF status stream on\r\n").await?;
        }
        "stream eskf off" => {
            STREAM_ESKF.store(false, Ordering::Relaxed);
            write_all(class, b"ESKF status stream off\r\n").await?;
        }
        "health" => {
            // Read from the latest-frame cells populated by the RC task.
            // A fresh RC_LINK_STATUS / RC_INPUT subscriber has no history —
            // try_next_message_pure() on it returns None even when frames
            // are arriving every ~10 ms, which shows up as link=inactive
            // in the report.
            let link = crate::sensors::rc::LATEST_RC_LINK_STATUS
                .lock(|c| c.borrow().clone())
                .map(|s| crate::health::LinkSnapshot {
                    active: true,
                    // saturating_: the RC task tags `s.timestamp` from its
                    // own `Instant::now()`; it can read as ahead of the
                    // shell's `Instant::now()` by a microsecond if the two
                    // tasks are scheduled across a tick boundary. Bare
                    // `duration_since` would panic on backwards subtraction
                    // and boot-loop the firmware after every `health` call.
                    age_ms: Instant::now()
                        .saturating_duration_since(s.timestamp)
                        .as_millis(),
                    quality: s.link_quality,
                })
                .unwrap_or(crate::health::LinkSnapshot {
                    active: false,
                    age_ms: 0,
                    quality: 0,
                });
            let throttle = crate::sensors::rc::LATEST_RC_INPUT
                .lock(|c| c.borrow().clone())
                .and_then(|r| {
                    // THROTTLE_CHANNEL is index 2; only trust if frame
                    // had at least that many channels.
                    if r.channel_count > 2 {
                        Some(r.channels[2])
                    } else {
                        None
                    }
                });

            let snap = crate::health::SystemHealth::snapshot(link, throttle);
            // Write the report in chunks so we don't blow the 256-byte
            // ShellLine buffer (the report is multi-line and easily
            // exceeds 256 bytes once attitude/faults are included).
            let mut buf = [0u8; 1024];
            let mut w = WriteBuf::new(&mut buf);
            let _ = snap.write_report(&mut w);
            write_all(class, w.as_slice()).await?;
        }
        "reboot" => {
            write_all(class, b"rebooting...\r\n").await?;
            platform::sys_reboot();
        }
        "reboot --dfu" => {
            write_all(class, b"entering DFU mode...\r\n").await?;
            platform::enter_dfu();
        }
        line if line.starts_with("motor ") => {
            dispatch_motor(class, line).await?;
        }
        line if line.starts_with("param") => {
            dispatch_param(class, line).await?;
        }
        #[cfg(feature = "outer_mpc")]
        line if line.starts_with("mission") => {
            dispatch_mission(class, line).await?;
        }
        "led on" | "led off" => {
            dispatch_led(class, line == "led on").await?;
        }
        "blackbox record on" | "blackbox record off" => {
            dispatch_blackbox_record(class, line == "blackbox record on").await?;
        }
        "blackbox status" => {
            dispatch_blackbox_status(class).await?;
        }
        line if line.starts_with("blackbox set ") || line == "blackbox set" => {
            dispatch_blackbox_set(class, line).await?;
        }
        "blackbox ls" => {
            dispatch_blackbox_ls(class).await?;
        }
        line if line.starts_with("blackbox get ") || line == "blackbox get" => {
            dispatch_blackbox_get(class, line).await?;
        }
        line if line.starts_with("blackbox rm ") || line == "blackbox rm" => {
            dispatch_blackbox_rm(class, line).await?;
        }
        "blackbox clean" => {
            dispatch_blackbox_clean(class).await?;
        }
        #[cfg(feature = "postmortem")]
        "postmortem show" => {
            dispatch_postmortem_show(class).await?;
        }
        #[cfg(feature = "postmortem")]
        "postmortem clear" => {
            dispatch_postmortem_clear(class).await?;
        }
        _ => {
            write_all(class, b"unknown command (try 'help')\r\n").await?;
        }
    }
    Ok(())
}

/// `blackbox record on/off` — manual recorder toggle for bench use.
///
/// Sets [`crate::blackbox::RECORDER_HOLD`] (`AtomicBool`). The
/// recorder's "should-record" predicate is `IS_ARMED ||
/// RECORDER_HOLD`, so toggling this flag is enough to start / stop
/// a flight log without touching `ARM_STATE` (which would also
/// cause the DShot task to drive idle PWM on the motor outputs).
async fn dispatch_blackbox_record<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
    hold: bool,
) -> Result<(), EndpointError> {
    crate::blackbox::RECORDER_HOLD.store(hold, Ordering::Release);
    let msg: &[u8] = if hold {
        b"blackbox record: ON  (recorder will open flight_NNNN.mcap on next debounce-confirmed edge)\r\n"
    } else {
        b"blackbox record: OFF (recorder will close after the next disarm-poll cycle)\r\n"
    };
    write_all(class, msg).await
}

/// `blackbox status` shell renderer. Reads the two atomics that
/// drive the recorder predicate (`motors::IS_ARMED` and
/// `blackbox::RECORDER_HOLD`) and reports the resulting state plus
/// what's currently holding it active.
///
/// Pure synchronous — no Signal round-trip through `blackbox_task` —
/// so it works even when the recorder is mid-flush and the task is
/// busy. Also runs on boards without storage; in that case the
/// atomics still exist (they're firmware-wide) but the recorder
/// itself isn't spawned, so we say so.
async fn dispatch_blackbox_status<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
) -> Result<(), EndpointError> {
    let armed = crate::motors::IS_ARMED.load(Ordering::Acquire);
    let hold = crate::blackbox::RECORDER_HOLD.load(Ordering::Acquire);
    let rs = crate::blackbox::record_set();

    let mut buf = [0u8; 256];
    let mut w = WriteBuf::new(&mut buf);
    let storage_note = if !bsp::HAS_BLACKBOX_STORAGE {
        " (board has no storage; recorder is not spawned)"
    } else {
        ""
    };
    let trigger = match (armed, hold) {
        (true, true) => "armed + shell hold",
        (true, false) => "armed",
        (false, true) => "shell hold",
        (false, false) => "none",
    };
    // The recorder ignores arm-edges when record_set is None, so
    // surface that as the effective state instead of "recording".
    let state = if (armed || hold) && rs.enabled() {
        "recording"
    } else if (armed || hold) && !rs.enabled() {
        "idle (record_set=none, edge ignored)"
    } else {
        "idle"
    };
    let p = crate::params::get();
    let _ = write!(
        w,
        "blackbox status: {}  (trigger: {}){}\r\n\
         \x20                 IS_ARMED={}, RECORDER_HOLD={}\r\n\
         \x20                 record_set={}, rate_div={}, mute_mask={:#010x}\r\n",
        state,
        trigger,
        storage_note,
        armed,
        hold,
        rs.name(),
        p.system.blackbox_rate_div,
        p.system.blackbox_mute_mask,
    );
    write_all(class, w.as_slice()).await?;

    // Last-session lines (Stage 7): what the most recent close produced,
    // plus the card-level I/O counters behind it.
    let last = crate::blackbox::LAST_SESSION.lock(|c| c.get());
    let mut buf = [0u8; 384];
    let mut w = WriteBuf::new(&mut buf);
    match last {
        None => {
            let _ = write!(w, "\x20                 last session: none since boot\r\n");
        }
        Some(s) => {
            let _ = write!(
                w,
                "\x20                 last session: /flight_{:04}.mcap  \
                 {} bytes, {} msgs, {} drops, {} enc-ovf",
                s.seq, s.bytes, s.messages, s.drops, s.encode_overflows,
            );
            match s.error {
                None => {
                    let _ = write!(w, "  [ok]\r\n");
                }
                Some(e) => {
                    let _ = write!(w, "  [FAULT: {}]\r\n", blackbox_fat_reason(e));
                }
            }
            let io = s.io;
            let avg = if io.writes > 0 { io.blocks_written / io.writes } else { 0 };
            let _ = write!(
                w,
                "\x20                 card io: {} writes ({} blocks, avg {} max {} per cmd), \
                 {} reads ({} blocks), {} CMD25 fallbacks\r\n",
                io.writes, io.blocks_written, avg, io.max_batch, io.reads, io.blocks_read, io.fallbacks,
            );
        }
    }
    write_all(class, w.as_slice()).await
}

/// `blackbox set <none|small|mid|large|sysid>` — change the active record
/// set and persist it to flash.
///
/// Three-step update: the live atomic the recorder reads, the
/// in-memory param copy, then `save_to_flash`. Refused while armed
/// because `save_to_flash` can trigger a flash erase that blocks the
/// thread executor (ESKF, MPC, IWDG feeder) for ~1–2 s with the IWDG
/// extended — fine on the bench, dangerous mid-flight. Mirrors the
/// armed-guard precedent set by `led on/off` and `mission set`.
///
/// The recorder snapshots the tier at session open, so even when
/// disarmed a mid-session change does nothing until the current
/// file closes — there's no way to be mid-session at the moment
/// this handler runs (we already required disarm), but the
/// `RECORDER_HOLD` shell-bench mode can still be active.
async fn dispatch_blackbox_set<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
    line: &str,
) -> Result<(), EndpointError> {
    if crate::motors::IS_ARMED.load(Ordering::Acquire) {
        return write_all(
            class,
            b"refused: disarm before changing blackbox record-set (flash save would stall CPU)\r\n",
        )
        .await;
    }
    let arg = line.split_ascii_whitespace().nth(2);
    let parsed = arg.and_then(crate::blackbox::RecordSet::parse);
    match parsed {
        Some(rs) => {
            // Update the live atomic the recorder reads.
            crate::blackbox::set_record_set(rs);
            // Persist in flash so the tier survives reboot. Mirrors
            // the `led on/off` auto-save path: mutate the in-memory
            // params copy, write back, then flush to sector 7.
            let mut params = crate::params::get();
            params.system.blackbox_record_set = rs as u8;
            crate::params::set(params);
            let save_status = match crate::params::save_to_flash() {
                Ok(()) => "saved to flash",
                Err(e) => {
                    defmt::warn!("blackbox set: flash save failed: {}", e);
                    "NOT saved (flash error - see defmt log)"
                }
            };
            let mut buf = [0u8; 192];
            let mut w = WriteBuf::new(&mut buf);
            // RECORDER_HOLD-only sessions can still be live (since the
            // armed guard only checks IS_ARMED). Note that case so the
            // user knows the in-flight file isn't switching tiers.
            let mid_note = if crate::blackbox::RECORDER_HOLD.load(Ordering::Acquire) {
                "  (recorder hold active - current file keeps its tier snapshot)"
            } else {
                ""
            };
            let _ = write!(
                w,
                "blackbox set: record_set = {} ({}){}\r\n",
                rs.name(),
                save_status,
                mid_note,
            );
            write_all(class, w.as_slice()).await
        }
        None => {
            write_all(
                class,
                b"usage: blackbox set <none|small|mid|large|sysid>\r\n\
                  \x20  none  - recorder muted (arm-edges produce no file)\r\n\
                  \x20  small - events + rc\r\n\
                  \x20  mid   - + full controller stream (default)\r\n\
                  \x20  large - + raw IMU (analyse.py-style RPM-notch fits)\r\n",
            )
            .await
        }
    }
}

/// `blackbox ls` shell renderer. Asks the blackbox task to scan the
/// FAT root and prints the result. Read-only, but still routed
/// through the task because that task owns the SDMMC peripheral.
async fn dispatch_blackbox_ls<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
) -> Result<(), EndpointError> {
    use crate::blackbox::{LsReport, LS_REQUEST, LS_RESULT};

    const LABEL: &str = "blackbox ls";

    if !bsp::HAS_BLACKBOX_STORAGE {
        return write_all(class, b"blackbox ls: no storage backend on this board\r\n").await;
    }

    let _ = LS_RESULT.try_take();
    LS_REQUEST.signal(());

    // 10 s slack covers a cold mount + scan on slow microSDs. The op
    // is read-only — even a card with thousands of root entries
    // returns well inside this.
    let task_timeout = Duration::from_secs(10);

    match with_timeout(task_timeout, LS_RESULT.wait()).await {
        Ok(LsReport::Ok {
            entries,
            truncated,
            total_files,
        }) => {
            // Header
            {
                let mut buf = [0u8; 96];
                let mut w = WriteBuf::new(&mut buf);
                let shown = entries.len();
                if truncated {
                    let _ = write!(
                        w,
                        "{}: {} files (showing first {}; bump LS_MAX_ENTRIES)\r\n",
                        LABEL, total_files, shown,
                    );
                } else {
                    let _ = write!(w, "{}: {} file(s)\r\n", LABEL, total_files);
                }
                write_all(class, w.as_slice()).await?;
            }
            // One line per entry. Print in chunks so a long list
            // doesn't blow the stack.
            for e in entries.iter() {
                let mut buf = [0u8; 96];
                let mut w = WriteBuf::new(&mut buf);
                let _ = write!(w, "  /{}  {} bytes\r\n", e.name.as_str(), e.size);
                write_all(class, w.as_slice()).await?;
            }
            Ok(())
        }
        Ok(LsReport::Failed(stage)) => {
            let mut buf = [0u8; 256];
            let mut w = WriteBuf::new(&mut buf);
            let _ = write!(w, "{}: FAILED — {}\r\n", LABEL, blackbox_fat_reason(stage));
            write_all(class, w.as_slice()).await
        }
        Ok(LsReport::Busy) => {
            write_all(
                class,
                b"blackbox ls: REJECTED - recorder is mid-flight (disarm first)\r\n",
            )
            .await
        }
        Err(_) => {
            let mut buf = [0u8; 96];
            let mut w = WriteBuf::new(&mut buf);
            let _ = write!(
                w,
                "{}: TIMEOUT — task didn't respond in {} ms\r\n",
                LABEL,
                task_timeout.as_millis(),
            );
            write_all(class, w.as_slice()).await
        }
    }
}

/// `blackbox get <file> [<offset> [<len>]]` — stream a FAT-root file
/// to the host as raw binary.
///
/// Wire protocol: one text header line — `OK <size>\r\n` on success,
/// or an error line and nothing else — then exactly `<size>` raw
/// bytes, then a text trailer `\r\nCRC <hex8>\r\n` (CRC-32 of the
/// raw bytes). A mid-stream fault truncates the binary and emits
/// `\r\nERR <reason>\r\n` in place of the CRC trailer.
///
/// Binary-safety: `write_all` is byte-transparent, and while this
/// handler runs the shell loop is not selecting on `SHELL_OUT`, so
/// async stream lines queue up rather than interleaving with the
/// payload. The intended consumer is `tools/blackbox_pull.py`
/// (`just blackbox-pull`), not a human at a terminal.
///
/// `<offset>`/`<len>` clamp to the file size, enabling partial and
/// resumed pulls; an offset at/past EOF yields a valid `OK 0`.
async fn dispatch_blackbox_get<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
    line: &str,
) -> Result<(), EndpointError> {
    use crate::blackbox::{GetChunk, GetReport, GetRequest, GET_REQUEST, GET_RESULT, GET_STREAM};

    const LABEL: &str = "blackbox get";
    const USAGE: &[u8] = b"usage: blackbox get <file> [<offset> [<len>]]\r\n";

    if !bsp::HAS_BLACKBOX_STORAGE {
        return write_all(class, b"blackbox get: no storage backend on this board\r\n").await;
    }

    let mut parts = line.split_ascii_whitespace();
    let _ = parts.next(); // "blackbox"
    let _ = parts.next(); // "get"
    let Some(name_str) = parts.next() else {
        return write_all(class, USAGE).await;
    };
    let mut name: crate::blackbox::fat::EntryName = heapless::String::new();
    if name.push_str(name_str.trim_start_matches('/')).is_err() {
        return write_all(class, b"blackbox get: name too long (max 32 chars)\r\n").await;
    }
    let offset = match parts.next() {
        None => 0,
        Some(s) => match s.parse::<u32>() {
            Ok(v) => v,
            Err(_) => return write_all(class, USAGE).await,
        },
    };
    let len = match parts.next() {
        None => None,
        Some(s) => match s.parse::<u32>() {
            Ok(v) => Some(v),
            Err(_) => return write_all(class, USAGE).await,
        },
    };
    if parts.next().is_some() {
        return write_all(class, USAGE).await;
    }

    // Fast-path refusal — the task would answer Busy anyway, but this
    // keeps the failure instant while a session is recording (the
    // task is blocked inside `run_session` and can't reply).
    if crate::blackbox::should_record() {
        return write_all(
            class,
            b"blackbox get: REJECTED - recorder is mid-flight (disarm first)\r\n",
        )
        .await;
    }

    // Drop stale state from a previous (possibly aborted) op.
    let _ = GET_RESULT.try_take();
    while GET_STREAM.try_receive().is_ok() {}
    GET_REQUEST.signal(GetRequest { name, offset, len });

    // 10 s slack covers a cold mount on slow microSDs (same as ls).
    let task_timeout = Duration::from_secs(10);
    let size = match with_timeout(task_timeout, GET_RESULT.wait()).await {
        Ok(GetReport::Ok { size }) => size,
        Ok(GetReport::NotFound) => {
            let mut buf = [0u8; 64];
            let mut w = WriteBuf::new(&mut buf);
            let _ = write!(w, "{}: NOT FOUND /{}\r\n", LABEL, name_str);
            return write_all(class, w.as_slice()).await;
        }
        Ok(GetReport::Failed(stage)) => {
            let mut buf = [0u8; 256];
            let mut w = WriteBuf::new(&mut buf);
            let _ = write!(w, "{}: FAILED — {}\r\n", LABEL, blackbox_fat_reason(stage));
            return write_all(class, w.as_slice()).await;
        }
        Ok(GetReport::Busy) => {
            return write_all(
                class,
                b"blackbox get: REJECTED - recorder is mid-flight (disarm first)\r\n",
            )
            .await;
        }
        Err(_) => {
            let mut buf = [0u8; 96];
            let mut w = WriteBuf::new(&mut buf);
            let _ = write!(
                w,
                "{}: TIMEOUT — task didn't respond in {} ms\r\n",
                LABEL,
                task_timeout.as_millis(),
            );
            return write_all(class, w.as_slice()).await;
        }
    };

    // Header, then raw payload until End/Err.
    {
        let mut buf = [0u8; 32];
        let mut w = WriteBuf::new(&mut buf);
        let _ = write!(w, "OK {}\r\n", size);
        write_all(class, w.as_slice()).await?;
    }
    // Per-chunk gap timeout can be short — the task is already
    // mid-file, so 5 s only ever fires if it died or the card hung.
    let chunk_timeout = Duration::from_secs(5);
    loop {
        match with_timeout(chunk_timeout, GET_STREAM.receive()).await {
            Ok(GetChunk::Data { buf, len }) => {
                write_all(class, &buf[..usize::from(len)]).await?;
            }
            Ok(GetChunk::End { crc32 }) => {
                let mut buf = [0u8; 32];
                let mut w = WriteBuf::new(&mut buf);
                let _ = write!(w, "\r\nCRC {:08x}\r\n", crc32);
                return write_all(class, w.as_slice()).await;
            }
            Ok(GetChunk::Err(stage)) => {
                let mut buf = [0u8; 256];
                let mut w = WriteBuf::new(&mut buf);
                let _ = write!(w, "\r\nERR {}\r\n", blackbox_fat_reason(stage));
                return write_all(class, w.as_slice()).await;
            }
            Err(_) => {
                return write_all(class, b"\r\nERR task stalled\r\n").await;
            }
        }
    }
}

/// `blackbox rm <file>` — delete a FAT-root file (e.g. a pulled
/// flight log) to free card space. Routed through the blackbox task
/// like ls/get because that task owns the SDMMC peripheral; refused
/// while recording.
async fn dispatch_blackbox_rm<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
    line: &str,
) -> Result<(), EndpointError> {
    use crate::blackbox::{RmReport, RM_REQUEST, RM_RESULT};

    const LABEL: &str = "blackbox rm";

    if !bsp::HAS_BLACKBOX_STORAGE {
        return write_all(class, b"blackbox rm: no storage backend on this board\r\n").await;
    }

    let mut parts = line.split_ascii_whitespace();
    let _ = parts.next(); // "blackbox"
    let _ = parts.next(); // "rm"
    let (Some(name_str), None) = (parts.next(), parts.next()) else {
        return write_all(class, b"usage: blackbox rm <file>\r\n").await;
    };
    let mut name: crate::blackbox::fat::EntryName = heapless::String::new();
    if name.push_str(name_str.trim_start_matches('/')).is_err() {
        return write_all(class, b"blackbox rm: name too long (max 32 chars)\r\n").await;
    }

    if crate::blackbox::should_record() {
        return write_all(
            class,
            b"blackbox rm: REJECTED - recorder is mid-flight (disarm first)\r\n",
        )
        .await;
    }

    let _ = RM_RESULT.try_take();
    RM_REQUEST.signal(name);

    let task_timeout = Duration::from_secs(10);
    match with_timeout(task_timeout, RM_RESULT.wait()).await {
        Ok(RmReport::Ok) => {
            let mut buf = [0u8; 64];
            let mut w = WriteBuf::new(&mut buf);
            let _ = write!(w, "{}: removed /{}\r\n", LABEL, name_str);
            write_all(class, w.as_slice()).await
        }
        Ok(RmReport::NotFound) => {
            let mut buf = [0u8; 64];
            let mut w = WriteBuf::new(&mut buf);
            let _ = write!(w, "{}: NOT FOUND /{}\r\n", LABEL, name_str);
            write_all(class, w.as_slice()).await
        }
        Ok(RmReport::Failed(stage)) => {
            let mut buf = [0u8; 256];
            let mut w = WriteBuf::new(&mut buf);
            let _ = write!(w, "{}: FAILED — {}\r\n", LABEL, blackbox_fat_reason(stage));
            write_all(class, w.as_slice()).await
        }
        Ok(RmReport::Busy) => {
            write_all(
                class,
                b"blackbox rm: REJECTED - recorder is mid-flight (disarm first)\r\n",
            )
            .await
        }
        Err(_) => {
            let mut buf = [0u8; 96];
            let mut w = WriteBuf::new(&mut buf);
            let _ = write!(
                w,
                "{}: TIMEOUT — task didn't respond in {} ms\r\n",
                LABEL,
                task_timeout.as_millis(),
            );
            write_all(class, w.as_slice()).await
        }
    }
}

/// `blackbox clean` — delete every `flight_NNNN.mcap` in the FAT root
/// in one card session. Other files on the card are left alone (see
/// `fat::is_flight_log`). Resets the in-memory sequence cache so the
/// next log is `flight_0001.mcap` again.
async fn dispatch_blackbox_clean<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
) -> Result<(), EndpointError> {
    use crate::blackbox::{CleanReport, CLEAN_REQUEST, CLEAN_RESULT};

    const LABEL: &str = "blackbox clean";

    if !bsp::HAS_BLACKBOX_STORAGE {
        return write_all(class, b"blackbox clean: no storage backend on this board\r\n").await;
    }
    if crate::blackbox::should_record() {
        return write_all(
            class,
            b"blackbox clean: REJECTED - recorder is mid-flight (disarm first)\r\n",
        )
        .await;
    }

    let _ = CLEAN_RESULT.try_take();
    CLEAN_REQUEST.signal(());

    // One directory rescan + one delete per file; a card with dozens
    // of large logs can take a few seconds on the FAT-table writes.
    let task_timeout = Duration::from_secs(30);
    match with_timeout(task_timeout, CLEAN_RESULT.wait()).await {
        Ok(CleanReport::Ok(s)) => {
            let mut buf = [0u8; 128];
            let mut w = WriteBuf::new(&mut buf);
            if s.aborted {
                let _ = write!(
                    w,
                    "{}: ABORTED after removing {} flight log(s) — a delete failed, run 'blackbox ls'\r\n",
                    LABEL, s.removed,
                );
            } else {
                let _ = write!(
                    w,
                    "{}: removed {} flight log(s); numbering restarts at flight_0001\r\n",
                    LABEL, s.removed,
                );
            }
            write_all(class, w.as_slice()).await
        }
        Ok(CleanReport::Failed(stage)) => {
            let mut buf = [0u8; 256];
            let mut w = WriteBuf::new(&mut buf);
            let _ = write!(w, "{}: FAILED — {}\r\n", LABEL, blackbox_fat_reason(stage));
            write_all(class, w.as_slice()).await
        }
        Ok(CleanReport::Busy) => {
            write_all(
                class,
                b"blackbox clean: REJECTED - recorder is mid-flight (disarm first)\r\n",
            )
            .await
        }
        Err(_) => {
            let mut buf = [0u8; 96];
            let mut w = WriteBuf::new(&mut buf);
            let _ = write!(
                w,
                "{}: TIMEOUT — task didn't respond in {} ms\r\n",
                LABEL,
                task_timeout.as_millis(),
            );
            write_all(class, w.as_slice()).await
        }
    }
}

/// `postmortem show` — print the prior-boot crash record.
///
/// Reads BKPSRAM directly via [`crate::postmortem::recovery::current_boot_record`]
/// (idempotent — the Signal-based `take_pending` is reserved for the
/// blackbox recorder's single-shot mirror). Prints reset cause, fatal
/// kind + register summary, last-N events, and the most recent
/// snapshot.
#[cfg(feature = "postmortem")]
async fn dispatch_postmortem_show<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
) -> Result<(), EndpointError> {
    use crate::postmortem::record::{self as pmrec, FatalKind};
    use crate::postmortem::recovery;
    use crate::postmortem::reset_cause;

    let rec = recovery::current_boot_record();
    let valid = pmrec::is_valid(&rec);

    let mut buf = [0u8; 256];
    let mut w = WriteBuf::new(&mut buf);
    if !valid {
        let cause = reset_cause::read();
        let _ = write!(
            w,
            "postmortem: no valid record (current boot's slot, reset_cause=0x{:08x} kind={:?})\r\n",
            cause,
            reset_cause::classify(cause)
        );
        return write_all(class, w.as_slice()).await;
    }
    let cause_kind = reset_cause::classify(rec.header.reset_cause);
    let fatal_kind = FatalKind::from_u8(rec.fatal.kind);
    let _ = write!(
        w,
        "postmortem: boot {} (reset_cause=0x{:08x} {:?}, fw=0x{:08x}, uptime={}ms)\r\n",
        rec.header.boot_count,
        rec.header.reset_cause,
        cause_kind,
        rec.header.fw_git_hash,
        rec.header.uptime_ms,
    );
    write_all(class, w.as_slice()).await?;

    if !matches!(fatal_kind, FatalKind::None) {
        let mut buf = [0u8; 256];
        let mut w = WriteBuf::new(&mut buf);
        let _ = write!(
            w,
            "  fatal: {:?}  pc=0x{:08x} lr=0x{:08x} psr=0x{:08x}\r\n\
             \x20        cfsr=0x{:08x} hfsr=0x{:08x} mmfar=0x{:08x} bfar=0x{:08x}\r\n",
            fatal_kind,
            rec.fatal.pc,
            rec.fatal.lr,
            rec.fatal.psr,
            rec.fatal.cfsr,
            rec.fatal.hfsr,
            rec.fatal.mmfar,
            rec.fatal.bfar,
        );
        write_all(class, w.as_slice()).await?;
    }

    // Events: walk in chronological order, print one line each. Stop
    // after a sensible cap so a wedged ring doesn't flood the shell.
    let mut count = 0u32;
    for ev in rec.events_in_order() {
        let mut buf = [0u8; 96];
        let mut w = WriteBuf::new(&mut buf);
        let _ = write!(
            w,
            "  ev[{}] @{}ms kind=0x{:02x} data=0x{:08x}\r\n",
            ev.seq, ev.timestamp_ms, ev.kind, ev.data,
        );
        write_all(class, w.as_slice()).await?;
        count += 1;
        if count >= pmrec::EVENT_RING_LEN as u32 {
            break;
        }
    }
    if count == 0 {
        write_all(class, b"  (no events captured this boot)\r\n").await?;
    }

    // Snapshot summary — single line.
    let mut buf = [0u8; 192];
    let mut w = WriteBuf::new(&mut buf);
    let q = rec.snapshots.attitude_quat_wijk;
    let p = rec.snapshots.position_xyz;
    let v = rec.snapshots.velocity_xyz;
    let _ = write!(
        w,
        "  last: q=[{:.3},{:.3},{:.3},{:.3}] p=[{:.2},{:.2},{:.2}] v=[{:.2},{:.2},{:.2}]\r\n",
        q[0], q[1], q[2], q[3], p[0], p[1], p[2], v[0], v[1], v[2],
    );
    write_all(class, w.as_slice()).await
}

/// `postmortem clear` — wipe the BKPSRAM slot.
///
/// Used after a developer has inspected `postmortem show` and wants
/// to ensure the next boot starts clean. The next boot will still
/// see this boot's reset_cause via `RCC.RSR`, but the slot itself
/// will be invalid (magic = 0).
#[cfg(feature = "postmortem")]
async fn dispatch_postmortem_clear<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
) -> Result<(), EndpointError> {
    crate::postmortem::bkpsram::with_record_mut(|r| {
        crate::postmortem::record::clear(r);
    });
    write_all(class, b"postmortem: BKPSRAM slot cleared\r\n").await
}

fn blackbox_fat_reason(stage: crate::blackbox::fat::OpError) -> &'static str {
    use crate::blackbox::fat::OpError;
    match stage {
        OpError::NoSubscriberSlot => {
            "topic has no free subscriber slot (bump SUBS on the channel)"
        }
        OpError::NoMessageInTimeout => {
            "no fresh message on the topic within the op's timeout \
             (sensor driver stalled or topic silent?)"
        }
        OpError::CardAcquire => {
            "card acquire failed (CMD0/ACMD41 — card missing/unseated/locked?)"
        }
        OpError::NoPartitionTable => {
            "sector 0 has neither MBR signature nor FAT BPB \
             (card blank/GPT/corrupted; reformat as FAT32)"
        }
        OpError::NoFatPartition => {
            "MBR present but no FAT partition entry \
             (card formatted as exFAT/ext4/NTFS; reformat as FAT32)"
        }
        OpError::PartitionIo => "MBR/partition I/O fault (see defmt log)",
        OpError::Mount => {
            "FAT boot sector unreadable \
             (partition exists but BPB corrupted; reformat as FAT32)"
        }
        OpError::Create => {
            "directory entry create failed (root full, illegal name, or write-protected?)"
        }
        OpError::Write => "write/flush failed (disk full or I/O fault; see defmt log)",
        OpError::PartialWrite => {
            "write faulted mid-session, but the partial log was flushed and closed \
             (file is readable up to the fault; see defmt log)"
        }
        OpError::Unmount => {
            "unmount-flush failed (file may not have hit the card; see defmt log)"
        }
        OpError::NotFound => "file not found in FAT root",
        OpError::Open => "open failed (see defmt log)",
        OpError::Read => {
            "read faulted mid-stream (I/O fault, or file shorter than its \
             directory entry; see defmt log)"
        }
        OpError::Remove => "remove failed (see defmt log)",
        OpError::Aborted => "transfer aborted (armed mid-transfer, or shell stopped draining)",
    }
}

async fn dispatch_motor<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
    line: &str,
) -> Result<(), EndpointError> {
    let mut parts = line.split_ascii_whitespace();
    parts.next(); // skip "motor"
    let idx_str = parts.next();
    let pct_str = parts.next();

    let (idx_str, pct_str) = match (idx_str, pct_str) {
        (Some(i), Some(p)) => (i, p),
        _ => return write_all(class, b"usage: motor <1-4> <0-100>\r\n").await,
    };

    let idx: u8 = match idx_str.parse() {
        Ok(v) if (1..=4).contains(&v) => v,
        _ => return write_all(class, b"motor index must be 1-4\r\n").await,
    };

    let pct: u8 = match pct_str.parse() {
        Ok(v) if v <= 100 => v,
        _ => return write_all(class, b"throttle must be 0-100\r\n").await,
    };

    let mut motor_commands: [msgs::NormalizedThrottle; 4] =
        [msgs::NormalizedThrottle::new_saturating(0.0); 4];

    motor_commands[(idx - 1) as usize] =
        msgs::NormalizedThrottle::new_saturating(pct as f32 / 100.0);
    ACTUATOR_MOTORS.signal(msgs::ActuatorMotors {
        timestamp: Instant::now(),
        motor_commands,
    });

    let mut buf = [0u8; 64];
    let mut w = WriteBuf::new(&mut buf);
    write!(w, "motor {} = {}%\r\n", idx, pct).ok();
    write_all(class, w.as_slice()).await
}

/// Compile-time build selections, as `(key, value)` pairs for `param list`.
///
/// These mirror the vehicle YAML's `build:` section plus the baked identity.
/// They are deliberately *not* registry parameters — a cargo feature cannot
/// be changed at runtime — but they belong in the same listing, because the
/// question "why is this board behaving like a different vehicle?" is
/// answered here, not in the tunables.
///
/// Values are `&'static str` so this stays a const-ish table with no
/// formatting state; `imu_rate` carries its resolved ODR because the feature
/// name alone (`8khz` vs `1khz`) is the input, not the effective rate.
fn build_facts_iter() -> impl Iterator<Item = (&'static str, &'static str)> {
    const BOARD: &str = if cfg!(feature = "board_sakurah743") {
        "sakurah743"
    } else if cfg!(feature = "board_foxeerh743") {
        "foxeerh743"
    } else if cfg!(feature = "board_micoair743v2") {
        "micoair743v2"
    } else {
        "<none>"
    };
    const POS_SOURCE: &str = if cfg!(feature = "est_pos_mocap") {
        "mocap"
    } else if cfg!(feature = "est_pos_gps") {
        "gps"
    } else {
        "<none>"
    };
    // outer_mpc_full implies outer_mpc (Cargo.toml), so the full model
    // must be tested first or every mpc_full build reports plain "mpc".
    const OUTER_LOOP: &str = if cfg!(feature = "outer_mpc_full") {
        "mpc_full"
    } else if cfg!(feature = "outer_mpc") {
        "mpc"
    } else if cfg!(feature = "outer_geometric") {
        "cascade (outer_geometric)"
    } else if cfg!(feature = "outer_rate") {
        "rate"
    } else {
        "<none>"
    };
    // Which mission-planning schema is compiled in. A vehicle that
    // rejects a mission the YAML clearly declares is answered here: the
    // offline build has no solver at all, so an unbaked mission simply
    // does not exist for it, whereas the online build can refuse one on
    // solver grounds (duration window, convergence).
    const MISSION_PLANNER: &str = if cfg!(feature = "plan_online") {
        "online (on-device BFGS)"
    } else {
        "offline (baked schedule)"
    };
    const RC_PROTOCOL: &str = if cfg!(feature = "rx_crsf") {
        "crsf"
    } else if cfg!(feature = "rx_ghst") {
        "ghst"
    } else {
        "<none>"
    };
    const IMU_RATE: &str = if cfg!(feature = "imu_1khz") {
        "1khz (1000 Hz ODR, low-noise)"
    } else {
        "8khz (8000 Hz ODR)"
    };
    // Which inner-loop law is compiled in. Worth a line of its own: a
    // vehicle that flies "differently than the tune suggests" is exactly
    // the question this listing exists to answer.
    const INDI: &str = if cfg!(feature = "indi_off") {
        "no (proportional rate controller, no increment)"
    } else {
        "yes"
    };
    const GPS_MODEL: &str = if cfg!(feature = "gps_unicore") {
        "unicore (UM982)"
    } else {
        "ublox"
    };
    // Antenna count is a separate fact from the receiver model: a UM982 is
    // heading-capable but can be wired single-antenna. The vehicle-yaml
    // bake rejects `yes` without unicore, but a hand-composed feature set
    // (the check-all matrix) can still reach that state, and this listing
    // is exactly where someone goes to find out why a heading never
    // arrives — so report the inconsistency instead of claiming fusion.
    const GPS_DUAL_ANTENNA: &str = if cfg!(feature = "gps_dual_antenna") {
        if cfg!(feature = "gps_unicore") {
            "yes (ANT2 fitted)"
        } else {
            "yes (ANT2 fitted, but driver has no heading!)"
        }
    } else {
        "no"
    };
    const POSTMORTEM: &str = if cfg!(feature = "postmortem") {
        "on"
    } else {
        "off"
    };
    const DEV_TELEM: &str = if cfg!(feature = "dev_telem") {
        "on"
    } else {
        "off"
    };

    [
        ("vehicle", crate::vehicle::BAKED_VEHICLE),
        (
            "airframe_name",
            match crate::vehicle::BAKED_AIRFRAME_NAME {
                Some(n) => n,
                None => "<unset>",
            },
        ),
        ("board", BOARD),
        ("pos_source", POS_SOURCE),
        ("outer_loop", OUTER_LOOP),
        ("mission_planner", MISSION_PLANNER),
        (
            "mpc_cost_policy",
            match crate::vehicle::BAKED_COST_POLICY_NAME {
                Some(n) => n,
                None => "<none>",
            },
        ),
        ("rc_protocol", RC_PROTOCOL),
        ("imu_rate", IMU_RATE),
        ("indi", INDI),
        ("gps_model", GPS_MODEL),
        ("gps_dual_antenna", GPS_DUAL_ANTENNA),
        ("postmortem", POSTMORTEM),
        ("dev_telem", DEV_TELEM),
    ]
    .into_iter()
}

async fn dispatch_param<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
    line: &str,
) -> Result<(), EndpointError> {
    use cybflight_core::param_registry::{ParamGroup, ParamName};
    use cybflight_core::params::{FirmwareConfig, PARAM_COUNT};

    let mut parts = line.split_ascii_whitespace();
    parts.next(); // skip "param"
    let a1 = parts.next().unwrap_or("");

    // Two spellings for the same thing:
    //   `param list [group]`   — verb first, group optional
    //   `param <group> list`   — group first, which reads better when you
    //                            are drilling into one subtree repeatedly
    // Look ahead on a separate iterator so `parts` stays positioned for the
    // other verbs (`get <name>`, `set <name> <v>`, `diff --yaml`, …).
    let mut look = line.split_ascii_whitespace();
    look.next(); // "param"
    look.next(); // a1
    let group_first = look.next() == Some("list") && a1 != "list";
    let (sub, group_arg) = if group_first {
        ("list", Some(a1))
    } else {
        (a1, None)
    };

    // Refuse mutations while armed, mirroring the armed-guard precedent of
    // `blackbox set` / `mission set` / `led`: `save` triggers a same-bank
    // flash erase that stalls the CPU bus for ~1–2 s (dangerous mid-flight),
    // and `set`/`defaults` would silently defer anyway since consumers only
    // hot-reload while disarmed. `list`/`get` stay available.
    if matches!(sub, "set" | "save" | "defaults" | "reset")
        && crate::motors::IS_ARMED.load(Ordering::Acquire)
    {
        return write_all(class, b"refused: disarm before modifying params\r\n").await;
    }

    match sub {
        "list" => {
            // Group filter: from the group-first form, else the token after
            // `list`. Empty = everything.
            let filter = group_first
                .then_some(group_arg)
                .flatten()
                .or_else(|| parts.next())
                .filter(|s| !s.is_empty());

            let mut shown = 0u32;

            // ── Build facts ──
            // Compile-time selections are not registry parameters — they are
            // cargo features and baked constants, so they can never be `set`.
            // They are listed anyway because they are the first thing you
            // want when a board is behaving unlike the vehicle you think it
            // is, and until now the only way to see them was to guess from
            // the firmware you believed you flashed.
            if filter.is_none_or(|f| ParamName::path_matches("build", f)) {
                write_all(class, b"[build]  (compile-time, read-only)\r\n").await?;
                for (k, v) in build_facts_iter() {
                    let mut buf = [0u8; 96];
                    let mut w = WriteBuf::new(&mut buf);
                    write!(w, "  {k:30} = {v}\r\n").ok();
                    write_all(class, w.as_slice()).await?;
                    shown += 1;
                }
            }

            // ── Registry parameters, grouped ──
            // Group members are contiguous by index (guaranteed by the
            // derive's layout and asserted in `params::path_tests`), so a
            // header is emitted whenever the path changes — no buffering,
            // no second pass.
            let params = crate::params::get();
            let mut prev_path: Option<ParamName> = None;
            for idx in 0..PARAM_COUNT {
                let (Some(name), Some(val), Some(path)) = (
                    ParamName::of::<FirmwareConfig>(idx),
                    params.param_get(idx),
                    ParamName::path_of::<FirmwareConfig>(idx),
                ) else {
                    continue;
                };
                if let Some(f) = filter
                    && !ParamName::path_matches(path.as_str(), f)
                {
                    continue;
                }
                if prev_path.as_ref().map(ParamName::as_str) != Some(path.as_str()) {
                    let mut buf = [0u8; 64];
                    let mut w = WriteBuf::new(&mut buf);
                    let shown_path = if path.as_str().is_empty() {
                        "(root)"
                    } else {
                        path.as_str()
                    };
                    write!(w, "[{shown_path}]\r\n").ok();
                    write_all(class, w.as_slice()).await?;
                    prev_path = Some(path);
                }

                let meta = FirmwareConfig::param_meta(idx);
                let mut buf = [0u8; 96];
                let mut w = WriteBuf::new(&mut buf);
                write!(w, "  {:30} = {}", name.as_str(), val).ok();
                if !meta.unit.is_empty() {
                    write!(w, " [{}]", meta.unit).ok();
                }
                // Reboot-flagged params ARE settable; they just don't take
                // effect until the next boot. Worth marking inline so nobody
                // sets one, sees no change, and concludes it didn't work.
                if meta.reboot {
                    write!(w, " (reboot)").ok();
                }
                write!(w, "\r\n").ok();
                write_all(class, w.as_slice()).await?;
                shown += 1;
            }

            if shown == 0 {
                let mut buf = [0u8; 96];
                let mut w = WriteBuf::new(&mut buf);
                write!(
                    w,
                    "no group matched {:?}. available groups:\r\n",
                    filter.unwrap_or("")
                )
                .ok();
                write_all(class, w.as_slice()).await?;
                write_all(class, b"  build\r\n").await?;
                let mut prev: Option<ParamName> = None;
                for idx in 0..PARAM_COUNT {
                    let Some(path) = ParamName::path_of::<FirmwareConfig>(idx) else {
                        continue;
                    };
                    if prev.as_ref().map(ParamName::as_str) == Some(path.as_str()) {
                        continue;
                    }
                    let mut buf = [0u8; 64];
                    let mut w = WriteBuf::new(&mut buf);
                    write!(w, "  {}\r\n", path.as_str()).ok();
                    write_all(class, w.as_slice()).await?;
                    prev = Some(path);
                }
            } else {
                let mut buf = [0u8; 64];
                let mut w = WriteBuf::new(&mut buf);
                write!(w, "({shown} shown)\r\n").ok();
                write_all(class, w.as_slice()).await?;
            }
        }
        // Overrides relative to the baked (vehicle-YAML) defaults.
        // `param diff --yaml` prints them as a `tuning:`-section snippet,
        // directly mergeable into vehicles/<VEHICLE>.yaml (the format the
        // host-side `just param-sync` consumes).
        "diff" => {
            let yaml = parts.next() == Some("--yaml");
            let params = crate::params::get();
            let baked = crate::vehicle::default_params();
            if yaml {
                // Identity handshake for the host-side merge tool: the
                // vehicle this firmware was baked for.
                let mut buf = [0u8; 64];
                let mut w = WriteBuf::new(&mut buf);
                write!(w, "# vehicle: {}\r\n", crate::vehicle::BAKED_VEHICLE).ok();
                write_all(class, w.as_slice()).await?;
            }
            let mut count = 0u32;
            for idx in 0..PARAM_COUNT {
                let (Some(cur), Some(bak)) = (params.param_get(idx), baked.param_get(idx))
                else {
                    continue;
                };
                if cur.as_f32().to_le_bytes() == bak.as_f32().to_le_bytes() {
                    continue;
                }
                count += 1;
                let Some(name) = ParamName::of::<FirmwareConfig>(idx) else {
                    continue;
                };
                let mut buf = [0u8; 96];
                let mut w = WriteBuf::new(&mut buf);
                if yaml {
                    write!(w, "  {}: {}\r\n", name.as_str(), cur).ok();
                } else {
                    write!(w, "  {:28} = {}  (baked {})\r\n", name.as_str(), cur, bak).ok();
                }
                write_all(class, w.as_slice()).await?;
            }
            if count == 0 {
                write_all(class, b"  (no overrides - matches baked defaults)\r\n").await?;
            }
        }
        // Revert one parameter (or everything) to the baked defaults.
        // In-memory only, like `set` - follow with `param save --prune` to
        // persist. Plain `param save` also works for *this* boot, but it
        // records the baked value rather than dropping the key, so the key
        // stays pinned against the next vehicle-YAML edit; `--prune`
        // removes it from the store.
        "reset" => match parts.next() {
            Some("all") => {
                let defaults = crate::vehicle::default_params();
                #[cfg(feature = "outer_mpc")]
                crate::control::offline_mission::init_active_from_index(
                    defaults.trajectory.mission_profile,
                );
                crate::params::set(defaults);
                write_all(class, b"all params reset to baked defaults (not saved)\r\n").await?;
            }
            Some(name) => match <FirmwareConfig as ParamGroup>::param_find(name) {
                Some(idx) => {
                    let baked = crate::vehicle::default_params();
                    let v = baked.param_get(idx).map_or(0.0, |v| v.as_f32());
                    let mut params = crate::params::get();
                    params.param_set_f32(idx, v);
                    let stored = params.param_get(idx);
                    crate::params::set(params);
                    let mut buf = [0u8; 64];
                    let mut w = WriteBuf::new(&mut buf);
                    match stored {
                        Some(v) => write!(w, "{} = {} (baked)\r\n", name, v).ok(),
                        None => write!(w, "{} reset\r\n", name).ok(),
                    };
                    write_all(class, w.as_slice()).await?;
                }
                None => {
                    write_all(class, b"unknown param (try 'param list')\r\n").await?;
                }
            },
            None => {
                write_all(class, b"usage: param reset <name>|all\r\n").await?;
            }
        },
        "get" => {
            let name = parts.next().unwrap_or("");
            match crate::params::get().get_named(name) {
                Some(val) => {
                    let mut buf = [0u8; 64];
                    let mut w = WriteBuf::new(&mut buf);
                    write!(w, "{} = {}\r\n", name, val).ok();
                    write_all(class, w.as_slice()).await?;
                }
                None => {
                    write_all(class, b"unknown param (try 'param list')\r\n").await?;
                }
            }
        }
        "set" => {
            let name = parts.next();
            let val_str = parts.next();
            match (name, val_str) {
                (Some(name), Some(val_str)) => {
                    // Stage-7 guard (mirrors Betaflight's
                    // blackboxMayEditConfig): the recorder snapshots
                    // blackbox_* at session start, so a mid-session
                    // edit silently defers — reject for clarity. The
                    // armed case is already covered above; this
                    // catches bench sessions via RECORDER_HOLD.
                    if name.starts_with("blackbox_") && crate::blackbox::should_record() {
                        return write_all(
                            class,
                            b"refused: recording session active - \
                              'blackbox record off' before editing blackbox_* params\r\n",
                        )
                        .await;
                    }
                    if let Some(val) = parse_f32(val_str) {
                        let idx = <FirmwareConfig as ParamGroup>::param_find(name);
                        if let Some(idx) = idx {
                            let meta = FirmwareConfig::param_meta(idx);
                            if !meta.in_range(val) {
                                let mut buf = [0u8; 96];
                                let mut w = WriteBuf::new(&mut buf);
                                write!(
                                    w,
                                    "rejected: {} must be in [{}, {}]{}{}\r\n",
                                    name,
                                    meta.min,
                                    meta.max,
                                    if meta.unit.is_empty() { "" } else { " " },
                                    meta.unit,
                                )
                                .ok();
                                return write_all(class, w.as_slice()).await;
                            }
                        }
                        let mut params = crate::params::get();
                        if params.set_named(name, val) {
                            // Echo the value as stored (typed conversion may
                            // have saturated/truncated the input).
                            let stored = params.get_named(name);
                            let reboot = idx
                                .map(|i| FirmwareConfig::param_meta(i).reboot)
                                .unwrap_or(false);
                            crate::params::set(params);
                            let mut buf = [0u8; 96];
                            let mut w = WriteBuf::new(&mut buf);
                            let note = if reboot { "  (reboot required)" } else { "" };
                            match stored {
                                Some(v) => write!(w, "{} = {}{}\r\n", name, v, note).ok(),
                                None => write!(w, "{} set{}\r\n", name, note).ok(),
                            };
                            write_all(class, w.as_slice()).await?;
                        } else {
                            write_all(class, b"unknown param (try 'param list')\r\n").await?;
                        }
                    } else {
                        write_all(class, b"invalid number\r\n").await?;
                    }
                }
                _ => {
                    write_all(class, b"usage: param set <name> <value>\r\n").await?;
                }
            }
        }
        // `param save` appends; `param save --prune` rewrites the store as
        // exactly the current diff-vs-baked set. The prune form is what
        // makes `param reset` stick: without it, reverting a key appends a
        // record holding today's baked value, which shadows the *next*
        // vehicle-YAML edit of that key (the log has no tombstone record).
        "save" => {
            let prune = parts.next() == Some("--prune");
            let outcome = if prune {
                crate::params::prune_to_flash()
            } else {
                crate::params::save_to_flash().map(|()| 0)
            };
            match outcome {
                Ok(kept) if prune => {
                    let mut buf = [0u8; 64];
                    let mut w = WriteBuf::new(&mut buf);
                    write!(w, "store pruned to {} override(s)\r\n", kept).ok();
                    write_all(class, w.as_slice()).await?;
                }
                Ok(_) => {
                    write_all(class, b"params saved to flash\r\n").await?;
                }
                Err(e) => {
                    let mut buf = [0u8; 64];
                    let mut w = WriteBuf::new(&mut buf);
                    write!(w, "error: {}\r\n", e).ok();
                    write_all(class, w.as_slice()).await?;
                }
            }
        }
        "defaults" => {
            let defaults = crate::vehicle::default_params();
            // Mirror the default mission_profile into the live atomic so
            // a `mission get` after `param defaults` sees the reset, not
            // the previous selection. (Without this, the atomic only
            // resyncs on the next reboot via `init_from_flash`.)
            #[cfg(feature = "outer_mpc")]
            crate::control::offline_mission::init_active_from_index(defaults.trajectory.mission_profile);
            crate::params::set(defaults);
            write_all(class, b"params reset to defaults (not saved)\r\n").await?;
        }
        _ => {
            write_all(
                class,
                b"usage: param list [group] | param <group> list\r\n\
                  \x20      param get|set|diff [--yaml]|reset <name>|all|save [--prune]|defaults\r\n\
                  \x20      'save' appends; 'save --prune' rewrites the store as exactly the\r\n\
                  \x20      current diff vs baked - use it after 'reset' so a YAML edit sticks.\r\n\
                  groups are the config tree: build, airframe.body, eskf.mocap_guard, \
                  indi, mpc, ...\r\n\
                  a group name matches its whole subtree ('eskf' includes \
                  eskf.filter, eskf.faults, ...);\r\n\
                  '.', '-' and '_' are interchangeable, so 'eskf-mocap_guard' \
                  works. 'param list <bad>' lists them.\r\n",
            )
            .await?;
        }
    }
    Ok(())
}

#[cfg(feature = "outer_mpc")]
async fn dispatch_mission<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
    line: &str,
) -> Result<(), EndpointError> {
    use crate::control::offline_mission;

    let mut parts = line.split_ascii_whitespace();
    parts.next(); // skip "mission"
    let sub = parts.next().unwrap_or("");

    match sub {
        "list" => {
            for (i, p) in offline_mission::PROFILES.iter().enumerate() {
                let active = i as u8 == offline_mission::active_index();
                // Last absolute timestamp = total trajectory duration.
                // The registry's compile-time guard asserts `n >= 1`.
                let dur_s = p.timestamps[p.timestamps.len() - 1];
                let mut buf = [0u8; 128];
                let mut w = WriteBuf::new(&mut buf);
                write!(
                    w,
                    "  {} {}: {} ({} {} {}, {}s)\r\n",
                    if active { '*' } else { ' ' },
                    i,
                    p.name,
                    p.env,
                    p.variant,
                    p.speed,
                    dur_s,
                )
                .ok();
                write_all(class, w.as_slice()).await?;
            }
        }
        "get" => {
            use crate::control::{MissionState, MISSION_STATE};
            use offline_mission::FlatnessMap;

            let p = offline_mission::active();
            let idx = offline_mission::active_index();
            // Last absolute timestamp = total duration (registry guards n >= 1).
            let dur_s = p.timestamps[p.timestamps.len() - 1];
            // Waypoint polyline from the YAML start. The flown MINCO curve
            // passes through every waypoint, so it is at least this long —
            // a lower bound on path length and on average speed.
            let mut prev = Vector3::from(p.start_pos);
            let mut path_m = 0.0f32;
            for wp in p.waypoints {
                let v = Vector3::from(*wp);
                path_m += (v - prev).norm();
                prev = v;
            }
            let end = p.waypoints[p.waypoints.len() - 1];
            let state = match MissionState::from_u8(MISSION_STATE.load(Ordering::Acquire)) {
                MissionState::Idle => "Idle",
                MissionState::Planning => "Planning",
                MissionState::Executing => "Executing",
            };

            let mut buf = [0u8; 160];
            let mut w = WriteBuf::new(&mut buf);
            write!(
                w,
                "{} (idx={}{}) state={}\r\n  env/variant/speed: {} / {} / {}\r\n",
                p.name,
                idx,
                if idx == offline_mission::DEFAULT_PROFILE_INDEX { ", vehicle default" } else { "" },
                state,
                p.env,
                p.variant,
                p.speed,
            )
            .ok();
            write_all(class, w.as_slice()).await?;

            let mut buf = [0u8; 160];
            let mut w = WriteBuf::new(&mut buf);
            write!(
                w,
                "  schedule: {} pieces, {:.3} s, path >= {:.1} m (avg >= {:.2} m/s)\r\n",
                p.num_pieces(),
                dur_s,
                path_m,
                path_m / dur_s,
            )
            .ok();
            write_all(class, w.as_slice()).await?;

            // Which head pose the offline solve actually flies from. By
            // default it is the live setpoint, so the YAML start is only
            // the reference the waypoints were planned against.
            #[cfg(not(feature = "plan_online"))]
            {
                let mut buf = [0u8; 160];
                let mut w = WriteBuf::new(&mut buf);
                let [sx, sy, sz] = p.start_pos;
                if crate::control::mission_planner::OFFLINE_USE_YAML_START {
                    write!(w, "  head: YAML start [{:.3}, {:.3}, {:.3}]\r\n", sx, sy, sz).ok();
                } else {
                    write!(
                        w,
                        "  head: live setpoint (YAML start [{:.3}, {:.3}, {:.3}] not flown)\r\n",
                        sx, sy, sz,
                    )
                    .ok();
                }
                write_all(class, w.as_slice()).await?;
            }

            let mut buf = [0u8; 160];
            let mut w = WriteBuf::new(&mut buf);
            write!(w, "  end:  [{:.3}, {:.3}, {:.3}]\r\n", end[0], end[1], end[2]).ok();
            // Same precedence as the planner (`mission_planner.rs`, yaw
            // source): lookahead wins, then headings, else hold entry yaw.
            if p.lookahead {
                write!(
                    w,
                    "  yaw: lookahead ON, dt {:.3} s, max rate {:.2} rad/s\r\n",
                    p.yaw_lookahead_dt_s, p.yaw_lookahead_max_rate_rad_s,
                )
                .ok();
            } else if let Some(h) = p.headings {
                write!(w, "  yaw: lookahead off, spline through {} waypoint headings\r\n", h.len())
                    .ok();
            } else {
                write!(w, "  yaw: lookahead off, hold yaw latched at mission entry\r\n").ok();
            }
            write_all(class, w.as_slice()).await?;

            let mut buf = [0u8; 160];
            let mut w = WriteBuf::new(&mut buf);
            let fm = match p.flatness_map {
                FlatnessMap::TiltYaw => {
                    "tilt_yaw (robust through 90 deg tilt; heading drifts from psi as tilt grows)"
                }
                FlatnessMap::TrueYaw => {
                    "true_yaw (heading tracks psi exactly; singular at 90 deg roll)"
                }
            };
            write!(w, "  flatness_map: {}\r\n", fm).ok();
            write_all(class, w.as_slice()).await?;
        }
        "set" => {
            // Refuse while armed: changing the trajectory mid-flight (or even
            // mid-arming) would cause the next mission trigger to load a
            // different schedule than the operator vetted on the ground.
            if crate::motors::IS_ARMED.load(Ordering::Acquire) {
                write_all(class, b"refused: disarm before changing mission\r\n").await?;
                return Ok(());
            }
            // One arg = mission name (missions/<name>.yaml file stem);
            // three args = the legacy (env, variant, speed) triple.
            let a = parts.next();
            let b = parts.next();
            let c = parts.next();
            let found = match (a, b, c) {
                (Some(name), None, None) => offline_mission::find_by_name(name),
                (Some(env), Some(variant), Some(speed)) => {
                    offline_mission::find(env, variant, speed)
                }
                _ => {
                    return write_all(
                        class,
                        b"usage: mission set <name> | mission set <env> <variant> <speed>\r\n",
                    )
                    .await;
                }
            };
            match found {
                Some(idx) => {
                    if !offline_mission::set_active(idx) {
                        // The only way set_active rejects a valid index is an
                        // env mismatch with the build (BUILD_ENV is fixed at
                        // compile time by the position-source feature).
                        let p = offline_mission::PROFILES[idx as usize];
                        let mut buf = [0u8; 160];
                        let mut w = WriteBuf::new(&mut buf);
                        write!(
                            w,
                            "refused: profile '{}' env='{}' incompatible with build env '{}'\r\n",
                            p.name,
                            p.env,
                            offline_mission::BUILD_ENV,
                        )
                        .ok();
                        write_all(class, w.as_slice()).await?;
                        return Ok(());
                    }
                    let mut params = crate::params::get();
                    params.trajectory.mission_profile = idx;
                    crate::params::set(params);
                    let p = offline_mission::active();
                    let mut buf = [0u8; 128];
                    let mut w = WriteBuf::new(&mut buf);
                    write!(
                        w,
                        "mission = {} (idx={}) — run 'param save' to persist\r\n",
                        p.name, idx,
                    )
                    .ok();
                    write_all(class, w.as_slice()).await?;
                }
                None => {
                    write_all(
                        class,
                        b"unknown mission (try 'mission list')\r\n",
                    )
                    .await?;
                }
            }
        }
        _ => {
            write_all(
                class,
                b"usage: mission list|get|set <name>|set <env> <variant> <speed>\r\n",
            )
            .await?;
        }
    }
    Ok(())
}

/// `led on` / `led off`: toggle the external arm LED, mirror the change into
/// the live atomic so the LED task picks it up within one poll tick, and
/// auto-save the new value to flash. Refused while armed because flash erase
/// stalls the CPU for ~1–2 s.
async fn dispatch_led<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
    enable: bool,
) -> Result<(), EndpointError> {
    if crate::motors::IS_ARMED.load(Ordering::Acquire) {
        return write_all(class, b"refused: disarm before changing LED config\r\n").await;
    }

    let mut params = crate::params::get();
    params.system.arm_led_enabled = enable;
    crate::params::set(params);

    crate::arm_led::ARM_LED_ENABLED.store(enable, Ordering::Relaxed);
    crate::arm_led::ARM_LED_REFRESH.signal(());

    match crate::params::save_to_flash() {
        Ok(()) => {
            let msg: &[u8] = if enable {
                b"arm LED on, saved\r\n"
            } else {
                b"arm LED off, saved\r\n"
            };
            write_all(class, msg).await
        }
        Err(e) => {
            let mut buf = [0u8; 96];
            let mut w = WriteBuf::new(&mut buf);
            write!(w, "arm LED toggled but flash save failed: {}\r\n", e).ok();
            write_all(class, w.as_slice()).await
        }
    }
}

/// Minimal f32 parser for no_std (core::str::parse::<f32> requires std).
/// Handles optional sign, integer part, optional decimal fraction.
fn parse_f32(s: &str) -> Option<f32> {
    if s.is_empty() {
        return None;
    }
    // Optional exponent suffix (e/E, signed integer) — the smallest
    // params (noise densities ~1e-4) are exactly the ones users type in
    // scientific notation.
    let (mantissa, exp): (&str, i32) = match s.find(['e', 'E']) {
        Some(pos) => {
            let e = &s[pos + 1..];
            let (e, exp_neg) = if let Some(rest) = e.strip_prefix('-') {
                (rest, true)
            } else if let Some(rest) = e.strip_prefix('+') {
                (rest, false)
            } else {
                (e, false)
            };
            if e.is_empty() || e.len() > 3 || !e.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let mut v: i32 = 0;
            for b in e.bytes() {
                v = v * 10 + (b - b'0') as i32;
            }
            (&s[..pos], if exp_neg { -v } else { v })
        }
        None => (s, 0),
    };
    let (mantissa, neg) = if let Some(rest) = mantissa.strip_prefix('-') {
        (rest, true)
    } else if let Some(rest) = mantissa.strip_prefix('+') {
        (rest, false)
    } else {
        (mantissa, false)
    };
    let (int_part, frac_part) = match mantissa.find('.') {
        Some(dot) => (&mantissa[..dot], Some(&mantissa[dot + 1..])),
        None => (mantissa, None),
    };
    if int_part.is_empty() && frac_part.map_or(true, |f| f.is_empty()) {
        return None;
    }
    let mut val: f64 = 0.0;
    for &b in int_part.as_bytes() {
        if !b.is_ascii_digit() {
            return None;
        }
        val = val * 10.0 + (b - b'0') as f64;
    }
    if let Some(frac) = frac_part {
        let mut factor = 0.1;
        for &b in frac.as_bytes() {
            if !b.is_ascii_digit() {
                return None;
            }
            val += (b - b'0') as f64 * factor;
            factor *= 0.1;
        }
    }
    if exp != 0 {
        val *= libm::pow(10.0, exp as f64);
    }
    if neg {
        val = -val;
    }
    // Overflow guard: a 40-digit literal must be rejected, not become
    // +inf — infinity passes `in_range` for any unbounded-max param.
    let out = val as f32;
    out.is_finite().then_some(out)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Timeout for one-shot shell commands waiting on a channel message.
const ONESHOT_TIMEOUT: Duration = Duration::from_millis(150);

/// Read one message from a subscriber (skipping lagged entries) and write it
/// to the CDC class.  Prints `"no data (timeout)\r\n"` if nothing arrives
/// within `ONESHOT_TIMEOUT`.
async fn oneshot<'d, M, T, const CAP: usize, const SUBS: usize, const PUBS: usize>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
    sub: &mut Subscriber<'_, M, T, CAP, SUBS, PUBS>,
    buf_size: usize,
) -> Result<(), EndpointError>
where
    M: embassy_sync::blocking_mutex::raw::RawMutex,
    T: Clone,
    for<'m> ShellMsg<'m, T>: core::fmt::Display,
{
    match with_timeout(ONESHOT_TIMEOUT, next_message(sub)).await {
        Ok(msg) => {
            let mut buf = [0u8; 256];
            let buf = &mut buf[..buf_size.min(256)];
            let mut w = WriteBuf::new(buf);
            write!(w, "{}\r\n", ShellMsg(&msg)).ok();
            write_all(class, w.as_slice()).await
        }
        Err(_) => write_all(class, b"no data (timeout)\r\n").await,
    }
}

/// Measurement window for `imurate`. 1 s at 1 kHz is 1000 samples, so the
/// single-sample truncation at the tail is 0.1 % — an order of magnitude
/// finer than the ICM's own ±1 % oscillator tolerance, which is what
/// actually bounds the answer.
const IMU_RATE_WINDOW: Duration = Duration::from_millis(1000);

/// Count publishes on `sub` over [`IMU_RATE_WINDOW`], returning
/// `(samples, elapsed)` or `None` if the channel is silent.
///
/// `Lagged(n)` counts as `n` samples. The shell task cannot drain a 1 kHz
/// publisher through a CAP=4 queue, but the gap count is exactly what it
/// missed, so the total stays exact rather than saturating at the queue
/// depth.
///
/// The window opens on the first sample so the head is aligned to a sample
/// boundary; only the tail can clip one. `elapsed` is returned rather than
/// assumed to equal the window because the caller divides by what the clock
/// actually measured.
///
/// This compares two independent clocks — the IMU's internal oscillator
/// against the MCU timebase — so it is a real measurement of the ODR, not a
/// readback of what the driver was asked to program.
async fn measure_rate<M, T, const CAP: usize, const SUBS: usize, const PUBS: usize>(
    sub: &mut Subscriber<'_, M, T, CAP, SUBS, PUBS>,
) -> Option<(u64, Duration)>
where
    M: embassy_sync::blocking_mutex::raw::RawMutex,
    T: Clone,
{
    // Bail out on a dead channel so a missing IMU reads as "no data"
    // rather than as 0 Hz after a full window of silence.
    with_timeout(ONESHOT_TIMEOUT, next_message(sub)).await.ok()?;

    let t0 = Instant::now();
    let deadline = t0 + IMU_RATE_WINDOW;
    let mut samples: u64 = 0;
    loop {
        match select(Timer::at(deadline), sub.next_message()).await {
            Either::First(_) => break,
            Either::Second(WaitResult::Message(_)) => samples += 1,
            Either::Second(WaitResult::Lagged(n)) => samples += n,
        }
    }
    Some((samples, Instant::now().saturating_duration_since(t0)))
}

/// `imurate` — measure the actual IMU publish rate and compare it against
/// the rate this build was compiled for.
///
/// The build facts in `param list` report what the firmware *asked* the chip
/// for; `Icm426xx::sample_rate_hz()` echoes the same enum, so the boot-time
/// ODR assert cannot catch a chip that is running at a different rate than
/// requested. This measures it.
async fn dispatch_imurate<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
) -> Result<(), EndpointError> {
    let ctrl_div = (crate::params::get().indi.controller.ctrl_decimation as u32).max(1);
    let mut buf = [0u8; 256];
    let mut w = WriteBuf::new(&mut buf);
    write!(
        w,
        "measuring for {} ms (build expects {} Hz; INDI at /{} = {} Hz)...\r\n",
        IMU_RATE_WINDOW.as_millis(),
        crate::rates::IMU_ODR_HZ as u32,
        ctrl_div,
        crate::rates::IMU_ODR_HZ as u32 / ctrl_div,
    )
    .ok();
    write_all(class, w.as_slice()).await?;

    let mut sub1 = match IMU_1.subscriber() {
        Ok(s) => s,
        Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
    };
    // Both IMUs are measured in the same window rather than back to back:
    // one window is half the wait, and a shared window makes the two
    // numbers directly comparable when they disagree.
    let mut sub2 = if bsp::IMU_COUNT >= 2 {
        match IMU_2.subscriber() {
            Ok(s) => Some(s),
            Err(_) => None,
        }
    } else {
        None
    };

    let (r1, r2) = match sub2.as_mut() {
        Some(s2) => join(measure_rate(&mut sub1), measure_rate(s2)).await,
        None => (measure_rate(&mut sub1).await, None),
    };

    let lost1 = crate::sensors::imu::IMU1_LOST_SAMPLES.load(Ordering::Relaxed);
    let lost2 = crate::sensors::imu::IMU2_LOST_SAMPLES.load(Ordering::Relaxed);
    for (name, result, lost) in [("imu1", r1, lost1), ("imu2", r2, lost2)] {
        if name == "imu2" && bsp::IMU_COUNT < 2 {
            continue;
        }
        let mut buf = [0u8; 160];
        let mut w = WriteBuf::new(&mut buf);
        match result {
            Some((samples, elapsed)) => {
                let secs = elapsed.as_micros() as f32 * 1.0e-6;
                let hz = samples as f32 / secs;
                write!(
                    w,
                    "  {}: {} samples / {:.3} s = {:.1} Hz ({:+.2} %); lost since boot: {}\r\n",
                    name,
                    samples,
                    secs,
                    hz,
                    (hz / crate::rates::IMU_ODR_HZ - 1.0) * 100.0,
                    lost,
                )
                .ok();
            }
            None => {
                write!(w, "  {}: no data (timeout)\r\n", name).ok();
            }
        }
        write_all(class, w.as_slice()).await?;
    }
    Ok(())
}

/// `resetcause` — why the board last reset.
///
/// Decodes the RCC.RSR / PWR.CSR1 flags captured in `pre_init`
/// (feature-independent, unlike `postmortem show`). The one debugging
/// question this answers on a silent build: after an unexpected reboot,
/// reconnect the shell and distinguish
///   - IndependentWatchdog → firmware hung (panic/hardfault parks the
///     core with IRQs off, or the thread executor starved past 500 ms),
///   - Brownout / PowerOn(BOR) → the supply dipped (hardware, not code),
///   - Pin / PowerOn → NRST or a normal power-up.
async fn dispatch_resetcause<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
) -> Result<(), EndpointError> {
    use crate::reset_cause::{self, flags};
    let raw = reset_cause::read();
    let mut buf = [0u8; 256];
    let mut w = WriteBuf::new(&mut buf);
    write!(
        w,
        "reset cause: {:?} (raw=0x{:08x})\r\n",
        DebugResetKind(reset_cause::classify(raw)),
        raw,
    )
    .ok();
    // Raw flag breakdown: `classify` picks the most-specific single
    // cause, but several latch together on real hardware (POR + PIN on
    // every cold boot) and seeing all of them is what tells a power
    // glitch apart from a clean NRST.
    write!(w, "  flags:").ok();
    for (bit, name) in [
        (flags::IWDG_RESET, "IWDG"),
        (flags::BROWNOUT, "PVD-brownout"),
        (flags::SOFTWARE_RESET, "SFT"),
        (flags::PIN_RESET, "PIN"),
        (flags::POR, "POR"),
        (flags::BOR, "BOR"),
        (flags::LPWR_RESET, "LPWR"),
    ] {
        if raw & bit != 0 {
            write!(w, " {}", name).ok();
        }
    }
    if raw & flags::CAPTURED == 0 {
        write!(w, " (capture never ran!)").ok();
    } else if raw & !flags::CAPTURED == 0 {
        write!(w, " (none latched)").ok();
    }
    write!(w, "\r\n").ok();
    write_all(class, w.as_slice()).await
}

/// `ResetKind` derives `defmt::Format` but not `Debug` (postmortem's
/// defmt-first convention). Wrap it for the `core::fmt` shell path,
/// same pattern as `DebugReason` in `health.rs`.
struct DebugResetKind(crate::reset_cause::ResetKind);

impl core::fmt::Debug for DebugResetKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use crate::reset_cause::ResetKind as K;
        let s = match self.0 {
            K::Unknown => "Unknown",
            K::PowerOn => "PowerOn",
            K::Pin => "Pin",
            K::Software => "Software",
            K::IndependentWatchdog => "IndependentWatchdog",
            K::LowPower => "LowPower",
            K::Brownout => "Brownout",
            K::Multiple => "Multiple",
        };
        f.write_str(s)
    }
}

/// Drain lagged subscriber messages until a fresh one arrives.
async fn next_message<M, T, const CAP: usize, const SUBS: usize, const PUBS: usize>(
    sub: &mut Subscriber<'_, M, T, CAP, SUBS, PUBS>,
) -> T
where
    M: embassy_sync::blocking_mutex::raw::RawMutex,
    T: Clone,
{
    loop {
        match sub.next_message().await {
            WaitResult::Message(m) => return m,
            WaitResult::Lagged(_) => continue,
        }
    }
}
