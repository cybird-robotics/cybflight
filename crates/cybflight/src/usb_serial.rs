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
use crate::sensors::gps::{carr_soln_str, GPS_HEALTH, LATEST_NAV_PVT};
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
use embassy_time::{with_timeout, Duration, Instant};
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::driver::EndpointError;
use embassy_usb::Builder;
use hal::usb::Driver;

type UsbDriver<'d> = Driver<'d, hal::peripherals::USB_OTG_FS>;

const PROMPT: &[u8] = b"> ";
const HELP_TEXT: &[u8] = b"\
  imu1                 one-shot IMU 1 snapshot\r\n\
  imu2                 one-shot IMU 2 snapshot\r\n\
  att                  one-shot attitude snapshot\r\n\
  ocp                  one-shot OCP solver output\r\n\
  rc                   one-shot RC channel values\r\n\
  rcstats              one-shot RC link status\r\n\
  dshot                one-shot DShot telemetry\r\n\
  power                one-shot power status\r\n\
  gps                  one-shot GPS fix\r\n\
  gpshealth            one-shot GPS init/fix health\r\n\
  gpsrtk               one-shot RTK / NAV-PVT detail (corrections, accuracy)\r\n\
  magext               one-shot external compass\r\n\
  magint               one-shot internal compass\r\n\
  baro1                one-shot barometer 1\r\n\
  baro2                one-shot barometer 2\r\n\
  vicon                one-shot Vicon pose\r\n\
  timesync             one-shot time sync status\r\n\
  eskf                 one-shot estimator status\r\n\
  health               aggregated arming/health snapshot\r\n\
  stream <topic> on    stream data on <topic>\r\n\
  stream <topic> off   stop data stream on <topic>\r\n\
                       (topics: imu1 imu2 att ocp rc rcstats dshot gps gpsrtk\r\n\
                        magext magint baro1 baro2 attcontrol vicon timesync eskf)\r\n\
  motor <1-4> <0-100>  set motor throttle (test mode)\r\n\
  param list           list all vehicle parameters\r\n\
  param get <name>     get a parameter value\r\n\
  param set <name> <v> set a parameter (in-memory)\r\n\
  param save           write params to flash\r\n\
  param defaults       reset to compile-time defaults\r\n\
  mission list                            list available offline trajectories\r\n\
  mission get                             show the active trajectory\r\n\
  mission set <env> <variant> <speed>     select a trajectory (in-memory; 'param save' to persist)\r\n\
  led on               enable arm LEDs (red top / blue bottom, brighter when armed)\r\n\
  led off              disable arm LEDs\r\n\
  blackbox record on   manually start a flight log (flight_NNNN.mcap)\r\n\
  blackbox record off  stop the manual flight log\r\n\
  blackbox status      show current recording state and what's triggering it\r\n\
  blackbox set <tier>  set the record-set tier (none|small|mid|large)\r\n\
                       small=events+rc, mid=+attitude, large=+imu (default), none=disabled\r\n\
  blackbox ls          list files in the FAT root (rejected while recording)\r\n\
                       (recorder also auto-starts on real ARM_STATE arm: RC switch / failsafe path)\r\n\
                       (file numbering picks up after the highest existing flight_NNNN; survives reboots)\r\n\
  reboot               software reset\r\n\
  reboot --dfu         reset into USB DFU bootloader\r\n\
  help                 show this message\r\n\
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

    let mut line_buf = [0u8; 64];
    let mut line_len = 0usize;
    let mut rx_buf = [0u8; 64];

    let mut banner_buf = [0u8; 128];
    let mut w = WriteBuf::new(&mut banner_buf);
    let _ = write!(
        w,
        "cybflight v{} ({}) {} [{}]\r\ntype 'help' for commands\r\n\r\n> ",
        crate::BUILD_VERSION,
        crate::GIT_HASH,
        crate::BUILD_TIMESTAMP,
        crate::bsp::BOARD_NAME,
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
            let mut buf = [0u8; 192];
            let mut w = WriteBuf::new(&mut buf);
            write!(w, "{}\r\n", health).ok();
            write_all(class, w.as_slice()).await?;
        }
        "gpsrtk" => {
            let snap = LATEST_NAV_PVT.lock(|c| c.get());
            let mut buf = [0u8; 384];
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
    let _ = write!(
        w,
        "blackbox status: {}  (trigger: {}){}\r\n\
         \x20                 IS_ARMED={}, RECORDER_HOLD={}\r\n\
         \x20                 record_set={}\r\n",
        state,
        trigger,
        storage_note,
        armed,
        hold,
        rs.name(),
    );
    write_all(class, w.as_slice()).await
}

/// `blackbox set <none|small|mid|large>` — change the active record
/// set and persist it to flash.
///
/// Three-step update: the live atomic the recorder reads, the
/// in-memory param copy, then `save_to_flash`. Refused while armed
/// because `save_to_flash` triggers a same-bank flash erase that
/// stalls the CPU bus for ~1–2 s with the IWDG extended — fine on
/// the bench, dangerous mid-flight. Mirrors the armed-guard
/// precedent set by `led on/off` and `mission set`.
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
            params.blackbox_record_set = rs as u8;
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
                b"usage: blackbox set <none|small|mid|large>\r\n\
                  \x20  none  - recorder muted (arm-edges produce no file)\r\n\
                  \x20  small - events + rc\r\n\
                  \x20  mid   - events + rc + attitude\r\n\
                  \x20  large - events + rc + attitude + imu1 (default)\r\n",
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
        OpError::Unmount => {
            "unmount-flush failed (file may not have hit the card; see defmt log)"
        }
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

async fn dispatch_param<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
    line: &str,
) -> Result<(), EndpointError> {
    use cybflight_core::params::{ParamKey, ALL_KEYS};

    let mut parts = line.split_ascii_whitespace();
    parts.next(); // skip "param"
    let sub = parts.next().unwrap_or("");

    match sub {
        "list" => {
            let params = crate::params::get();
            for &key in ALL_KEYS {
                let mut buf = [0u8; 64];
                let mut w = WriteBuf::new(&mut buf);
                write!(w, "  {:12} = {}\r\n", key.as_str(), params.get(key)).ok();
                write_all(class, w.as_slice()).await?;
            }
        }
        "get" => {
            let name = parts.next();
            match name.and_then(ParamKey::from_str) {
                Some(key) => {
                    let val = crate::params::get().get(key);
                    let mut buf = [0u8; 64];
                    let mut w = WriteBuf::new(&mut buf);
                    write!(w, "{} = {}\r\n", key.as_str(), val).ok();
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
            let key = name.and_then(ParamKey::from_str);
            match (key, val_str) {
                (Some(key), Some(val_str)) => {
                    if let Some(val) = parse_f32(val_str) {
                        let mut params = crate::params::get();
                        params.set(key, val);
                        crate::params::set(params);
                        let mut buf = [0u8; 64];
                        let mut w = WriteBuf::new(&mut buf);
                        write!(w, "{} = {}\r\n", key.as_str(), val).ok();
                        write_all(class, w.as_slice()).await?;
                    } else {
                        write_all(class, b"invalid number\r\n").await?;
                    }
                }
                _ => {
                    write_all(class, b"usage: param set <name> <value>\r\n").await?;
                }
            }
        }
        "save" => match crate::params::save_to_flash() {
            Ok(()) => {
                write_all(class, b"params saved to flash\r\n").await?;
            }
            Err(e) => {
                let mut buf = [0u8; 64];
                let mut w = WriteBuf::new(&mut buf);
                write!(w, "error: {}\r\n", e).ok();
                write_all(class, w.as_slice()).await?;
            }
        },
        "defaults" => {
            let defaults = crate::vehicle::default_params();
            // Mirror the default mission_profile into the live atomic so
            // a `mission get` after `param defaults` sees the reset, not
            // the previous selection. (Without this, the atomic only
            // resyncs on the next reboot via `init_from_flash`.)
            #[cfg(feature = "outer_mpc")]
            crate::control::offline_mission::init_active_from_index(defaults.mission_profile);
            crate::params::set(defaults);
            write_all(class, b"params reset to defaults (not saved)\r\n").await?;
        }
        _ => {
            write_all(class, b"usage: param list|get|set|save|defaults\r\n").await?;
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
            let p = offline_mission::active();
            let mut buf = [0u8; 96];
            let mut w = WriteBuf::new(&mut buf);
            write!(
                w,
                "{} (idx={}, n={})\r\n",
                p.name,
                offline_mission::active_index(),
                p.num_pieces(),
            )
            .ok();
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
            let env = parts.next();
            let variant = parts.next();
            let speed = parts.next();
            let (env, variant, speed) = match (env, variant, speed) {
                (Some(a), Some(b), Some(c)) => (a, b, c),
                _ => {
                    return write_all(
                        class,
                        b"usage: mission set <env> <variant> <speed>\r\n",
                    )
                    .await;
                }
            };
            match offline_mission::find(env, variant, speed) {
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
                    params.mission_profile = idx;
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
                b"usage: mission list|get|set <env> <variant> <speed>\r\n",
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
    params.arm_led_enabled = enable;
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
    let (s, neg) = if let Some(rest) = s.strip_prefix('-') {
        (rest, true)
    } else if let Some(rest) = s.strip_prefix('+') {
        (rest, false)
    } else {
        (s, false)
    };
    let (int_part, frac_part) = match s.find('.') {
        Some(dot) => (&s[..dot], Some(&s[dot + 1..])),
        None => (s, None),
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
    if neg {
        val = -val;
    }
    Some(val as f32)
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
