use core::fmt::Write;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::bsp;
use crate::comm;
use crate::control::ATTITUDE_CONTROL_SETPOINT;
use crate::control::OCP_SOLVER_OUTPUT;
use crate::hal;
use crate::motors::ACTUATOR_MOTORS;
use crate::msgs;
use crate::platform;
use crate::sensors::{
    BARO_1, BARO_2, DSHOT_TELEMETRY, GPS_FIX, IMU_1, IMU_2, MAG_EXT, MAG_INT, RC_INPUT,
    RC_LINK_STATUS, VEHICLE_ATTITUDE, VICON_POSE,
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
  gps                  one-shot GPS fix\r\n\
  magext               one-shot external compass\r\n\
  magint               one-shot internal compass\r\n\
  baro1                one-shot barometer 1\r\n\
  baro2                one-shot barometer 2\r\n\
  vicon                one-shot Vicon pose\r\n\
  timesync             one-shot time sync status\r\n\
  stream <topic> on    stream data on <topic>\r\n\
  stream <topic> off   stop data stream on <topic>\r\n\
  motor <1-4> <0-100>  set motor throttle (test mode)\r\n\
  param list           list all vehicle parameters\r\n\
  param get <name>     get a parameter value\r\n\
  param set <name> <v> set a parameter (in-memory)\r\n\
  param save           write params to flash\r\n\
  param defaults       reset to compile-time defaults\r\n\
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
pub static STREAM_GPS: AtomicBool = AtomicBool::new(false);
pub static STREAM_MAGEXT: AtomicBool = AtomicBool::new(false);
pub static STREAM_MAGINT: AtomicBool = AtomicBool::new(false);
pub static STREAM_BARO1: AtomicBool = AtomicBool::new(false);
pub static STREAM_BARO2: AtomicBool = AtomicBool::new(false);
pub static STREAM_ATTITUDE_CONTROL: AtomicBool = AtomicBool::new(false);
pub static STREAM_VICON: AtomicBool = AtomicBool::new(false);
pub static STREAM_TIMESYNC: AtomicBool = AtomicBool::new(false);

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
pub async fn gps_stream_task() {
    msg_stream_task(&GPS_FIX, &STREAM_GPS).await
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
    STREAM_GPS.store(false, Ordering::Relaxed);
    STREAM_MAGEXT.store(false, Ordering::Relaxed);
    STREAM_MAGINT.store(false, Ordering::Relaxed);
    STREAM_BARO1.store(false, Ordering::Relaxed);
    STREAM_BARO2.store(false, Ordering::Relaxed);
    STREAM_ATTITUDE_CONTROL.store(false, Ordering::Relaxed);
    STREAM_VICON.store(false, Ordering::Relaxed);
    STREAM_TIMESYNC.store(false, Ordering::Relaxed);

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
        "gps" => {
            let mut sub = match GPS_FIX.subscriber() {
                Ok(s) => s,
                Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
            };
            oneshot(class, &mut sub, 256).await?;
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
        _ => {
            write_all(class, b"unknown command (try 'help')\r\n").await?;
        }
    }
    Ok(())
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
            crate::params::set(crate::vehicle::default_params());
            write_all(class, b"params reset to defaults (not saved)\r\n").await?;
        }
        _ => {
            write_all(class, b"usage: param list|get|set|save|defaults\r\n").await?;
        }
    }
    Ok(())
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
