use core::fmt::Write;

use crate::bsp;
use crate::control::OCP_SOLVER_OUTPUT;
use crate::hal;
use crate::msgs;
use crate::platform;
use crate::sensors::{RAW_IMU, RC_INPUT, RC_LINK_STATUS, VEHICLE_ATTITUDE};
use crate::shell::format::ShellMsg;
use crate::shell::write_all;
use crate::shell::WriteBuf;
use embassy_futures::join::join;
use embassy_futures::select::{select6, Either6};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::pubsub::{Subscriber, WaitResult};
use embassy_time::{with_timeout, Duration};
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::driver::EndpointError;
use embassy_usb::Builder;
use hal::usb::Driver;

type UsbDriver<'d> = Driver<'d, hal::peripherals::USB_OTG_FS>;
type ImuSub = Subscriber<'static, CriticalSectionRawMutex, msgs::Imu, 4, 4, 2>;
type AttSub = Subscriber<'static, CriticalSectionRawMutex, msgs::VehicleAttitude, 4, 4, 1>;
type OcpSub = Subscriber<'static, CriticalSectionRawMutex, msgs::OcpSolverOutput, 4, 4, 1>;
type RcSub = Subscriber<'static, CriticalSectionRawMutex, msgs::RcInput, 4, 4, 1>;
type RcLinkSub = Subscriber<'static, CriticalSectionRawMutex, msgs::RcLinkStatus, 2, 3, 1>;

const PROMPT: &[u8] = b"> ";
const HELP_TEXT: &[u8] = b"\
  imu                one-shot IMU snapshot\r\n\
  att                one-shot attitude snapshot\r\n\
  stream <topic> on  stream data on <topic>\r\n\
  stream <topic> off stop data stream on <topic>\r\n\
  reboot             software reset\r\n\
  reboot --dfu       reset into USB DFU bootloader\r\n\
  help               show this message\r\n\
";

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

#[derive(Default)]
struct StreamState {
    pub stream_imu: bool,
    pub stream_att: bool,
    pub stream_ocp: bool,
    pub stream_rc: bool,
    pub stream_rcstats: bool,
}

// ---------------------------------------------------------------------------
// Shell loop
// ---------------------------------------------------------------------------

async fn shell_loop<'d>(class: &mut CdcAcmClass<'d, UsbDriver<'d>>) {
    let mut imu_sub: ImuSub = match RAW_IMU.subscriber() {
        Ok(s) => s,
        Err(_) => {
            defmt::error!("USB shell: RAW_IMU subscriber slots exhausted");
            return;
        }
    };
    let mut att_sub: AttSub = match VEHICLE_ATTITUDE.subscriber() {
        Ok(s) => s,
        Err(_) => {
            defmt::error!("USB shell: VEHICLE_ATTITUDE subscriber slots exhausted");
            return;
        }
    };

    let mut ocp_sub = match OCP_SOLVER_OUTPUT.subscriber() {
        Ok(s) => s,
        Err(_) => {
            defmt::error!("USB shell: OCP_SOLVER_OUTPUT subscriber slots exhausted");
            return;
        }
    };

    let mut rc_sub: RcSub = match RC_INPUT.subscriber() {
        Ok(s) => s,
        Err(_) => {
            defmt::error!("USB shell: RC_INPUT subscriber slots exhausted");
            return;
        }
    };

    let mut rc_link_sub: RcLinkSub = match RC_LINK_STATUS.subscriber() {
        Ok(s) => s,
        Err(_) => {
            defmt::error!("USB shell: RC_LINK_STATUS subscriber slots exhausted");
            return;
        }
    };

    let mut line_buf = [0u8; 64];
    let mut line_len = 0usize;

    let mut state = StreamState::default();
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
        match select6(
            class.read_packet(&mut rx_buf),
            imu_sub.next_message(),
            att_sub.next_message(),
            ocp_sub.next_message(),
            rc_sub.next_message(),
            rc_link_sub.next_message(),
        )
        .await
        {
            Either6::First(Err(_)) => return,

            Either6::First(Ok(n)) => {
                for &b in &rx_buf[..n] {
                    match b {
                        b'\r' | b'\n' => {
                            if write_all(class, b"\r\n").await.is_err() {
                                return;
                            }
                            let line = core::str::from_utf8(&line_buf[..line_len])
                                .unwrap_or("")
                                .trim();
                            if dispatch(
                                class,
                                line,
                                &mut state,
                                &mut imu_sub,
                                &mut att_sub,
                                &mut ocp_sub,
                                &mut rc_sub,
                                &mut rc_link_sub,
                            )
                            .await
                            .is_err()
                            {
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

            Either6::Second(WaitResult::Message(imu)) => {
                if state.stream_imu {
                    let mut buf = [0u8; 128];
                    let mut w = WriteBuf::new(buf.as_mut_slice());
                    let _ = write!(w, "{}\r\n", ShellMsg(&imu));
                    if write_all(class, w.as_slice()).await.is_err() {
                        return;
                    }
                }
            }
            Either6::Second(WaitResult::Lagged(n)) => {
                defmt::warn!("USB shell: dropped {} IMU samples", n);
            }

            Either6::Third(WaitResult::Message(att)) => {
                if state.stream_att {
                    let mut buf = [0u8; 64];
                    let mut w = WriteBuf::new(buf.as_mut_slice());
                    let _ = write!(w, "{}\r\n", ShellMsg(&att));
                    if write_all(class, w.as_slice()).await.is_err() {
                        return;
                    }
                }
            }
            Either6::Third(WaitResult::Lagged(n)) => {
                defmt::warn!("USB shell: dropped {} attitude samples", n);
            }

            Either6::Fourth(WaitResult::Message(ocp)) => {
                if state.stream_ocp {
                    let mut buf = [0u8; 256];
                    let mut w = WriteBuf::new(buf.as_mut_slice());
                    let _ = write!(w, "{}\r\n", ShellMsg(&ocp));
                    if write_all(class, w.as_slice()).await.is_err() {
                        return;
                    }
                }
            }
            Either6::Fourth(WaitResult::Lagged(n)) => {
                defmt::warn!("USB shell: dropped {} OCP solver outputs", n);
            }

            Either6::Fifth(WaitResult::Message(rc)) => {
                if state.stream_rc {
                    let mut buf = [0u8; 256];
                    let mut w = WriteBuf::new(buf.as_mut_slice());
                    let _ = write!(w, "{}\r\n", ShellMsg(&rc));
                    if write_all(class, w.as_slice()).await.is_err() {
                        return;
                    }
                }
            }
            Either6::Fifth(WaitResult::Lagged(n)) => {
                defmt::warn!("USB shell: dropped {} RC input samples", n);
            }

            Either6::Sixth(WaitResult::Message(link)) => {
                if state.stream_rcstats {
                    let mut buf = [0u8; 128];
                    let mut w = WriteBuf::new(buf.as_mut_slice());
                    let _ = write!(w, "{}\r\n", ShellMsg(&link));
                    if write_all(class, w.as_slice()).await.is_err() {
                        return;
                    }
                }
            }
            Either6::Sixth(WaitResult::Lagged(n)) => {
                defmt::warn!("USB shell: dropped {} RC link status samples", n);
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
    state: &mut StreamState,
    imu_sub: &mut ImuSub,
    att_sub: &mut AttSub,
    ocp_sub: &mut OcpSub,
    rc_sub: &mut RcSub,
    rc_link_sub: &mut RcLinkSub,
) -> Result<(), EndpointError> {
    match line {
        "" => {}
        "help" => {
            write_all(class, HELP_TEXT).await?;
        }
        "imu" => {
            if let Some(msg) = next_message(imu_sub).await {
                let mut buf = [0u8; 256];
                let mut w = WriteBuf::new(buf.as_mut_slice());
                write!(w, "{}\r\n", ShellMsg(&msg)).ok();
                write_all(class, w.as_slice()).await?;
            } else {
                write_all(class, b"no data (timeout)\r\n").await?;
            }
        }
        "att" => {
            if let Some(msg) = next_message(att_sub).await {
                let mut buf = [0u8; 256];
                let mut w = WriteBuf::new(buf.as_mut_slice());
                write!(w, "{}\r\n", ShellMsg(&msg)).ok();
                write_all(class, w.as_slice()).await?;
            } else {
                write_all(class, b"no data (timeout)\r\n").await?;
            }
        }
        "ocp" => {
            if let Some(msg) = next_message(ocp_sub).await {
                let mut buf = [0u8; 256];
                let mut w = WriteBuf::new(buf.as_mut_slice());
                write!(w, "{}\r\n", ShellMsg(&msg)).ok();
                write_all(class, w.as_slice()).await?;
            } else {
                write_all(class, b"no data (timeout)\r\n").await?;
            }
        }
        "rc" => {
            if let Some(msg) = next_message(rc_sub).await {
                let mut buf = [0u8; 256];
                let mut w = WriteBuf::new(buf.as_mut_slice());
                write!(w, "{}\r\n", ShellMsg(&msg)).ok();
                write_all(class, w.as_slice()).await?;
            } else {
                write_all(class, b"no data (timeout)\r\n").await?;
            }
        }
        "rcstats" => {
            if let Some(msg) = next_message(rc_link_sub).await {
                let mut buf = [0u8; 128];
                let mut w = WriteBuf::new(buf.as_mut_slice());
                write!(w, "{}\r\n", ShellMsg(&msg)).ok();
                write_all(class, w.as_slice()).await?;
            } else {
                write_all(class, b"no data (timeout)\r\n").await?;
            }
        }
        "stream imu on" => {
            state.stream_imu = true;
            write_all(class, b"IMU stream on\r\n").await?;
        }
        "stream imu off" => {
            state.stream_imu = false;
            write_all(class, b"IMU stream off\r\n").await?;
        }
        "stream att on" => {
            state.stream_att = true;
            write_all(class, b"attitude stream on\r\n").await?;
        }
        "stream att off" => {
            state.stream_att = false;
            write_all(class, b"attitude stream off\r\n").await?;
        }
        "stream ocp on" => {
            state.stream_ocp = true;
            write_all(class, b"OCP solver output stream on\r\n").await?;
        }
        "stream ocp off" => {
            state.stream_ocp = false;
            write_all(class, b"OCP solver output stream off\r\n").await?;
        }
        "stream rc on" => {
            state.stream_rc = true;
            write_all(class, b"RC stream on\r\n").await?;
        }
        "stream rc off" => {
            state.stream_rc = false;
            write_all(class, b"RC stream off\r\n").await?;
        }
        "stream rcstats on" => {
            state.stream_rcstats = true;
            write_all(class, b"RC link status stream on\r\n").await?;
        }
        "stream rcstats off" => {
            state.stream_rcstats = false;
            write_all(class, b"RC link status stream off\r\n").await?;
        }
        "reboot" => {
            write_all(class, b"rebooting...\r\n").await?;
            platform::sys_reboot();
        }
        "reboot --dfu" => {
            write_all(class, b"entering DFU mode...\r\n").await?;
            platform::enter_dfu();
        }
        _ => {
            write_all(class, b"unknown command (try 'help')\r\n").await?;
        }
    }
    Ok(())
}

/// Timeout for one-shot shell commands waiting on a channel message.
/// Matches Betaflight's RXLOSS_TRIGGER_INTERVAL (150ms).
const ONESHOT_TIMEOUT: Duration = Duration::from_millis(150);

/// Drain lagged subscriber messages until a fresh one arrives, with timeout.
async fn next_message<M, T, const CAP: usize, const SUBS: usize, const PUBS: usize>(
    sub: &mut Subscriber<'_, M, T, CAP, SUBS, PUBS>,
) -> Option<T>
where
    M: embassy_sync::blocking_mutex::raw::RawMutex,
    T: Clone,
{
    with_timeout(ONESHOT_TIMEOUT, async {
        loop {
            match sub.next_message().await {
                WaitResult::Message(m) => return m,
                WaitResult::Lagged(_) => continue,
            }
        }
    })
    .await
    .ok()
}
