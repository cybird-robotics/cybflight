use core::fmt::Write;

use crate::bsp;
use crate::hal;
use crate::msgs;
use crate::platform;
use crate::sensors::{RAW_IMU, VEHICLE_ATTITUDE};
use crate::shell::format::ShellMsg;
use crate::shell::WriteBuf;
use embassy_futures::join::join;
use embassy_futures::select::{select3, Either3};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::pubsub::{Subscriber, WaitResult};
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::driver::EndpointError;
use embassy_usb::Builder;
use hal::usb::Driver;

type UsbDriver<'d> = Driver<'d, hal::peripherals::USB_OTG_FS>;
type ImuSub = Subscriber<'static, CriticalSectionRawMutex, msgs::Imu, 4, 4, 2>;
type AttSub = Subscriber<'static, CriticalSectionRawMutex, msgs::VehicleAttitude, 4, 4, 1>;

const PROMPT: &[u8] = b"> ";
const HELP_TEXT: &[u8] = b"\
  imu              one-shot IMU snapshot\r\n\
  att              one-shot attitude snapshot\r\n\
  stream imu on    stream IMU at sensor rate\r\n\
  stream imu off   stop IMU stream\r\n\
  stream att on    stream vehicle attitude\r\n\
  stream att off   stop attitude stream\r\n\
  reboot           software reset\r\n\
  reboot --dfu     reset into USB DFU bootloader\r\n\
  help             show this message\r\n\
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

    let mut line_buf = [0u8; 64];
    let mut line_len = 0usize;
    let mut stream_imu = false;
    let mut stream_att = false;
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
        match select3(
            class.read_packet(&mut rx_buf),
            imu_sub.next_message(),
            att_sub.next_message(),
        )
        .await
        {
            Either3::First(Err(_)) => return,

            Either3::First(Ok(n)) => {
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
                                &mut stream_imu,
                                &mut stream_att,
                                &mut imu_sub,
                                &mut att_sub,
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

            Either3::Second(WaitResult::Message(imu)) => {
                if stream_imu {
                    let mut buf = [0u8; 128];
                    let mut w = WriteBuf::new(buf.as_mut_slice());
                    let _ = write!(w, "{}\r\n", ShellMsg(&imu));
                    if write_all(class, w.as_slice()).await.is_err() {
                        return;
                    }
                }
            }
            Either3::Second(WaitResult::Lagged(n)) => {
                defmt::warn!("USB shell: dropped {} IMU samples", n);
            }

            Either3::Third(WaitResult::Message(att)) => {
                if stream_att {
                    let mut buf = [0u8; 64];
                    let mut w = WriteBuf::new(buf.as_mut_slice());
                    let _ = write!(w, "{}\r\n", ShellMsg(&att));
                    if write_all(class, w.as_slice()).await.is_err() {
                        return;
                    }
                }
            }
            Either3::Third(WaitResult::Lagged(n)) => {
                defmt::warn!("USB shell: dropped {} attitude samples", n);
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
    stream_imu: &mut bool,
    stream_att: &mut bool,
    imu_sub: &mut ImuSub,
    att_sub: &mut AttSub,
) -> Result<(), EndpointError> {
    match line {
        "" => {}
        "help" => {
            write_all(class, HELP_TEXT).await?;
        }
        "imu" => {
            let msg = next_message(imu_sub).await;
            let mut buf = [0u8; 256];
            let mut w = WriteBuf::new(buf.as_mut_slice());
            write!(w, "{}\r\n", ShellMsg(&msg)).ok();
            write_all(class, w.as_slice()).await?;
        }
        "att" => {
            let msg = next_message(att_sub).await;
            let mut buf = [0u8; 256];
            let mut w = WriteBuf::new(buf.as_mut_slice());
            write!(w, "{}\r\n", ShellMsg(&msg)).ok();
            write_all(class, w.as_slice()).await?;
        }
        "stream imu on" => {
            *stream_imu = true;
            write_all(class, b"IMU stream on\r\n").await?;
        }
        "stream imu off" => {
            *stream_imu = false;
            write_all(class, b"IMU stream off\r\n").await?;
        }
        "stream att on" => {
            *stream_att = true;
            write_all(class, b"attitude stream on\r\n").await?;
        }
        "stream att off" => {
            *stream_att = false;
            write_all(class, b"attitude stream off\r\n").await?;
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

// ---------------------------------------------------------------------------
// USB write helper
// ---------------------------------------------------------------------------

/// Write `data` to the CDC class in ≤64-byte packets.
///
/// A zero-length packet (ZLP) is appended when `data` is an exact multiple of
/// 64 bytes, signalling end-of-transfer to the USB host.
async fn write_all<'d>(
    class: &mut CdcAcmClass<'d, UsbDriver<'d>>,
    data: &[u8],
) -> Result<(), EndpointError> {
    for chunk in data.chunks(64) {
        class.write_packet(chunk).await?;
    }
    if !data.is_empty() && data.len() % 64 == 0 {
        class.write_packet(&[]).await?;
    }
    Ok(())
}
