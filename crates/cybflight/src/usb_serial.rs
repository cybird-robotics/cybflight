use core::fmt::Write;

use crate::bsp;
use crate::hal;
use crate::sensors::FUSED_IMU;
use crate::ImuSample;
use embassy_futures::join::join;
use embassy_usb::Builder;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use hal::usb::Driver;

#[embassy_executor::task]
pub async fn task(
    usb_otg: hal::Peri<'static, hal::peripherals::USB_OTG_FS>,
    dp: hal::Peri<'static, hal::peripherals::PA12>,
    dm: hal::Peri<'static, hal::peripherals::PA11>,
) {
    run(usb_otg, dp, dm).await
}

/// Run the USB CDC serial task.
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
    config.product = Some("cybflight-imu");
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
            defmt::info!("USB CDC connected");
            loop {
                let sample = FUSED_IMU.receive().await;
                let mut buf = [0u8; 128];
                let n = format_imu(&sample, &mut buf);
                if class.write_packet(&buf[..n]).await.is_err() {
                    break;
                }
            }
            defmt::info!("USB CDC disconnected");
        }
    })
    .await;
}

/// Format an IMU sample as CSV text into a fixed buffer.
fn format_imu(s: &ImuSample, buf: &mut [u8; 128]) -> usize {
    let mut w = WriteBuf {
        buf: buf.as_mut_slice(),
        pos: 0,
    };
    let _ = write!(
        w,
        "imu,{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1}\r\n",
        s.accel.x, s.accel.y, s.accel.z, s.gyro.x, s.gyro.y, s.gyro.z, s.temp_c,
    );
    w.pos
}

struct WriteBuf<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> Write for WriteBuf<'a> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let remaining = &mut self.buf[self.pos..];
        if bytes.len() > remaining.len() {
            return Err(core::fmt::Error);
        }
        remaining[..bytes.len()].copy_from_slice(bytes);
        self.pos += bytes.len();
        Ok(())
    }
}
