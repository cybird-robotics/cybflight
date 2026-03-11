//! GPS sensor task — reads NAV-PVT frames from a u-blox M10 receiver and
//! publishes `GpsFix` messages to the `GPS_FIX` channel.

use cybflight_drivers::gps::UbloxM10;
use embassy_time::Instant;

use crate::hal;
use cybflight_msgs as msgs;

pub type GpsUart = hal::usart::BufferedUart<'static>;

pub struct GpsRunner {
    gps: UbloxM10<GpsUart>,
}

impl GpsRunner {
    pub fn new(gps: UbloxM10<GpsUart>) -> Self {
        Self { gps }
    }

    pub async fn run(&mut self) -> ! {
        let publisher = super::GPS_FIX.immediate_publisher();
        defmt::info!("GPS task running — waiting for NAV-PVT frames");
        loop {
            match self.gps.read_fix().await {
                Ok(pvt) => {
                    // Convert NED velocity (GPS native) → ENU (filter frame) here,
                    // so the channel always carries frame-consistent data.
                    // ENU: X=East, Y=North, Z=Up = (velE, velN, -velD).
                    use nalgebra::Vector3;
                    let vel_enu_m_s = Vector3::new(
                        pvt.vel_east_mm_s as f32 / 1000.0,
                        pvt.vel_north_mm_s as f32 / 1000.0,
                        -pvt.vel_down_mm_s as f32 / 1000.0,
                    );
                    publisher.publish_immediate(msgs::GpsFix {
                        timestamp: Instant::now(),
                        lat_deg: pvt.lat_1e7 as f64 * 1e-7,
                        lon_deg: pvt.lon_1e7 as f64 * 1e-7,
                        alt_msl_mm: pvt.alt_msl_mm,
                        ground_speed_mm_s: pvt.ground_speed_mm_s,
                        heading_mot_1e5: pvt.heading_mot_1e5,
                        vel_enu_m_s,
                        s_acc_m_s: pvt.s_acc_mm_s as f32 / 1000.0,
                        fix_type: pvt.fix_type,
                        num_sv: pvt.num_sv,
                        h_acc_mm: pvt.h_acc_mm,
                        v_acc_mm: pvt.v_acc_mm,
                        pdop: pvt.pdop,
                    });
                }
                Err(e) => {
                    defmt::warn!("GPS read error: {}", e);
                }
            }
        }
    }
}

#[embassy_executor::task]
pub async fn ublox_gps_task(mut runner: GpsRunner) {
    runner.run().await;
}
