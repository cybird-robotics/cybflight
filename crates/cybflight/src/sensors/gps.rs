//! GPS sensor task — reads NAV-PVT frames from a u-blox M10 receiver and
//! fans them out to both `GPS_FIX` (trimmed, for telemetry) and
//! `GPS_NAV_PVT` (full NAV-PVT signal used by the ESKF GPS path).

use cybflight_drivers::gps::UbloxM10;
use embassy_time::Instant;

use crate::hal;
use cybflight_msgs as msgs;

pub type GpsUart = hal::usart::BufferedUart<'static>;

/// Full NAV-PVT fix used by `eskf_imu_gps` for position + velocity updates.
/// Kept local until `cybflight-msgs::GpsFix` grows NED velocity and
/// speed-accuracy fields; published to `super::GPS_NAV_PVT` as a `Signal`
/// (latest-wins, 5 Hz cadence — no queuing needed).
#[derive(Clone, Copy)]
pub struct GpsNavPvt {
    pub timestamp: Instant,
    pub fix_type: u8,
    pub num_sv: u8,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_msl_mm: i32,
    pub vel_north_mm_s: i32,
    pub vel_east_mm_s: i32,
    pub vel_down_mm_s: i32,
    pub h_acc_mm: u32,
    pub v_acc_mm: u32,
    pub s_acc_mm_s: u32,
}

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
                    let now = Instant::now();
                    let lat_deg = pvt.lat_1e7 as f64 * 1e-7;
                    let lon_deg = pvt.lon_1e7 as f64 * 1e-7;
                    publisher.publish_immediate(msgs::GpsFix {
                        timestamp: now,
                        lat_deg,
                        lon_deg,
                        alt_msl_mm: pvt.alt_msl_mm,
                        ground_speed_mm_s: pvt.ground_speed_mm_s,
                        heading_mot_1e5: pvt.heading_mot_1e5,
                        fix_type: pvt.fix_type,
                        num_sv: pvt.num_sv,
                        h_acc_mm: pvt.h_acc_mm,
                        v_acc_mm: pvt.v_acc_mm,
                        pdop: pvt.pdop,
                    });
                    super::GPS_NAV_PVT.signal(GpsNavPvt {
                        timestamp: now,
                        fix_type: pvt.fix_type,
                        num_sv: pvt.num_sv,
                        lat_deg,
                        lon_deg,
                        alt_msl_mm: pvt.alt_msl_mm,
                        vel_north_mm_s: pvt.vel_north_mm_s,
                        vel_east_mm_s: pvt.vel_east_mm_s,
                        vel_down_mm_s: pvt.vel_down_mm_s,
                        h_acc_mm: pvt.h_acc_mm,
                        v_acc_mm: pvt.v_acc_mm,
                        s_acc_mm_s: pvt.s_acc_mm_s,
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
