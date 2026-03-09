use cybflight_msgs as msgs;

use core::fmt;

pub struct ShellMsg<'a, T>(pub &'a T);

impl fmt::Display for ShellMsg<'_, msgs::Imu> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(
            f,
            "Imu(timestamp={:.4},accel_m_s2=[{:.4},{:.4},{:.4}],gyro_rad_s=[{:.4},{:.4},{:.4}],temp_c={:.4})",
            s.timestamp.as_millis(),
            s.accel_m_s2.x,
            s.accel_m_s2.y,
            s.accel_m_s2.z,
            s.gyro_rad_s.x,
            s.gyro_rad_s.y,
            s.gyro_rad_s.z,
            s.temp_c
        )
    }
}

impl fmt::Display for ShellMsg<'_, msgs::VehicleAttitude> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(
            f,
            "VehicleAttitude(timestamp={:.4}, orientation=Quaternion(x={:.4},y={:.4},z={:.4},w={:.4}))",
            s.timestamp.as_millis(),
            s.orientation.i,
            s.orientation.j,
            s.orientation.k,
            s.orientation.w,
        )
    }
}

impl fmt::Display for ShellMsg<'_, msgs::Pose> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(
            f,
            "Pose(position=[{:.4},{:.4},{:.4}], orientation=Quaternion(x={:.4},y={:.4},z={:.4},w={:.4}))",
            s.position.x,
            s.position.y,
            s.position.z,
            s.orientation.i,
            s.orientation.j,
            s.orientation.k,
            s.orientation.w
        )
    }
}

impl fmt::Display for ShellMsg<'_, msgs::Twist> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(
            f,
            "Twist(linear=[{:.4},{:.4},{:.4}], angular=[{:.4},{:.4},{:.4}])",
            s.linear.x, s.linear.y, s.linear.z, s.angular.x, s.angular.y, s.angular.z,
        )
    }
}

impl fmt::Display for ShellMsg<'_, msgs::VehicleOdometry> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(
            f,
            "VehicleOdometry(timestamp={:.4}, pose={}, twist={})",
            s.timestamp.as_millis(),
            ShellMsg(&s.pose),
            ShellMsg(&s.twist)
        )
    }
}

impl fmt::Display for ShellMsg<'_, msgs::OcpSolverOutput> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(
            f,
            "OcpSolverOutput(timestamp={:.4}, command={:?}, iterations={}, converged={}, solve_time_ms={})",
            s.timestamp.as_millis(),
            s.command.as_slice(),
            s.iterations,
            s.converged,
            s.solve_time_us as f64 / 1000.0
        )
    }
}

impl fmt::Display for ShellMsg<'_, msgs::RcInput> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(
            f,
            "RcInput(timestamp={}, channels=[",
            s.timestamp.as_millis()
        )?;
        for i in 0..s.channel_count as usize {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", s.channels[i])?;
        }
        write!(f, "], count={})", s.channel_count)
    }
}

impl fmt::Display for ShellMsg<'_, msgs::RcLinkStatus> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(
            f,
            "RcLinkStatus(timestamp={}, rssi={}dBm, lq={}%, snr={}dB, rf_mode={})",
            s.timestamp.as_millis(),
            s.rssi_dbm,
            s.link_quality,
            s.snr,
            s.rf_mode
        )
    }
}

impl fmt::Display for ShellMsg<'_, msgs::GpsFix> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(
            f,
            "GpsFix(t={}, fix={}, sv={}, lat={:.7}, lon={:.7}, alt_msl={}mm, gspd={}mm/s, hacc={}mm, vacc={}mm, pdop={})",
            s.timestamp.as_millis(),
            s.fix_type,
            s.num_sv,
            s.lat_deg,
            s.lon_deg,
            s.alt_msl_mm,
            s.ground_speed_mm_s,
            s.h_acc_mm,
            s.v_acc_mm,
            s.pdop
        )
    }
}

impl fmt::Display for ShellMsg<'_, msgs::MagSample> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(
            f,
            "MagSample(t={}, field_ut=[{:.2},{:.2},{:.2}], temp_c={:.1})",
            s.timestamp.as_millis(),
            s.field_ut.x,
            s.field_ut.y,
            s.field_ut.z,
            s.temp_c
        )
    }
}

impl fmt::Display for ShellMsg<'_, msgs::BaroSample> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(
            f,
            "BaroSample(t={}, pressure_pa={:.2}, temp_c={:.2})",
            s.timestamp.as_millis(),
            s.pressure_pa,
            s.temp_c
        )
    }
}

impl fmt::Display for ShellMsg<'_, msgs::DshotTelemetry> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use cybflight_msgs::dshot::TelemetryValue;
        write!(f, "DshotTelem(t={}", self.0.timestamp.as_millis())?;
        for (i, m) in self.0.motors.iter().enumerate() {
            match m.value {
                TelemetryValue::Erpm(e) => write!(f, " M{}={}erpm", i + 1, e as u32 * 100)?,
                TelemetryValue::Stopped => write!(f, " M{}=stopped", i + 1)?,
                TelemetryValue::Edt(edt) => {
                    write!(f, " M{}=edt:{:?}={}", i + 1, edt.edt_type, edt.data)?
                }
                TelemetryValue::Invalid => write!(f, " M{}=invalid", i + 1)?,
            }
        }
        write!(f, ")")
    }
}

impl fmt::Display for ShellMsg<'_, msgs::AttitudeControlSetpoint> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(
            f,
            "AttitudeControlSetpoint(timestamp={:.4}, collective_thrust_n={:.4},attitude_quaternion=Quaternion(x={:.4},y={:.4},z={:.4},w={:.4}), body_rate_rad_s=[{:.4},{:.4},{:.4}], torque_n_m=[{:.4},{:.4},{:.4}])",
            s.timestamp.as_millis(),
            s.collective_thrust_n,
            s.attitude_quaternion.i,
            s.attitude_quaternion.j,
            s.attitude_quaternion.k,
            s.attitude_quaternion.w,
            s.body_rate_rad_s.x,
            s.body_rate_rad_s.y,
            s.body_rate_rad_s.z,
            s.torque_n_m.x,
            s.torque_n_m.y,
            s.torque_n_m.z
        )
    }
}

pub trait Printable {
    fn should_print(&self, ctx: &ShellState) -> bool;
    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result;
}

// Global toggle state for the shell
#[derive(Default)]
pub struct ShellState {
    pub stream_imu: bool,
    pub stream_att: bool,
    pub stream_odom: bool,
    pub stream_rc: bool,
    pub stream_rcstats: bool,
    pub stream_dshot: bool,
    pub stream_gps: bool,
    pub stream_magext: bool,
    pub stream_magint: bool,
    pub stream_baro1: bool,
    pub stream_baro2: bool,
    pub stream_attcontrol: bool,
}

impl Printable for msgs::Imu {
    fn should_print(&self, ctx: &ShellState) -> bool {
        ctx.stream_imu
    }

    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result {
        write!(w, "{}", ShellMsg(self))
    }
}

impl Printable for msgs::VehicleAttitude {
    fn should_print(&self, ctx: &ShellState) -> bool {
        ctx.stream_att
    }

    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result {
        write!(w, "{}", ShellMsg(self))
    }
}

impl Printable for msgs::VehicleOdometry {
    fn should_print(&self, ctx: &ShellState) -> bool {
        ctx.stream_odom
    }

    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result {
        write!(w, "{}", ShellMsg(self))
    }
}

impl Printable for msgs::RcInput {
    fn should_print(&self, ctx: &ShellState) -> bool {
        ctx.stream_rc
    }

    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result {
        write!(w, "{}", ShellMsg(self))
    }
}

impl Printable for msgs::RcLinkStatus {
    fn should_print(&self, ctx: &ShellState) -> bool {
        ctx.stream_rcstats
    }

    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result {
        write!(w, "{}", ShellMsg(self))
    }
}

impl Printable for msgs::OcpSolverOutput {
    fn should_print(&self, _ctx: &ShellState) -> bool {
        true
    }

    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result {
        write!(w, "{}", ShellMsg(self))
    }
}

impl Printable for msgs::DshotTelemetry {
    fn should_print(&self, ctx: &ShellState) -> bool {
        ctx.stream_dshot
    }

    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result {
        write!(w, "{}", ShellMsg(self))
    }
}

impl Printable for msgs::GpsFix {
    fn should_print(&self, ctx: &ShellState) -> bool {
        ctx.stream_gps
    }

    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result {
        write!(w, "{}", ShellMsg(self))
    }
}

impl Printable for msgs::MagSample {
    fn should_print(&self, ctx: &ShellState) -> bool {
        ctx.stream_magext
    }

    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result {
        write!(w, "{}", ShellMsg(self))
    }
}

impl Printable for msgs::BaroSample {
    fn should_print(&self, _ctx: &ShellState) -> bool {
        true
    }

    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result {
        write!(w, "{}", ShellMsg(self))
    }
}

impl Printable for msgs::AttitudeControlSetpoint {
    fn should_print(&self, ctx: &ShellState) -> bool {
        ctx.stream_attcontrol
    }

    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result {
        write!(w, "{}", ShellMsg(self))
    }
}
