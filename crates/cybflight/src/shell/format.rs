use crate::msgs;

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
        write!(f, "RcInput(timestamp={}, channels=[", s.timestamp.as_millis())?;
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
