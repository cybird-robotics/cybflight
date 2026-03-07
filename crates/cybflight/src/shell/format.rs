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
