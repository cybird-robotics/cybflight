/// Active flight mode, selected by AUX3 RC switch.
#[derive(Clone, Copy, PartialEq, Eq, defmt::Format)]
pub enum FlightMode {
    /// Body rate control (default). RC sticks → body rates → rate PID → mixer.
    Acro,
    /// Attitude stabilization. RC sticks → attitude angles → attitude PID → rate PID → mixer.
    Angle,
    /// Autonomous mode. auto_task produces setpoints; low-level controller depends on setpoint type.
    Auto,
}

/// AUX3 channel index (ch7, zero-based index 6).
pub const MODE_CHANNEL: usize = 6;

/// Threshold values for AUX3 three-position switch.
pub const MODE_AUTO_MAX: u16 = 1300;
pub const MODE_ANGLE_MAX: u16 = 1700;

/// Hysteresis margin (µs) applied to mode thresholds to prevent oscillation.
pub const HYSTERESIS: u16 = 50;

// Setpoint published by auto_task. Tagged enum tells flight_controller_task
// which low-level controller to engage.
// #[derive(Clone, Copy)]
// pub enum AutoSetpoint {
//     /// From PD position controller: desired attitude + collective thrust.
//     /// flight_controller runs: attitude PID → rate PID → mixer.
//     AttitudeThrust {
//         attitude: UnitQuaternion<f32>,
//         body_rate: Vector3<f32>,
//         collective_thrust_n: f32,
//     },
//     /// From NMPC: desired thrust + body rate command.
//     /// flight_controller runs: rate PID → mixer (skip attitude PID).
//     ThrustRate {
//         collective_thrust_n: f32,
//         body_rate: Vector3<f32>,
//     },
//     /// From NMPC: per-rotor thrust commands.
//     /// flight_controller runs: INDI get_motor_commands() → motors directly.
//     RotorThrusts {
//         rotor_thrusts: [f32; 4],
//     },
// }
