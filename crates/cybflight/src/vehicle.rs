use cybflight_core::mixer::{
    LinearAllocator, MotorEffectiveness, MotorParams, RigidBodyParams, SpinDir,
};

// ---------------------------------------------------------------------------
// Quadrotor physical parameters — Betaflight QuadX motor ordering, FLU frame.
//
// Body frame: FLU (Forward-Left-Up): x = forward, y = left, z = up.
//
// Motor index convention (Betaflight QuadX):
//   0 = REAR_RIGHT  (CW  from above) — position (−d, −d): rear and right = negative y
//   1 = FRONT_RIGHT (CCW from above) — position (+d, −d)
//   2 = REAR_LEFT   (CCW from above) — position (−d, +d): left = positive y
//   3 = FRONT_LEFT  (CW  from above) — position (+d, +d)
//
// Arm length 100 mm at 45° → motor offset d = 0.1 / √2 ≈ 70.7 mm.
// ---------------------------------------------------------------------------

pub const QUADROTOR_BODY: RigidBodyParams = RigidBodyParams {
    mass_kg: 0.55,
    // Diagonal inertia [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz] (kg·m²).
    // Roll/pitch symmetric (Ixx = Iyy = 0.02), yaw larger (Izz = 0.04).
    // Calibrate from a bifilar pendulum test or CAD model.
    inertia_kg_m2: [0.0021, 0.0, 0.0, 0.0, 0.0018, 0.0, 0.0, 0.0, 0.003],
    max_rate_rad_s: [10.0, 10.0, 6.0],
};

const MAX_THRUST_N: f32 = 12.5;

pub const QUADROTOR_MOTORS: [MotorParams; 4] = [
    // M0: REAR_RIGHT — CW, position (−d, −d) in FLU (right = −y).
    MotorParams {
        position_m: [-0.075, -0.1],
        spin_dir: SpinDir::Cw,
        max_thrust_n: MAX_THRUST_N, // ~600 g per motor for a 5" prop. Calibrate from test stand.
        torque_coeff_m: 0.022,
    },
    // M1: FRONT_RIGHT — CCW, position (+d, −d) in FLU.
    MotorParams {
        position_m: [0.075, -0.1],
        spin_dir: SpinDir::Ccw,
        max_thrust_n: MAX_THRUST_N,
        torque_coeff_m: 0.022,
    },
    // M2: REAR_LEFT — CCW, position (−d, +d) in FLU (left = +y).
    MotorParams {
        position_m: [-0.075, 0.1],
        spin_dir: SpinDir::Ccw,
        max_thrust_n: MAX_THRUST_N,
        torque_coeff_m: 0.022,
    },
    // M3: FRONT_LEFT — CW, position (+d, +d) in FLU.
    MotorParams {
        position_m: [0.075, 0.1],
        spin_dir: SpinDir::Cw,
        max_thrust_n: MAX_THRUST_N,
        torque_coeff_m: 0.022,
    },
];

/// Compile-time default control gains matching the inner-loop hardcoded values.
pub const DEFAULT_CONTROL_GAINS: cybflight_core::params::ControlGains =
    cybflight_core::params::ControlGains {
        pos_kp: [4.0, 4.0, 8.0],
        pos_kd: [4.0, 4.0, 6.0],
        att_k_rate: [3.0, 3.0, 1.0],
    };

/// Return the compile-time default vehicle parameters.
pub fn default_params() -> cybflight_core::params::VehicleParams {
    cybflight_core::params::VehicleParams {
        body: QUADROTOR_BODY,
        motors: QUADROTOR_MOTORS,
        control: DEFAULT_CONTROL_GAINS,
        indi_effectiveness: cybflight_core::params::IndiEffectivenessParams::default(),
        indi_controller: cybflight_core::params::IndiControllerParams::default(),
        learner: cybflight_core::params::LearnerParams::default(),
        mpc: cybflight_core::params::MpcParams::default(),
        planner: cybflight_core::params::PlannerParams::default(),
    }
}

/// Construct the quadrotor linear allocator.
///
/// Called once at firmware startup (e.g. in `board_init` or the attitude task).
/// Panics at construction time if the motor geometry is degenerate.
pub fn quadrotor_allocator() -> LinearAllocator<4> {
    LinearAllocator::new(MotorEffectiveness::from_motors(&QUADROTOR_MOTORS))
}
