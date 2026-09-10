// Motor mixer: physical effectiveness model and thrust/torque allocation.
//
// # Coordinate frame
// Body frame is FLU (Forward-Left-Up): x = forward, y = left, z = up.
// Positive roll  = left side up   (right-hand rotation around +x).
// Positive pitch = nose up        (right-hand rotation around +y).
// Positive yaw   = CCW from above (right-hand rotation around +z).
//
// Note: Betaflight uses NED/FRD (Forward-Right-Down). When comparing mixer
// coefficients, the yaw sign is inverted and the y-axis (roll) sign is inverted
// relative to Betaflight's `motorMixer_t` table.
//
// # Motor indexing (Betaflight QuadX convention)
// Index 0 = REAR_RIGHT  (CW  from above)
// Index 1 = FRONT_RIGHT (CCW from above)
// Index 2 = REAR_LEFT   (CCW from above)
// Index 3 = FRONT_LEFT  (CW  from above)
//
// # Allocation model
// The G1 effectiveness matrix maps per-motor throttle commands u ∈ [0,1]^N to
// virtual control v = [collective_thrust_N, roll_Nm, pitch_Nm, yaw_Nm]:
//
//   v = G1 · u
//
// Columns of G1 are motor contributions at full throttle (u_i = 1.0):
//   G1[0, i] = max_thrust_n[i]
//   G1[1, i] = +py[i] · max_thrust_n[i]           (roll torque, FLU: +y is left)
//   G1[2, i] = −px[i] · max_thrust_n[i]           (pitch torque)
//   G1[3, i] = spin_sign[i] · c[i] · max_thrust_n[i]  (reaction yaw torque)
//
// Derivation: thrust force F = (0, 0, +T) in FLU; τ = r × F where r = (px, py, 0).
//   τ_x = py·T,  τ_y = −px·T.
// Yaw reaction: CW-from-above motor (negative ω_z) → reaction in +z → spin_sign = +1.
//
// The pseudoinverse (G1⁺) maps demanded v → motor throttles u:
//   u = G1⁺ · v
//
// # Future: Indiflight active-set WLS allocator
//
// `LinearAllocator` computes the minimum-norm unconstrained solution.
// To onboard Indiflight's active-set WLS (`solveActiveSet`), do the following:
//
// 1. **RPM telemetry prerequisite.** Indiflight's `actG2` (motor rate-dependent
//    thrust term) requires DShot300+ bidirectional RPM telemetry. Enable this
//    first (`cfg(feature = "indi")`). Without it, `actG2` must stay zero-filled.
//
// 2. **Port `wls_alloc`.** Extract Indiflight's `src/main/flight/mixer_init.c`
//    `solveActiveSet()` as a standalone `no_std` crate. The solver signature is:
//      `wls_alloc(g1: &SMatrix<f32,4,N>, v: &Vector4<f32>,
//                 u_min: &SVector<f32,N>, u_max: &SVector<f32,N>,
//                 w_v: &Vector4<f32>, w_u: &SVector<f32,N>,
//                 gamma: f32) -> SVector<f32,N>`
//    Note: Indiflight's G1 is stored in FRD/NED; flip yaw and roll columns when
//    porting, or re-derive G1 in FLU (this module already uses FLU).
//
// 3. **Replace `allocate()`.** Swap `g1_pinv * demand` with:
//      `wls_alloc(&self.effectiveness.g1, &(demand - g2 * omega_dot_prev),
//                 &u_min, &u_max, &w_v, &w_u, gamma)`
//    where `u_min = zeros`, `u_max = ones`, and weights are tuned per-vehicle.
//
// 4. **No interface changes needed.** `MotorEffectiveness` already stores `g1`
//    in the correct format (4×N, physical units). The `LinearAllocator::allocate`
//    signature is unchanged; only the inner solver is swapped.

use nalgebra as na;

/// Direction a propeller spins, viewed from above the vehicle
/// (i.e., looking in the −z direction in FLU, from sky toward ground).
///
/// In FLU the right-hand rule gives:
/// - CCW from above = **positive** rotation about z  (ω_z > 0)
/// - CW  from above = **negative** rotation about z  (ω_z < 0)
///
/// The yaw reaction torque on the body is always opposite to the propeller spin:
/// a CW propeller (ω_z < 0) applies a +z reaction torque on the body (positive yaw).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpinDir {
    /// Clockwise from above: ω_z < 0.
    /// Reaction torque on body is in +z (positive yaw = CCW in FLU).
    Cw = 1,
    /// Counter-clockwise from above: ω_z > 0.
    /// Reaction torque on body is in −z (negative yaw = CW in FLU).
    Ccw = -1,
}

/// Flash/shell encoding for [`SpinDir`]: Cw = 1, Ccw = 0 (matches the
/// pre-registry blob encoding). Out-of-range values map to Ccw.
impl crate::param_registry::ParamEnum for SpinDir {
    fn to_u8(self) -> u8 {
        match self {
            SpinDir::Cw => 1,
            SpinDir::Ccw => 0,
        }
    }
    fn from_u8(v: u8) -> Self {
        if v == 1 { SpinDir::Cw } else { SpinDir::Ccw }
    }
}

/// Motor time constant [s] used when a vehicle does not identify its own —
/// the historical `INDI_MOTOR_PARAMS` const. A zero here is structurally
/// meaningless (the G2 scaler divides by it), so unlike mass or geometry
/// this field carries a real default rather than a zero sentinel.
pub const DEFAULT_MOTOR_TAU_S: f32 = 0.02;

/// Motor max speed [rad/s] used when a vehicle does not identify its own:
/// 40 000 RPM, the historical `INDI_MOTOR_PARAMS` const. Same
/// no-zero-sentinel reasoning as [`DEFAULT_MOTOR_TAU_S`].
pub const DEFAULT_MOTOR_MAX_OMEGA_RAD_S: f32 = 4188.79;

/// Physical and geometric parameters for one motor + propeller combination.
///
/// Everything here is an *actuator fact* — measured from the motor, ESC and
/// propeller, independent of which controller happens to read it. The
/// dynamics fields (`tau`, `omega_max`, `g2_*`, `nonlin`) live here rather
/// than under an `indi_` prefix for exactly that reason: they would be
/// identical under a geometric or rate controller, and INDI is merely their
/// only current consumer.
///
/// Two flavours of field, distinguished by how they reach the firmware:
///
/// - **Geometry/identity** (`px`, `py`, `spin`, `thrust`, `torque`) comes
///   from the REQUIRED `airframe.motors` list in the vehicle YAML and has
///   no default — a wrong value flies and looks plausible, so it must be
///   stated.
/// - **Identified dynamics** (`tau`, `omega_max`, `g2_*`, `nonlin`) is
///   optional, carries a real default (or a documented zero sentinel), and
///   is pinned per vehicle under `tuning:` once bench-identified.
///
/// Param keys are exposed with a per-motor prefix (`m0_px`, `m1_spin`, …)
/// via the `nested_array` field in `AirframeParams`.
#[derive(Clone, Copy, Debug, cybflight_params_derive::Params)]
pub struct MotorParams {
    /// Motor position in body XY plane [x_m, y_m] in FLU frame.
    /// x = forward, y = left.
    #[param(keys = "px,py", unit = "m", min = -1.0, max = 1.0, reboot)]
    pub position_m: [f32; 2],
    /// Propeller spin direction (viewed from above).
    #[param(enum_u8, key = "spin", reboot)]
    pub spin_dir: SpinDir,
    /// Maximum thrust this motor+propeller produces at full throttle (Newtons).
    #[param(key = "thrust", unit = "N", min = 0.1, max = 200.0, reboot)]
    pub max_thrust_n: f32,
    /// Reaction torque per unit thrust (metres). Ratio of yaw reaction torque to
    /// thrust force. Typically 0.005–0.02 for a 5" propeller.
    #[param(key = "torque", unit = "m", min = 0.0, max = 0.2, reboot)]
    pub torque_coeff_m: f32,
    /// First-order spool-up time constant of motor+ESC+prop [s], from a
    /// throttle-step bench test. Feeds the INDI G2 scaler
    /// (`ω_max²/(2·τ)`), the actuator-state PT1 and the RPM estimator.
    #[param(key = "tau", unit = "s", min = 0.001, max = 1.0)]
    pub time_const_s: f32,
    /// Motor speed at full throttle [rad/s] — Kv × pack voltage, or read
    /// off RPM telemetry at full stick.
    #[param(key = "omega_max", unit = "rad/s", min = 50.0, max = 20000.0)]
    pub max_omega_rad_s: f32,
    /// G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular
    /// acceleration induced by *rotor* angular acceleration (rotor polar
    /// inertia against vehicle inertia). Only the yaw entry is non-zero
    /// for a standard multirotor, and its sign follows spin direction
    /// (CW +, CCW −) — which is why this stays per-motor rather than
    /// collapsing to one scalar.
    ///
    /// Range mirrors the controller's `G_MAG_MAX` (±1e4) so an
    /// implausible magnitude is rejected at the write, not first
    /// discovered by the apply-path degrade.
    #[param(keys = "g2_rr,g2_rp,g2_ry", min = -1e4, max = 1e4)]
    pub g2: [f32; 3],
    /// Thrust-curve nonlinearity `k` for this motor+prop. **0 selects the
    /// compile-time fallback matched to the vehicle's `thrust_model`** —
    /// the one deliberate zero sentinel here.
    ///
    /// Range mirrors the hard clamp in
    /// `indi::linearization::ThrustLinearization::new` (`[0.025, 1.0]`):
    /// below the floor the quadratic inverse degenerates, and 1.0 is the
    /// physical ceiling (thrust ∝ d² exactly). The two must stay equal —
    /// advertising a wider range here would let a value validate, persist
    /// and display while the controller silently flew the clamped one.
    #[param(key = "nonlin", min = 0.025, max = 1.0)]
    pub nonlinearity: f32,
}

/// Vehicle rigid-body parameters.
#[derive(Clone, Copy, Debug, cybflight_params_derive::Params)]
pub struct RigidBodyParams {
    /// Vehicle mass (kg).
    #[param(key = "mass", unit = "kg", min = 0.05, max = 20.0, reboot)]
    pub mass_kg: f32,
    /// Inertia tensor stored row-major: [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz].
    /// For a symmetric body, off-diagonal terms are zero.
    /// Stored as a flat array for `const`-compatible initialization.
    #[param(keys = "ixx,ixy,ixz,iyx,iyy,iyz,izx,izy,izz", unit = "kg·m²", min = -1.0, max = 1.0, reboot)]
    pub inertia_kg_m2: [f32; 9],
    #[param(keys = "max_rate_r,max_rate_p,max_rate_y", unit = "rad/s", min = 0.1, max = 50.0, reboot)]
    pub max_rate_rad_s: [f32; 3],
}

impl MotorParams {
    /// Zeroed geometry plus stock actuator dynamics — the base for
    /// struct-update construction where only geometry is being stated:
    ///
    /// ```ignore
    /// MotorParams { position_m: [-0.075, -0.1], spin_dir: SpinDir::Cw,
    ///               max_thrust_n: 12.0, torque_coeff_m: 0.022,
    ///               ..MotorParams::STOCK_DYNAMICS }
    /// ```
    ///
    /// This is *not* a `Default` impl, deliberately: geometry must always
    /// be stated (a zero mass or motor arm is unflyable by construction),
    /// so there is no whole-struct default to reach for by accident.
    pub const STOCK_DYNAMICS: Self = Self {
        position_m: [0.0, 0.0],
        spin_dir: SpinDir::Ccw,
        max_thrust_n: 0.0,
        torque_coeff_m: 0.0,
        time_const_s: DEFAULT_MOTOR_TAU_S,
        max_omega_rad_s: DEFAULT_MOTOR_MAX_OMEGA_RAD_S,
        g2: [0.0; 3],
        nonlinearity: 0.0,
    };
}

impl RigidBodyParams {
    /// Extract the inertia tensor as a 3×3 matrix.
    pub fn inertia_matrix(&self) -> na::Matrix3<f32> {
        na::Matrix3::from_row_slice(&self.inertia_kg_m2)
    }
}

/// Pre-computed effectiveness model for an N-motor vehicle.
///
/// Stores G1 (4×N) and its right pseudoinverse G1⁺ (N×4). Both are computed
/// once at construction time; `allocate_unclamped` is a single matrix multiply.
pub struct MotorEffectiveness<const N: usize> {
    /// G1 effectiveness matrix (4×N). Columns are motors; rows are
    /// [collective_thrust, roll_torque, pitch_torque, yaw_torque] at full throttle.
    pub g1: na::SMatrix<f32, 4, N>,
    /// Right pseudoinverse of G1 (N×4).
    /// G1⁺ = G1ᵀ (G1 G1ᵀ)⁻¹ — minimum-norm solution for any N ≥ 4.
    g1_pinv: na::SMatrix<f32, N, 4>,
}

impl<const N: usize> MotorEffectiveness<N> {
    /// Build the effectiveness model from an array of motor parameters.
    ///
    /// # Panics
    /// Panics at construction time if the motor geometry is degenerate (singular G1 G1ᵀ).
    /// This can only happen if motors are co-located or produce identical torque vectors.
    pub fn from_motors(motors: &[MotorParams; N]) -> Self {
        let mut g1 = na::SMatrix::<f32, 4, N>::zeros();

        for (i, m) in motors.iter().enumerate() {
            let [px, py] = m.position_m;
            let t = m.max_thrust_n;
            // CW from above = ω_z < 0.  Drag opposes rotation → drag on prop is in +z.
            // Body reaction (Newton 3) is also in +z → spin_sign = +1.
            // CCW from above = ω_z > 0.  Drag in −z → body reaction in −z → spin_sign = −1.
            let spin_sign = m.spin_dir as i32 as f32;

            // Thrust: F = (0, 0, +T) in FLU (upward = +z).
            // Torque: τ = r × F = (py·T, −px·T, 0) + spin reaction.
            g1.column_mut(i).copy_from(&na::Vector4::new(
                t,
                py * t,  // roll torque
                -px * t, // pitch torque
                spin_sign * m.torque_coeff_m * t,
            ));
        }

        // Right pseudoinverse: G1⁺ = G1ᵀ (G1 G1ᵀ)⁻¹.
        // For N=4 this equals G1⁻¹. For N>4 it gives the minimum-norm solution.
        let g1_g1t: na::Matrix4<f32> = g1 * g1.transpose();
        let g1_g1t_inv = g1_g1t
            .try_inverse()
            .expect("mixer: motor geometry is degenerate — G1·G1ᵀ is singular");
        let g1_pinv: na::SMatrix<f32, N, 4> = g1.transpose() * g1_g1t_inv;

        Self { g1, g1_pinv }
    }

    /// Allocate motor throttle commands (unclamped) for a demanded virtual control vector.
    ///
    /// `demand` = [collective_thrust_N, roll_Nm, pitch_Nm, yaw_Nm].
    ///
    /// Returns per-motor throttle commands. Values may exceed [0, 1] — caller
    /// (or `LinearAllocator::allocate`) is responsible for saturation.
    #[inline]
    pub fn allocate_unclamped(&self, demand: na::Vector4<f32>) -> na::SVector<f32, N> {
        self.g1_pinv * demand
    }

    /// Sum of maximum thrusts across all motors (Newtons). Equals G1 row 0 sum.
    pub fn max_collective_thrust_n(&self) -> f32 {
        self.g1.row(0).iter().copied().sum()
    }
}

/// Linear (pseudoinverse-based) motor allocator.
///
/// Converts a [thrust, roll, pitch, yaw] demand into per-motor throttle commands
/// in [0, 1]. The allocation minimises the 2-norm of the throttle vector, which
/// distributes load evenly across motors when the vehicle is over-actuated (N > 4).
///
/// See the module-level comment for the active-set WLS upgrade path.
pub struct LinearAllocator<const N: usize> {
    effectiveness: MotorEffectiveness<N>,
}

impl<const N: usize> LinearAllocator<N> {
    pub fn new(effectiveness: MotorEffectiveness<N>) -> Self {
        Self { effectiveness }
    }

    /// Allocate and saturate motor throttle commands.
    ///
    /// `demand` = [collective_thrust_N, roll_Nm, pitch_Nm, yaw_Nm] in FLU frame.
    ///
    /// Returns per-motor throttle in [0.0, 1.0]. Note: simple saturation does not
    /// re-distribute the allocation error to remaining motors. Upgrade to WLS for
    /// prioritised desaturation (see module-level comment).
    #[inline]
    pub fn allocate(&self, demand: na::Vector4<f32>) -> na::SVector<f32, N> {
        self.effectiveness
            .allocate_unclamped(demand)
            .map(|v| v.clamp(0.0, 1.0))
    }

    /// Maximum collective thrust the vehicle can produce (N).
    /// Use this to convert a normalized thrust command [0, 1] to Newtons.
    pub fn max_collective_thrust_n(&self) -> f32 {
        self.effectiveness.max_collective_thrust_n()
    }

    /// Reference to the underlying effectiveness model (e.g. for diagnostics).
    pub fn effectiveness(&self) -> &MotorEffectiveness<N> {
        &self.effectiveness
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // QuadX layout in FLU: arm = 0.1 m, 45° → d = 0.1/√2.
    // y = left, so right motors have negative y.
    const D: f32 = 0.070_71;
    const C: f32 = 0.01;
    const T_MAX: f32 = 5.9;

    fn quad_motors() -> [MotorParams; 4] {
        [
            // M0 REAR_RIGHT:  x=−d, y=−d (rear, right = negative y in FLU)
            MotorParams {
                position_m: [-D, -D],
                spin_dir: SpinDir::Cw,
                max_thrust_n: T_MAX,
                torque_coeff_m: C,
                ..MotorParams::STOCK_DYNAMICS
            },
            // M1 FRONT_RIGHT: x=+d, y=−d
            MotorParams {
                position_m: [D, -D],
                spin_dir: SpinDir::Ccw,
                max_thrust_n: T_MAX,
                torque_coeff_m: C,
                ..MotorParams::STOCK_DYNAMICS
            },
            // M2 REAR_LEFT:   x=−d, y=+d (left = positive y in FLU)
            MotorParams {
                position_m: [-D, D],
                spin_dir: SpinDir::Ccw,
                max_thrust_n: T_MAX,
                torque_coeff_m: C,
                ..MotorParams::STOCK_DYNAMICS
            },
            // M3 FRONT_LEFT:  x=+d, y=+d
            MotorParams {
                position_m: [D, D],
                spin_dir: SpinDir::Cw,
                max_thrust_n: T_MAX,
                torque_coeff_m: C,
                ..MotorParams::STOCK_DYNAMICS
            },
        ]
    }

    #[test]
    fn g1_structure() {
        let eff = MotorEffectiveness::from_motors(&quad_motors());

        // All motors contribute equally to collective thrust.
        for i in 0..4 {
            assert!(
                (eff.g1[(0, i)] - T_MAX).abs() < 1e-4,
                "thrust row motor {i}"
            );
        }

        // Yaw row: CW motors positive, CCW motors negative (FLU convention).
        let expected_yaw = [C * T_MAX, -C * T_MAX, -C * T_MAX, C * T_MAX];
        for i in 0..4 {
            assert!(
                (eff.g1[(3, i)] - expected_yaw[i]).abs() < 1e-4,
                "yaw row motor {i}"
            );
        }
    }

    #[test]
    fn round_trip_collective_thrust() {
        let alloc = LinearAllocator::new(MotorEffectiveness::from_motors(&quad_motors()));
        // Demand: hover thrust, no torques.
        let hover_n = 14.715; // 1.5 kg * 9.81 m/s²
        let demand = na::Vector4::new(hover_n, 0.0, 0.0, 0.0);
        let throttles = alloc.allocate(demand);

        // All motors should get equal throttle for a symmetric quad.
        let expected = hover_n / alloc.max_collective_thrust_n();
        for i in 0..4 {
            assert!((throttles[i] - expected).abs() < 1e-4, "throttle motor {i}");
        }
    }

    #[test]
    fn round_trip_yaw_torque() {
        let alloc = LinearAllocator::new(MotorEffectiveness::from_motors(&quad_motors()));
        let hover_n = 14.715_f32;
        let yaw_nm = 0.05_f32;
        let demand = na::Vector4::new(hover_n, 0.0, 0.0, yaw_nm);
        let throttles = alloc.allocate(demand);

        // Verify G1 * throttles ≈ demand.
        let achieved = alloc.effectiveness().g1 * throttles;
        assert!((achieved[0] - hover_n).abs() < 1e-3);
        assert!((achieved[3] - yaw_nm).abs() < 1e-3);
    }

    #[test]
    fn max_collective_thrust() {
        let alloc = LinearAllocator::new(MotorEffectiveness::from_motors(&quad_motors()));
        assert!((alloc.max_collective_thrust_n() - 4.0 * T_MAX).abs() < 1e-4);
    }

    #[test]
    fn rigid_body_inertia_matrix() {
        let body = RigidBodyParams {
            mass_kg: 1.5,
            inertia_kg_m2: [0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.04],
            max_rate_rad_s: [10.0, 10.0, 6.0],
        };
        let mat = body.inertia_matrix();
        assert_eq!(mat[(0, 0)], 0.02);
        assert_eq!(mat[(2, 2)], 0.04);
        assert_eq!(mat[(0, 1)], 0.0);
    }
}
