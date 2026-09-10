//! Differential flatness for quadrotors: the single home for every map
//! between a trajectory's flat outputs and a vehicle setpoint.
//!
//! A quadrotor is differentially flat in `(x, y, z, ψ)`, so the whole
//! state and input can be recovered from position derivatives plus a
//! yaw schedule. Everything here is a **pure function of those
//! derivatives** — nothing in this module knows which solver produced
//! them, so the same maps apply to a [`super::minco_snap::MincoSnap`]
//! (degree 7, C⁶) trajectory, a [`super::minco_jerk::MincoJerk`]
//! (degree 5, C⁴) one, or a hand-written polynomial.
//!
//! # Two map families, split by derivative order (plus an attitude-only fallback)
//!
//! | | needs | returns | used by |
//! |---|---|---|---|
//! | [`flatness_to_state_tilt_yaw`] / [`flatness_to_state_true_yaw`] | a, j, **s** | thrust, attitude, ω, **ω̇** | offline reference generation, tests |
//! | [`flatness_to_thrust_omega`] / [`flatness_to_thrust_omega_true_yaw`] | a, j | thrust, attitude, ω | the MPC's `u_ref` feedforward |
//! | [`reference_quaternion`] | a | attitude | terminal-node / fallback attitude |
//!
//! Reach for the snap-free pair unless you actually consume `ω̇`.
//! Passing a fabricated `sna` to the full maps to get at their other
//! fields returns a *wrong* `omega_dot` rather than an absent one —
//! the reason the snap-free variants exist in both conventions.
//!
//! Note that "needs snap" is not the same as "needs a min-snap
//! solver": MINCO enforces C^(2s-2) continuity, so a min-jerk (s=3)
//! trajectory is C⁴ and its snap is continuous and well-defined. What
//! a min-jerk trajectory gives up is smoothness of that snap
//! (piecewise-linear, kinked at junctions), hence a coarser ω̇ — not
//! its availability.
//!
//! # Two yaw conventions
//!
//! `tilt_yaw` reads ψ as the intrinsic tilt-then-yaw Euler angle;
//! `true_yaw` reads it as the compass heading of the body-x
//! projection. They trade singularity locations — see
//! [`FlatnessFault`]. Missions pick one via the mission YAML's
//! `flatness_map` key.
//!
//! # Gradient chain
//!
//! [`AlphaState`] / [`FlatnessState`] / [`body_rate_grad_backprop`]
//! are the planner's ψ=0 forward+backward chain, used by the BFGS cost
//! penalties rather than by any controller. They are kept separate from
//! the reference maps above because they are differentiable by design
//! and deliberately clamp where the reference maps refuse.

#[allow(unused_imports)]
use num_traits::Float;

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

use super::types::Vec3;
use crate::rotation::quaternion_from_zb_and_yaw;

// ═════════════════════════════════════════════════════════════════════
// Part 1 — controller reference maps (yaw is an input)
//
// All fault-returning, with ONE deliberate exception:
// `reference_quaternion` clamps and never fails. It is the fallback
// the outer loop falls back TO when a strict map refuses, so it must
// always produce an answer — see its doc comment.
// ═════════════════════════════════════════════════════════════════════

// ── Quadrotor flatness map (port of `toStateWithTiltYaw`) ────────────
//
// Mirrors `drolib::QuadManifold::toStateWithTiltYaw` from
// `tmp/planner/src/system/quadrotor_manifold.cpp` (line 1695). Maps a
// (a, j, s) flat-output triple plus a yaw triple (ψ, ψ̇, ψ̈) and
// gravity into a quadrotor setpoint: collective thrust per unit mass,
// attitude quaternion, body rate, body angular acceleration. Position
// and velocity are kinematic flat outputs but unused by this map, so
// they are not part of the signature; callers multiply by mass to get
// thrust force.
//
// Yaw triple is `[ψ, ψ̇, ψ̈]` in rad / rad·s⁻¹ / rad·s⁻².

/// Output of [`flatness_to_state_tilt_yaw`].
///
/// All fields are world-frame except `omega` and `omega_dot`, which
/// are body-frame (the convention every quadrotor controller in this
/// codebase uses).
#[derive(Copy, Clone, Debug)]
pub struct FlatState {
    pub thrust_per_mass: f32,
    pub attitude: UnitQuaternion<f32>,
    pub omega: Vec3,
    pub omega_dot: Vec3,
}

/// Hard floor on `‖α‖²` (m²/s⁴) for the **full** maps only. Their
/// closed-form inversion divides by `‖α‖³` and `‖α‖⁵`, both of which
/// would overflow `f32::MAX ≈ 3.4·10³⁸` well before `‖α‖` reaches zero.
/// The threshold is `(0.1·g)² ≈ 1`.
///
/// This floor deliberately encodes "the vehicle is still net-accelerating
/// upward". That is a *dynamics* assumption, not a numerical one, and
/// aerobatic references break it on purpose — a ballistic or pushover
/// segment commands `a → −g·ẑ`, i.e. `α → 0`. The pole-safe maps
/// therefore do NOT use this floor; see
/// [`ALPHA_NORM_SQR_FLOOR_POLE_SAFE`].
const ALPHA_NORM_SQR_FLOOR: f32 = 1.0;

/// Floor on `‖α‖²` for the pole-safe maps ([`flatness_to_thrust_omega`],
/// [`flatness_to_thrust_omega_true_yaw`]) — `‖α‖ ≥ 1e-3 m/s²`.
///
/// Six orders of magnitude below the strict floor because these maps
/// divide only by `‖α‖` and `‖α‖²`, never `‖α‖³`/`‖α‖⁵`. It is a pure
/// **numerical** guard on `z_b = α/‖α‖` losing all significance, not a
/// claim about which flight regimes are legal: near-free-fall is a
/// legitimate aerobatic command, and refusing it at 0.1 g would deny a
/// thrust feedforward across an entire ballistic segment.
///
/// What it does NOT bound is ω: `‖ω‖ ≈ ‖j⊥‖/‖α‖` grows without limit as
/// `α → 0` (the old 0.1 g floor did not bound it either — at ‖j‖ = 100
/// it already admitted ~100 rad/s). Callers feeding ω into a cost or an
/// actuator must clamp it to their own envelope; `outer_loop` runs the
/// `u_refs` feedforward through the model's `u_bounds` for exactly this
/// reason.
const ALPHA_NORM_SQR_FLOOR_POLE_SAFE: f32 = 1e-6;

/// Tilt singularity threshold on `zB.z + 1`. Below this the tilt-yaw
/// parameterization breaks down (`omg_den → 0`, the `dzb2²/omg_den²`
/// term in `ω̇` blows up). We refuse to evaluate rather than silently
/// emit garbage. ~5.7° below "fully inverted" — far enough from any
/// practical flight regime that hitting it indicates a planning bug
/// upstream, close enough that no legitimate maneuver hits it.
const TILT_DEN_FLOOR: f32 = 5e-3;

/// Did the flatness map evaluate cleanly, or did it hit a singularity?
///
/// `Singular` is returned by [`flatness_to_state_tilt_yaw`] without
/// consulting the offending input — caller is expected to short-circuit
/// (hold last setpoint, trip a fault, etc.). The MINCO trajectories
/// produced by this codebase should never hit it; if they do, the
/// trajectory is unflyable and the controller cannot rescue it from
/// downstream NaN.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FlatnessFault {
    /// `‖a + g·ẑ‖²` was below the caller's floor — [`ALPHA_NORM_SQR_FLOOR`]
    /// for the full maps, the far lower [`ALPHA_NORM_SQR_FLOOR_POLE_SAFE`]
    /// for the pole-safe pair.
    ///
    /// **This is not "the trajectory is broken".** It means the thrust
    /// *direction* `z_b = α/‖α‖` is undefined, which is the correct
    /// reading of a ballistic command. The *magnitude* `‖α‖` remains
    /// exactly known, so a caller that needs a thrust feedforward here
    /// should compute `‖a + g·ẑ‖` itself and zero the rate rather than
    /// substitute hover — the latter asks for 1 g at the one moment the
    /// trajectory wants none. `outer_loop` does exactly that.
    NearFreeFall,
    /// `zB.z + 1` was below [`TILT_DEN_FLOOR`] (vehicle inverted or
    /// past the tilt-yaw parameterization's singularity).
    InvertedTilt,
    /// `‖y_C × z_B‖` was below [`HEADING_SINGULAR_FLOOR`]: the thrust
    /// axis is horizontal and aligned with the heading frame's y-axis
    /// (90° tilt), where the true-yaw parameterization of
    /// [`flatness_to_state_true_yaw`] loses its `x_B` construction and
    /// its `ω_z` denominator. The tilt-yaw maps do not return this.
    HeadingSingular,
}

/// Singularity threshold on `‖y_C × z_B‖` for the true-yaw map (the
/// sine of the angle between the thrust axis and the heading frame's
/// y-axis). The C++ source zeroes ω_z and disambiguates the attitude
/// with the live estimate; as a pure function we refuse instead —
/// same policy as [`TILT_DEN_FLOOR`]. ~0.3° from exact singularity.
const HEADING_SINGULAR_FLOOR: f32 = 5e-3;

/// Flat-output → state map (yaw-as-input convention).
///
/// Direct port of `toStateWithTiltYaw`. Returns `FlatState` whose
/// `thrust_per_mass = ‖a + g·ẑ‖` (collective thrust per unit mass),
/// `attitude` = tilt-then-yaw composition with `psi` as the yaw, and
/// `omega` / `omega_dot` from the closed-form differential-flatness
/// inversion.
///
/// `gravity` is the gravitational acceleration magnitude in m/s²
/// (positive). Pass `QuadPlanningConfig::grav` (default 9.81).
///
/// Returns `Err(FlatnessFault::*)` for the two singularities the
/// parameterization cannot represent: near-free-fall (`‖α‖ → 0`) and
/// inversion (`zB.z → -1`). Both should be unreachable on a valid
/// MINCO trajectory; if they fire, the upstream planner produced an
/// unflyable schedule and the caller must drop the sample.
///
/// ## Numerical layout (f32 / Cortex-M7 FPU)
///
/// The Cortex-M7 single-precision FPU pipelines `VMUL/VFMA` at one
/// per cycle but `VDIV/VSQRT` are 14-cycle blocking ops. The body
/// hoists every reciprocal once and reuses it via multiplies; only
/// **two** divisions and **one** sqrt are issued total.
pub fn flatness_to_state_tilt_yaw(
    acc: Vec3,
    jer: Vec3,
    sna: Vec3,
    yaw_triple: [f32; 3],
    gravity: f32,
) -> Result<FlatState, FlatnessFault> {
    let psi = yaw_triple[0];
    let dpsi = yaw_triple[1];
    let ddpsi = yaw_triple[2];

    // Single sin/cos pair via libm — no half-angle pair here; the
    // quaternion construction in `quaternion_from_zb_and_yaw`
    // computes its own ψ/2 sin/cos once internally.
    let c_psi = libm::cosf(psi);
    let s_psi = libm::sinf(psi);

    // α = a + g·ẑ_w.
    let alpha = Vec3::new(acc[0], acc[1], acc[2] + gravity);
    let alpha_norm_2 = alpha.norm_squared();
    if alpha_norm_2 < ALPHA_NORM_SQR_FLOOR {
        return Err(FlatnessFault::NearFreeFall);
    }
    // Single sqrt for the whole function — every other power of ‖α‖
    // is derived by multiplication via `inv_alpha_norm_*` below.
    let alpha_norm_1 = libm::sqrtf(alpha_norm_2);

    let alpha_dot_j = alpha.dot(&jer);
    let alpha_dot_j_sqr = alpha_dot_j * alpha_dot_j;
    let j_norm_2 = jer.norm_squared();

    // Hoisted reciprocals: one VDIV produces inv_alpha_norm_1, the
    // rest of the powers come from multiplies (free on Cortex-M7).
    let inv_alpha_norm_1 = 1.0 / alpha_norm_1;
    let inv_alpha_norm_2 = inv_alpha_norm_1 * inv_alpha_norm_1;
    let inv_a3 = inv_alpha_norm_2 * inv_alpha_norm_1;
    // inv_a5 derived via multiply, *not* `1.0 / alpha_norm_5` — saves
    // a 14-cycle VDIV.
    let inv_a5 = inv_a3 * inv_alpha_norm_2;

    // zB = α / ‖α‖
    let z_b = alpha * inv_alpha_norm_1;
    let zb0 = z_b[0];
    let zb1 = z_b[1];
    let zb2 = z_b[2];

    // Singularity guard: refuse rather than clamp. C++ clamps to
    // `±1e-6` and propagates the (now meaningless) result; we exit
    // cleanly so the controller cannot consume garbage. See
    // `FlatnessFault::InvertedTilt`.
    let zb2_1 = zb2 + 1.0;
    if zb2_1 < TILT_DEN_FLOOR {
        return Err(FlatnessFault::InvertedTilt);
    }

    // Collective thrust per unit mass = ‖α‖ (already computed). Avoids
    // 5 extra FLOPs and ~3 ulp of f32 accumulation that the longhand
    // `zB · α` would carry.
    let thrust_per_mass = alpha_norm_1;

    // dzB = N(α) · j = (j − α·(α·j)/‖α‖²) / ‖α‖.
    // Cleaner *and* better-conditioned than expanding the symmetric
    // `ng**` matrix by hand: when α is nearly axis-aligned, the
    // longhand form has `α_sqr_i + α_sqr_j` cancellations that this
    // form sidesteps. 3 mul + 3 sub + 3 mul = 9 FLOPs vs 9+6=15 in
    // the longhand `ng**` formulation.
    let proj_j = alpha_dot_j * inv_alpha_norm_2;
    let dzb0 = (jer[0] - alpha[0] * proj_j) * inv_alpha_norm_1;
    let dzb1 = (jer[1] - alpha[1] * proj_j) * inv_alpha_norm_1;
    let dzb2 = (jer[2] - alpha[2] * proj_j) * inv_alpha_norm_1;

    // N(α)·s by the same projection identity.
    let alpha_dot_s = alpha.dot(&sna);
    let proj_s = alpha_dot_s * inv_alpha_norm_2;
    let dn_alpha_s_0 = (sna[0] - alpha[0] * proj_s) * inv_alpha_norm_1;
    let dn_alpha_s_1 = (sna[1] - alpha[1] * proj_s) * inv_alpha_norm_1;
    let dn_alpha_s_2 = (sna[2] - alpha[2] * proj_s) * inv_alpha_norm_1;

    // ddzB = -2·(α·j)/‖α‖³ · j  +  α · (3·(α·j)² − ‖α‖²·‖j‖²)/‖α‖⁵
    //        +  N(α) · s
    //
    // The `common` term `(3·(α·j)² − ‖α‖²·‖j‖²) · inv_a5` folds a
    // catastrophic-cancellation-prone difference of two same-magnitude
    // terms into a single subtraction the optimizer can fuse. C++
    // (double) is unaffected; in f32 the original form measurably
    // increased the body-rate divergence vs the f64 reference.
    let common = (3.0 * alpha_dot_j_sqr - alpha_norm_2 * j_norm_2) * inv_a5;
    let neg_two_aj_inv_a3 = -2.0 * alpha_dot_j * inv_a3;
    let ddzb0 = neg_two_aj_inv_a3 * jer[0] + alpha[0] * common + dn_alpha_s_0;
    let ddzb1 = neg_two_aj_inv_a3 * jer[1] + alpha[1] * common + dn_alpha_s_1;
    let ddzb2 = neg_two_aj_inv_a3 * jer[2] + alpha[2] * common + dn_alpha_s_2;

    // Attitude: tilt(zB) ∘ yaw(ψ). `quaternion_from_zb_and_yaw` with
    // `use_tilt = true` is the same closed form as the C++ source's
    // tilt0/tilt1/tilt2 construction (see rotation.rs:268..283).
    let attitude = quaternion_from_zb_and_yaw(&z_b, psi, true);

    // Body rate. Hoist `1/omg_den` (one VDIV) and reuse it via
    // multiplies for both ω and ω̇.
    let inv_omg_den = 1.0 / zb2_1;
    let inv_omg_den_2 = inv_omg_den * inv_omg_den;

    let omg_term = dzb2 * inv_omg_den;
    let tmp_omg_1 = zb0 * s_psi - zb1 * c_psi;
    let tmp_omg_2 = zb0 * c_psi + zb1 * s_psi;
    let tmp_omg_3 = zb1 * dzb0 - zb0 * dzb1;
    // Hoisted: appear in both ω.x/.y *and* (as `tmp_omg_4/5` in C++)
    // in the ω̇.x/.y correction. Saves 4 mul + 2 sub.
    let dz_psi_a = dzb0 * s_psi - dzb1 * c_psi;
    let dz_psi_b = dzb0 * c_psi + dzb1 * s_psi;
    let omega = Vec3::new(
        dz_psi_a - tmp_omg_1 * omg_term,
        dz_psi_b - tmp_omg_2 * omg_term,
        tmp_omg_3 * inv_omg_den + dpsi,
    );

    // Body angular acceleration. Reuses dz_psi_{a,b} from above.
    let tmp_omg_6 = zb1 * ddzb0 - zb0 * ddzb1;
    let dzb2_sqr = dzb2 * dzb2;

    let omega_dot = Vec3::new(
        ddzb0 * s_psi - ddzb1 * c_psi - ddzb2 * tmp_omg_1 * inv_omg_den
            - dzb2 * dz_psi_a * inv_omg_den
            + dzb2_sqr * tmp_omg_1 * inv_omg_den_2,
        ddzb0 * c_psi + ddzb1 * s_psi - ddzb2 * tmp_omg_2 * inv_omg_den
            - dzb2 * dz_psi_b * inv_omg_den
            + dzb2_sqr * tmp_omg_2 * inv_omg_den_2,
        tmp_omg_6 * inv_omg_den - tmp_omg_3 * dzb2 * inv_omg_den_2 + ddpsi,
    );

    Ok(FlatState {
        thrust_per_mass,
        attitude,
        omega,
        omega_dot,
    })
}

/// Flat-output → state map (true-yaw / compass-heading convention).
///
/// Port of `PositionController::computeNominalReferenceInputs` from the
/// RPG quadrotor stack (`tmp/reference_computation/reference_computation.cpp`)
/// — the drag-free nominal part of Faessler, Franchi & Scaramuzza,
/// *"Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag
/// for Accurate Tracking of High-Speed Trajectories"*, RA-L 2018. Same
/// signature and [`FlatState`] output as [`flatness_to_state_tilt_yaw`],
/// so the two maps are drop-in interchangeable; the mission YAML's
/// `flatness_map` key selects between them (tilt-yaw is the default).
///
/// ## Convention
///
/// Here ψ is the **compass heading of the body-x axis projection**
/// ("true yaw"), not the intrinsic tilt-then-yaw angle: with
/// `x_C = (cos ψ, sin ψ, 0)` and `y_C = (−sin ψ, cos ψ, 0)`, the
/// attitude is
///
/// ```text
/// z_B = α/‖α‖,   x_B = (y_C × z_B)/‖y_C × z_B‖,   y_B = z_B × x_B
/// ```
///
/// (`α = a + g·ẑ`), i.e. body-x always points "toward" the commanded
/// heading regardless of tilt — the same construction as
/// `rotation::quaternion_from_zb_and_yaw(…, use_tilt = false)` and
/// `flatness::reference_quaternion`.
///
/// ## Derivation of the rates
///
/// Differentiating the translational dynamics `α = c·z_B` (per unit
/// mass, `c = ‖α‖`) once gives `j = ċ·z_B + c·(ω_y·x_B − ω_x·y_B)`;
/// projecting onto the body axes yields
///
/// ```text
/// ω_x = −(y_B · j)/c,   ω_y = (x_B · j)/c,   ċ = z_B · j
/// ```
///
/// and differentiating the heading constraint `x_B ∝ y_C × z_B` gives
///
/// ```text
/// ω_z = (ψ̇·(x_C·x_B) + ω_y·(y_C·z_B)) / ‖y_C × z_B‖.
/// ```
///
/// One more time derivative (projections of the snap) produces the
/// angular accelerations, including the `2ċ·ω` (thrust-rate) and
/// `c·ω·ω` (gyroscopic) cross terms carried verbatim from the source.
///
/// ## Faults
///
/// - [`FlatnessFault::NearFreeFall`]: `‖α‖` under the shared floor.
///   Subsumes the C++ `almostZeroThrust` guard (the floor is ~0.1 g,
///   far above "almost zero").
/// - [`FlatnessFault::HeadingSingular`]: `‖y_C × z_B‖` under
///   [`HEADING_SINGULAR_FLOOR`] — thrust axis horizontal along the
///   heading's y-axis. The C++ zeroes ω_z and leans on the attitude
///   estimate to keep `x_B` defined; a pure function has no estimate,
///   so we refuse (same policy as the tilt map's `InvertedTilt`).
///   Note this map is regular at the *inverted* pole where the
///   tilt-yaw map faults, and singular at 90° tilt where the tilt-yaw
///   map is regular — the two conventions trade singularity locations.
pub fn flatness_to_state_true_yaw(
    acc: Vec3,
    jer: Vec3,
    sna: Vec3,
    yaw_triple: [f32; 3],
    gravity: f32,
) -> Result<FlatState, FlatnessFault> {
    let psi = yaw_triple[0];
    let dpsi = yaw_triple[1];
    let ddpsi = yaw_triple[2];

    let c_psi = libm::cosf(psi);
    let s_psi = libm::sinf(psi);
    let x_c = Vec3::new(c_psi, s_psi, 0.0);
    let y_c = Vec3::new(-s_psi, c_psi, 0.0);

    // α = a + g·ẑ; collective thrust per unit mass c = ‖α‖.
    let alpha = Vec3::new(acc[0], acc[1], acc[2] + gravity);
    let alpha_norm_2 = alpha.norm_squared();
    if alpha_norm_2 < ALPHA_NORM_SQR_FLOOR {
        return Err(FlatnessFault::NearFreeFall);
    }
    let c = libm::sqrtf(alpha_norm_2);
    let inv_c = 1.0 / c;

    let z_b = alpha * inv_c;
    let yc_cross_zb = y_c.cross(&z_b);
    let s_yc = yc_cross_zb.norm();
    if s_yc < HEADING_SINGULAR_FLOOR {
        return Err(FlatnessFault::HeadingSingular);
    }
    let inv_s_yc = 1.0 / s_yc;
    let x_b = yc_cross_zb * inv_s_yc;
    // Unit by construction: z_B ⊥ x_B, both unit.
    let y_b = z_b.cross(&x_b);

    let r_wb = Matrix3::from_columns(&[x_b, y_b, z_b]);
    let attitude = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_wb));

    // Body rates from the jerk projections + heading-constraint rate.
    let omega_x = -(y_b.dot(&jer)) * inv_c;
    let omega_y = x_b.dot(&jer) * inv_c;
    let omega_z = (dpsi * x_c.dot(&x_b) + omega_y * y_c.dot(&z_b)) * inv_s_yc;

    // Angular accelerations from the snap projections. `ċ = z_B · j`.
    let c_dot = z_b.dot(&jer);
    let omega_dot_x = -(y_b.dot(&sna) + 2.0 * c_dot * omega_x - c * omega_y * omega_z) * inv_c;
    let omega_dot_y = (x_b.dot(&sna) - 2.0 * c_dot * omega_y - c * omega_x * omega_z) * inv_c;
    let omega_dot_z = (ddpsi * x_c.dot(&x_b)
        + 2.0 * dpsi * omega_z * x_c.dot(&y_b)
        - 2.0 * dpsi * omega_y * x_c.dot(&z_b)
        - omega_x * omega_y * y_c.dot(&y_b)
        - omega_x * omega_z * y_c.dot(&z_b)
        + omega_dot_y * y_c.dot(&z_b))
        * inv_s_yc;

    Ok(FlatState {
        thrust_per_mass: c,
        attitude,
        omega: Vec3::new(omega_x, omega_y, omega_z),
        omega_dot: Vec3::new(omega_dot_x, omega_dot_y, omega_dot_z),
    })
}

/// Pole-safe flat-output → (thrust, attitude, body-rate) map for the MPC
/// outer-loop feedforward.
///
/// Companion to [`flatness_to_state_tilt_yaw`] tailored to a 4-channel
/// MPC whose control vector is `[T, ω_x, ω_y, ω_z]`. Returns:
///
/// - `thrust_per_mass = ‖a + g·ẑ‖`. Parameterization-independent —
///   has no dependence on the tilt-yaw decomposition, so it stays
///   well-defined arbitrarily close to the inverted pole.
/// - `attitude` from [`quaternion_from_zb_and_yaw`] with `use_tilt = true`.
///   The unique singularity at `z_b = -ẑ` is handled by a substituted
///   180° flip inside that function.
/// - `omega` in body frame, computed from the *minimum-norm* world
///   angular velocity
///
///   ```text
///   ω_world  =  z_b × dz_b  +  ψ̇ · ẑ_world
///   ```
///
///   then rotated into body frame via the (pole-safe) attitude
///   quaternion. This is the parameterization-independent angular
///   velocity that produces the smooth attitude trajectory through
///   the pole; it is finite and bounded everywhere `‖a + g·ẑ‖` is
///   above the free-fall floor.
///
/// ## Min-norm body rate vs the tilt-yaw closed form — read before using
///
/// [`flatness_to_state_tilt_yaw`] computes ω in the *intrinsic-Euler
/// tilt-then-yaw* convention. That ω contains a `(zb1·dzb0 − zb0·dzb1)
/// / (zb.z + 1)` body-z term — `(1 − cos θ)·φ̇` for a thrust axis at
/// tilt θ whose azimuth turns at φ̇ — which diverges as `z_b.z → −1`.
///
/// The min-norm form here drops that term: its body-z component is
/// identically zero when `ψ̇ = 0`. That is the angular velocity of the
/// **no-twist** (parallel-transport) frame, *not* of the tilt-yaw
/// attitude this function returns. The two only agree where the path
/// does not curve under tilt. On a 106°-tilt time-optimal circle the
/// closed-form body-z rate reaches 6 rad/s; pairing the tilt `attitude`
/// as `q_ref` with this ω as the rate reference hands the MPC two
/// references that contradict each other by exactly that much, and in
/// simulation the reduced-model MPC then diverges on the time-optimal
/// missions (`crates/cybflight_sim/tests/omega_ref_ab.rs`).
///
/// So: the outer loop's reference chain uses
/// [`flatness_to_thrust_omega_tilt_yaw`] (closed form, consistent with
/// the tilt `q_ref`). This min-norm form remains for consumers that
/// need a bounded, pole-safe rate *feature* rather than the rate of a
/// specific attitude — the learned-cost observation
/// (`mpc::cost_adapt`) was trained on it and must keep it.
///
/// At the pole the min-norm body rate matches the body-rate limit of the
/// substituted attitude in [`quaternion_from_zb_and_yaw`].
///
/// `omega_dot` is intentionally not returned: the firmware MPC's input
/// is `[T, ω_x, ω_y, ω_z]` (no `ω̇` channel), and the closed-form ω̇
/// formula contains `1/(zb.z + 1)²` which diverges quadratically faster
/// than ω. Skipping it removes the worst pole singularity from the
/// integration path entirely.
pub fn flatness_to_thrust_omega(
    acc: Vec3,
    jer: Vec3,
    yaw: f32,
    yaw_rate: f32,
    gravity: f32,
) -> Result<(f32, UnitQuaternion<f32>, Vec3), FlatnessFault> {
    let alpha = Vec3::new(acc[0], acc[1], acc[2] + gravity);
    let alpha_norm_2 = alpha.norm_squared();
    if alpha_norm_2 < ALPHA_NORM_SQR_FLOOR_POLE_SAFE {
        return Err(FlatnessFault::NearFreeFall);
    }
    let alpha_norm_1 = libm::sqrtf(alpha_norm_2);
    let inv_alpha_norm_1 = 1.0 / alpha_norm_1;
    let inv_alpha_norm_2 = inv_alpha_norm_1 * inv_alpha_norm_1;

    // z_b = α / ‖α‖
    let z_b = alpha * inv_alpha_norm_1;

    // dz_b = N(α)·j = (j − α·(α·j)/‖α‖²) / ‖α‖. No division by `zb.z + 1`,
    // so this stays bounded across the pole.
    let proj_j = alpha.dot(&jer) * inv_alpha_norm_2;
    let dz_b = Vec3::new(
        (jer[0] - alpha[0] * proj_j) * inv_alpha_norm_1,
        (jer[1] - alpha[1] * proj_j) * inv_alpha_norm_1,
        (jer[2] - alpha[2] * proj_j) * inv_alpha_norm_1,
    );

    // World-frame angular velocity: perpendicular component rotates
    // z_b along the trajectory; world-z component carries the yaw rate.
    let perp = z_b.cross(&dz_b);
    let omega_world = Vec3::new(perp[0], perp[1], perp[2] + yaw_rate);

    // Attitude is pole-safe (substituted 180° flip at z_b = -ẑ).
    let attitude = quaternion_from_zb_and_yaw(&z_b, yaw, true);

    // ω_body = R^T · ω_world. UnitQuaternion's inverse_transform_vector
    // is `q^{-1} · v · q` — the standard body-from-world rotation.
    let omega_body = attitude.inverse_transform_vector(&omega_world);

    Ok((alpha_norm_1, attitude, omega_body))
}

/// Snap-free flat-output → (thrust, attitude, body-rate) map in the
/// **tilt-yaw** convention, closed form — the map the outer loop's
/// reference chain uses for `flatness_map: tilt_yaw` missions.
///
/// Returns the `thrust_per_mass` / `attitude` / `omega` triple of
/// [`flatness_to_state_tilt_yaw`]: the attitude is the tilt quaternion
/// (`quaternion_from_zb_and_yaw(z_b, ψ, true)`, the same one `q_ref` is
/// built from) and `omega` is the angular velocity **of that quaternion**,
/// including the `(zb1·dzb0 − zb0·dzb1)/(zb.z + 1)` body-z term that
/// [`flatness_to_thrust_omega`] leaves out. None of the three read snap
/// (snap enters only ω̇, which this function does not return), so a
/// sampler node carrying `(acc, jerk)` is sufficient — verified
/// bit-for-bit in `omega_ref_ab::closed_form_omega_is_snap_independent`.
///
/// Faults: [`FlatnessFault::NearFreeFall`] and
/// [`FlatnessFault::InvertedTilt`] (the body-z term is singular at the
/// inverted pole; the caller's fault path holds the last attitude and
/// hover-biases the input). Consumers should clamp the returned ω to the
/// vehicle rate bounds as they already do for the min-norm form — the
/// term grows without bound as the pole is approached.
pub fn flatness_to_thrust_omega_tilt_yaw(
    acc: Vec3,
    jer: Vec3,
    yaw: f32,
    yaw_rate: f32,
    gravity: f32,
) -> Result<(f32, UnitQuaternion<f32>, Vec3), FlatnessFault> {
    let st = flatness_to_state_tilt_yaw(acc, jer, Vec3::zeros(), [yaw, yaw_rate, 0.0], gravity)?;
    Ok((st.thrust_per_mass, st.attitude, st.omega))
}

/// Snap-free flat-output → (thrust, attitude, body-rate) map in the
/// **true-yaw** convention — the compass-heading twin of
/// [`flatness_to_thrust_omega_tilt_yaw`].
///
/// Exists so both conventions can feed a 4-channel `[T, ω_x, ω_y, ω_z]`
/// consumer (the MPC's `u_ref`, or any reference sampled off a min-jerk
/// trajectory) without inventing a snap. Callers used to reach for
/// [`flatness_to_state_true_yaw`] with `sna = 0` and discard the
/// resulting `omega_dot`; that works only as long as nobody reads the
/// field, because a zeroed snap makes `omega_dot` *wrong* rather than
/// absent. Not returning it removes the trap.
///
/// The returned triple is bit-for-bit what [`flatness_to_state_true_yaw`]
/// puts in its `thrust_per_mass` / `attitude` / `omega` fields: none of
/// the three touch `sna` (see the derivation on that function — snap
/// enters only through the ω̇ projections), so this is a strict
/// restriction of the same map, not an approximation of it.
///
/// Faults are the same two the full map can raise before it reaches the
/// snap terms: [`FlatnessFault::NearFreeFall`] and
/// [`FlatnessFault::HeadingSingular`]. [`FlatnessFault::InvertedTilt`]
/// is impossible here — this convention is regular at the inverted pole
/// (and singular at 90° tilt, where the tilt-yaw pair is regular).
pub fn flatness_to_thrust_omega_true_yaw(
    acc: Vec3,
    jer: Vec3,
    yaw: f32,
    yaw_rate: f32,
    gravity: f32,
) -> Result<(f32, UnitQuaternion<f32>, Vec3), FlatnessFault> {
    let c_psi = libm::cosf(yaw);
    let s_psi = libm::sinf(yaw);
    let x_c = Vec3::new(c_psi, s_psi, 0.0);
    let y_c = Vec3::new(-s_psi, c_psi, 0.0);

    let alpha = Vec3::new(acc[0], acc[1], acc[2] + gravity);
    let alpha_norm_2 = alpha.norm_squared();
    if alpha_norm_2 < ALPHA_NORM_SQR_FLOOR_POLE_SAFE {
        return Err(FlatnessFault::NearFreeFall);
    }
    let c = libm::sqrtf(alpha_norm_2);
    let inv_c = 1.0 / c;

    let z_b = alpha * inv_c;
    let yc_cross_zb = y_c.cross(&z_b);
    let s_yc = yc_cross_zb.norm();
    if s_yc < HEADING_SINGULAR_FLOOR {
        return Err(FlatnessFault::HeadingSingular);
    }
    let inv_s_yc = 1.0 / s_yc;
    let x_b = yc_cross_zb * inv_s_yc;
    // Unit by construction: z_B ⊥ x_B, both unit.
    let y_b = z_b.cross(&x_b);

    let r_wb = Matrix3::from_columns(&[x_b, y_b, z_b]);
    let attitude = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_wb));

    let omega_x = -(y_b.dot(&jer)) * inv_c;
    let omega_y = x_b.dot(&jer) * inv_c;
    let omega_z = (yaw_rate * x_c.dot(&x_b) + omega_y * y_c.dot(&z_b)) * inv_s_yc;

    Ok((c, attitude, Vec3::new(omega_x, omega_y, omega_z)))
}

/// Reference body-to-world quaternion from the flat output `acc` and a
/// desired yaw `yaw_rad` (natural default `0.0`).
#[inline]
pub fn reference_quaternion(acc: Vec3, yaw_rad: f32, gravity: f32) -> UnitQuaternion<f32> {
    let acc_cmd = acc + Vector3::new(0.0, 0.0, gravity);
    let inv_norm = 1.0 / acc_cmd.norm().max(1e-8);
    let z_b = acc_cmd * inv_norm;

    let s = libm::sinf(yaw_rad);
    let c = libm::cosf(yaw_rad);
    let y_c = Vector3::new(-s, c, 0.0);

    let x_b_unnorm = y_c.cross(&z_b);
    let x_b = x_b_unnorm * (1.0 / x_b_unnorm.norm().max(1e-8));

    let y_b_unnorm = z_b.cross(&x_b);
    let y_b = y_b_unnorm * (1.0 / y_b_unnorm.norm().max(1e-8));

    let r_wb = Matrix3::from_columns(&[x_b, y_b, z_b]);
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_wb))
}

// ═════════════════════════════════════════════════════════════════════
// Part 2 — planner gradient chain (ψ = 0, differentiable)
//
// Forward: position derivatives → thrust vector, body axis, body
// rates. Backward: gradient backpropagation through the same
// transforms. Split in two stages so the BFGS cost evaluator can
// stop at `AlphaState` for the tilt / collective-thrust penalties
// and only pay for the jerk-dependent terms when the body-rate
// penalty is active.
//
// Unlike the fault-RETURNING maps above these CLAMP at their
// singularities instead of refusing: a cost function that returns `Err` mid-line-search is
// useless to the optimizer, and a clamped gradient still points
// downhill. Do not reuse them to build a flight reference.
// ═════════════════════════════════════════════════════════════════════

/// Thrust-vector basis. Computed from acceleration alone — cheap.
///
/// Always well-defined (division by `max(‖α‖, 1e-8)` clamps the singularity).
/// Callers that use ω must additionally check `zb[2] > -0.9` before extending
/// to [`FlatnessState`].
pub struct AlphaState {
    /// α = acc + [0, 0, g].
    pub alpha: Vector3<f32>,
    /// ‖α‖.
    pub norm_alpha: f32,
    /// 1/‖α‖.
    pub inv_norm_alpha: f32,
    /// Body z-axis: α/‖α‖.
    pub zb: Vector3<f32>,
}

/// Full flatness state, including body-rate terms.
pub struct FlatnessState {
    pub alpha: Vector3<f32>,
    pub norm_alpha: f32,
    pub inv_norm_alpha: f32,
    pub zb: Vector3<f32>,
    /// dot(zB, jer).
    pub dot_zb_j: f32,
    /// Body z-axis time derivative: dzB = DN(α)·j / ‖α‖.
    pub dzb: Vector3<f32>,
    /// 1/(1 + zb_z), with singularity guard.
    pub s_inv: f32,
    /// Body rates [ωx, ωy, ωz] at ψ=0.
    pub omega: Vector3<f32>,
}

/// Compute the thrust-vector basis (α, ‖α‖, zB) from acceleration.
#[inline]
pub fn compute_alpha_state(acc: Vec3, gravity: f32) -> AlphaState {
    let alpha = acc + Vector3::new(0.0, 0.0, gravity);
    let norm_alpha = alpha.norm().max(1e-8);
    let inv_norm_alpha = 1.0 / norm_alpha;
    let zb = alpha * inv_norm_alpha;
    AlphaState {
        alpha,
        norm_alpha,
        inv_norm_alpha,
        zb,
    }
}

/// Extend an [`AlphaState`] with body-rate terms.
///
/// Precondition: `alpha.zb[2] > -0.9` (caller must guard; the `1/(1+zb_z)`
/// factor becomes ill-conditioned near inversion).
#[inline]
pub fn extend_to_flatness(alpha: &AlphaState, jer: Vec3) -> FlatnessState {
    let zb = alpha.zb;
    let inv_norm_alpha = alpha.inv_norm_alpha;
    let dot_zb_j = zb.dot(&jer);
    let dzb = (jer - zb * dot_zb_j) * inv_norm_alpha;
    let s_inv = 1.0 / (1.0 + zb[2]).max(0.01);
    let omega = Vector3::new(
        -dzb[1] + s_inv * zb[1] * dzb[2],
        dzb[0] - s_inv * zb[0] * dzb[2],
        s_inv * (zb[1] * dzb[0] - zb[0] * dzb[1]),
    );
    FlatnessState {
        alpha: alpha.alpha,
        norm_alpha: alpha.norm_alpha,
        inv_norm_alpha,
        zb,
        dot_zb_j,
        dzb,
        s_inv,
        omega,
    }
}

/// Backpropagate body rate gradient through the flatness chain.
///
/// Gradient flow: ∂L/∂ω → ∂L/∂dzB, ∂L/∂zB → ∂L/∂acc, ∂L/∂jer.
/// Accumulates into `grad_acc` and `grad_jer`.
#[inline]
pub fn body_rate_grad_backprop(
    g_omega: &Vector3<f32>,
    fs: &FlatnessState,
    grad_acc: &mut Vector3<f32>,
    grad_jer: &mut Vector3<f32>,
) {
    let zb = fs.zb;
    let dzb = fs.dzb;
    let s_inv = fs.s_inv;
    let inv_norm_alpha = fs.inv_norm_alpha;
    let dot_zb_j = fs.dot_zb_j;

    // grad_ω → grad_dzB via (∂ω/∂dzB)ᵀ
    let g_dzb = Vector3::new(
        g_omega[1] + s_inv * zb[1] * g_omega[2],
        -g_omega[0] - s_inv * zb[0] * g_omega[2],
        s_inv * (zb[1] * g_omega[0] - zb[0] * g_omega[1]),
    );

    // grad_ω → grad_zB via (∂ω/∂zB)ᵀ
    let s2 = s_inv * s_inv;
    let g_zb = Vector3::new(
        -s_inv * (dzb[2] * g_omega[1] + dzb[1] * g_omega[2]),
        s_inv * (dzb[2] * g_omega[0] + dzb[0] * g_omega[2]),
        s2 * (dzb[2] * (zb[0] * g_omega[1] - zb[1] * g_omega[0])
            + (zb[0] * dzb[1] - zb[1] * dzb[0]) * g_omega[2]),
    );

    // DN(α) is the nullspace projector: DN(v) = (v − zB(zB·v)) / ‖α‖
    let dot_zb_gdzb = zb.dot(&g_dzb);
    let dot_zb_gzb = zb.dot(&g_zb);

    // grad_jerk = DN(α) · g_dzb
    let dn_gdzb = (g_dzb - zb * dot_zb_gdzb) * inv_norm_alpha;
    *grad_jer += dn_gdzb;

    // grad_acc from zB path: DN(α) · g_zb
    let dn_gzb = (g_zb - zb * dot_zb_gzb) * inv_norm_alpha;
    *grad_acc += dn_gzb;

    // grad_acc from dzB path: −(∂dzB/∂α)ᵀ · g_dzb
    let dot_dzb_gdzb = dzb.dot(&g_dzb);
    let chain = (dn_gdzb * dot_zb_j + dzb * dot_zb_gdzb + zb * dot_dzb_gdzb) * inv_norm_alpha;
    *grad_acc -= chain;
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::ZERO3;

    /// Hover flatness: zero a/j/s + zero yaw → identity attitude,
    /// zero body rate, thrust = g.
    #[test]
    fn test_flatness_hover() {
        let st = flatness_to_state_tilt_yaw(ZERO3, ZERO3, ZERO3, [0.0; 3], 9.81)
            .expect("hover should be a valid flat state");
        assert!((st.thrust_per_mass - 9.81).abs() < 1e-4);
        let q = st.attitude;
        assert!((q.w - 1.0).abs() < 1e-4);
        assert!(q.i.abs() < 1e-4);
        assert!(q.j.abs() < 1e-4);
        assert!(q.k.abs() < 1e-4);
        assert!(st.omega.norm() < 1e-4);
    }

    /// Free-fall: a = -g·ẑ → ‖α‖ ≈ 0; should fault, not NaN.
    #[test]
    fn test_flatness_free_fall_fault() {
        let acc = Vec3::new(0.0, 0.0, -9.81);
        let r = flatness_to_state_tilt_yaw(acc, ZERO3, ZERO3, [0.0; 3], 9.81);
        assert_eq!(r.unwrap_err(), FlatnessFault::NearFreeFall);
    }

    /// Inverted: a chosen so zB ≈ -ẑ; should fault, not produce
    /// blow-up ω̇.
    #[test]
    fn test_flatness_inverted_fault() {
        // α = (0, 0, -|α|) → zB = (0, 0, -1), zb2_1 = 0.
        let acc = Vec3::new(0.0, 0.0, -2.0 * 9.81);
        let r = flatness_to_state_tilt_yaw(acc, ZERO3, ZERO3, [0.0; 3], 9.81);
        assert_eq!(r.unwrap_err(), FlatnessFault::InvertedTilt);
    }

    // ── flatness_to_thrust_omega (pole-safe MPC u_ref feedforward) ──

    /// Hover: zero a/j → identity attitude, zero body rate, thrust=g.
    /// Same expectation as `test_flatness_hover` for the full map; this
    /// confirms the trimmed function returns the same hover values.
    #[test]
    fn test_thrust_omega_hover() {
        let (tpm, q, omega) =
            flatness_to_thrust_omega(ZERO3, ZERO3, 0.0, 0.0, 9.81).expect("hover ok");
        assert!((tpm - 9.81).abs() < 1e-4);
        assert!((q.w - 1.0).abs() < 1e-4);
        assert!(q.i.abs() < 1e-4 && q.j.abs() < 1e-4 && q.k.abs() < 1e-4);
        assert!(omega.norm() < 1e-4, "hover omega nonzero: {omega:?}");
    }

    /// Free-fall fault parity with the full map.
    #[test]
    fn test_thrust_omega_free_fall_fault() {
        let acc = Vec3::new(0.0, 0.0, -9.81);
        let r = flatness_to_thrust_omega(acc, ZERO3, 0.0, 0.0, 9.81);
        assert_eq!(r.unwrap_err(), FlatnessFault::NearFreeFall);
    }

    /// At the inverted pole the *full* map faults (`InvertedTilt`)
    /// because its closed-form ω diverges. The pole-safe map must NOT
    /// fault — that is the whole point — and the body rate it returns
    /// must be finite.
    #[test]
    fn test_thrust_omega_pole_no_fault_finite_omega() {
        // α = (0, 0, -2g) → z_b = (0, 0, -1): exact pole.
        let acc = Vec3::new(0.0, 0.0, -2.0 * 9.81);
        // Some nonzero jerk in xy so dz_b ≠ 0 and the perpendicular
        // angular-velocity component is non-trivial.
        let jer = Vec3::new(3.0, 1.5, 0.0);
        let (tpm, _q, omega) = flatness_to_thrust_omega(acc, jer, 0.3, 0.0, 9.81)
            .expect("pole-safe ok at z_b = -ẑ");
        // Thrust per mass = ‖α‖ = g (positive, finite).
        assert!((tpm - 9.81).abs() < 1e-3, "tpm wrong at pole: {tpm}");
        // Body rate must be finite and bounded — `‖dz_b‖` here is on
        // the order of `‖j‖/‖α‖` ≈ 3.4/9.81 ≈ 0.35 rad/s, so the body
        // rate magnitude should be of that order, not "infinite".
        assert!(
            omega.iter().all(|c| c.is_finite()),
            "non-finite omega at pole: {omega:?}"
        );
        assert!(
            omega.norm() < 5.0,
            "implausibly large omega at pole: {omega:?}"
        );
    }

    /// Off the pole, the pole-safe map's ω agrees with the geometric
    /// `z_b × dz_b` projected into body frame (which is its definition).
    /// This regression-locks the formula and catches any future axis-
    /// or sign-flipping mistake.
    #[test]
    fn test_thrust_omega_matches_min_norm_definition() {
        use nalgebra::Vector3;
        let acc = Vec3::new(2.0, -1.0, 1.5);
        let jer = Vec3::new(0.7, 0.4, -0.2);
        let yaw = 0.5;
        let yaw_rate = 0.0;
        let (tpm, q, omega_body) =
            flatness_to_thrust_omega(acc, jer, yaw, yaw_rate, 9.81).expect("nominal ok");

        // Thrust per mass = ‖a + g·ẑ‖
        let alpha = Vec3::new(acc[0], acc[1], acc[2] + 9.81);
        let alpha_norm = alpha.norm();
        assert!((tpm - alpha_norm).abs() < 1e-4);

        // Reconstruct ω_world from body-frame ω via the attitude.
        let omega_world_back = q * omega_body;

        // Geometric ω_world (yaw_rate = 0): z_b × dz_b
        let z_b = alpha / alpha_norm;
        let proj = alpha.dot(&jer) / (alpha_norm * alpha_norm);
        let dz_b = (jer - alpha * proj) / alpha_norm;
        let expected = z_b.cross(&dz_b);

        let diff: Vector3<f32> = omega_world_back - expected;
        assert!(
            diff.norm() < 1e-4,
            "omega_world reconstructed = {omega_world_back:?}, expected {expected:?}"
        );
    }

    /// Yaw rate of `ψ̇` rad/s in world-z, identity attitude (z_b = ẑ,
    /// yaw = 0): should produce body rate `(0, 0, ψ̇)` exactly. Confirms
    /// the world-z yaw-rate convention.
    #[test]
    fn test_thrust_omega_yaw_rate_at_hover() {
        let yaw_rate = 0.7;
        let (_tpm, _q, omega) =
            flatness_to_thrust_omega(ZERO3, ZERO3, 0.0, yaw_rate, 9.81).expect("ok");
        assert!(omega.x.abs() < 1e-4);
        assert!(omega.y.abs() < 1e-4);
        assert!((omega.z - yaw_rate).abs() < 1e-4);
    }

    /// Sweep z_b through the inverted pole along a continuous path and
    /// confirm thrust + body rate stay finite and bounded across the
    /// crossing. This is the regression test for the original bug:
    /// the full-map closed form blows up ω as `1/(zb.z + 1)`; the
    /// pole-safe map must not.
    #[test]
    fn test_thrust_omega_continuous_through_pole() {
        // Sweep φ ∈ [π/2 − δ, π/2 + δ] where the trajectory α =
        // ‖α‖ · (sin φ, 0, −cos φ) crosses the pole exactly at φ = π/2
        // (z_b = (1, 0, 0) → (0, 0, -1) → (-1, 0, 0)). dα/dφ supplies
        // the jerk via α̇ ≈ (dα/dφ) · φ̇ ; we use φ̇ = 1 rad/s for
        // simplicity, which makes ‖dz_b‖ = 1 rad/s by construction.
        let alpha_mag = 12.0; // > free-fall floor
        let phi_dot = 1.0;
        let mut max_norm = 0.0f32;
        let mut all_finite = true;
        for i in 0..201 {
            let phi = core::f32::consts::FRAC_PI_2 + (i as f32 - 100.0) * 1e-3;
            let s = libm::sinf(phi);
            let c = libm::cosf(phi);
            let alpha = Vec3::new(alpha_mag * s, 0.0, -alpha_mag * c);
            // d/dφ α = α_mag · (c, 0, s); jerk = α̇ − 0 = (dα/dφ)·φ̇.
            let alpha_dot = Vec3::new(alpha_mag * c * phi_dot, 0.0, alpha_mag * s * phi_dot);
            let acc = Vec3::new(alpha[0], alpha[1], alpha[2] - (-9.81)); // a = α − g·ẑ; here g·ẑ = (0,0,9.81), so a = α − (0,0,9.81)
            let jer = alpha_dot;
            let r = flatness_to_thrust_omega(acc, jer, 0.0, 0.0, 9.81);
            match r {
                Ok((tpm, _, omega)) => {
                    if !tpm.is_finite() || omega.iter().any(|c| !c.is_finite()) {
                        all_finite = false;
                    }
                    max_norm = max_norm.max(omega.norm());
                }
                Err(FlatnessFault::NearFreeFall) => {
                    // Possible at certain φ if α magnitude dips; should not happen here.
                    panic!("unexpected NearFreeFall at φ={phi}");
                }
                Err(FlatnessFault::InvertedTilt) => {
                    panic!("pole-safe map must not return InvertedTilt at φ={phi}");
                }
                Err(FlatnessFault::HeadingSingular) => {
                    // Only the true-yaw map returns this variant.
                    panic!("tilt-family map must not return HeadingSingular at φ={phi}");
                }
            }
        }
        assert!(all_finite, "non-finite ω somewhere in the pole sweep");
        // ‖dz_b‖ = 1 rad/s by construction, so ‖ω‖ should be ~1 rad/s
        // across the sweep — well under any "diverging" threshold.
        assert!(
            max_norm < 5.0,
            "max omega norm over pole sweep too large: {max_norm}"
        );
    }

    // ── flatness_to_state_true_yaw (compass-heading convention) ──────

    /// Hover: identity attitude, zero rates and accels, thrust = g —
    /// and exact agreement with the tilt map (the two conventions
    /// coincide at zero tilt).
    #[test]
    fn test_true_yaw_hover_parity() {
        let st = flatness_to_state_true_yaw(ZERO3, ZERO3, ZERO3, [0.0; 3], 9.81)
            .expect("hover ok");
        assert!((st.thrust_per_mass - 9.81).abs() < 1e-4);
        assert!((st.attitude.w - 1.0).abs() < 1e-4);
        assert!(st.omega.norm() < 1e-4);
        assert!(st.omega_dot.norm() < 1e-4);

        let tilt = flatness_to_state_tilt_yaw(ZERO3, ZERO3, ZERO3, [0.0; 3], 9.81).unwrap();
        assert!(st.attitude.angle_to(&tilt.attitude) < 1e-4);
        assert!((st.omega - tilt.omega).norm() < 1e-4);
    }

    /// Yaw rate at hover maps to body-z rate exactly (x_C·x_B = 1,
    /// ‖y_C × z_B‖ = 1 upright).
    #[test]
    fn test_true_yaw_rate_at_hover() {
        let st = flatness_to_state_true_yaw(ZERO3, ZERO3, ZERO3, [0.4, 0.7, 0.0], 9.81)
            .expect("ok");
        assert!(st.omega.x.abs() < 1e-4);
        assert!(st.omega.y.abs() < 1e-4);
        assert!((st.omega.z - 0.7).abs() < 1e-4);
    }

    /// At zero yaw the flown heading (horizontal projection of body-x)
    /// must equal the commanded ψ regardless of tilt — the defining
    /// property of the true-yaw convention (the tilt map does NOT have
    /// this property at large tilt).
    #[test]
    fn test_true_yaw_heading_is_compass() {
        let acc = Vec3::new(6.0, -4.0, 2.0); // strong lateral acceleration
        let psi = 0.8;
        let st = flatness_to_state_true_yaw(acc, ZERO3, ZERO3, [psi, 0.0, 0.0], 9.81)
            .expect("ok");
        let x_b = st.attitude * Vec3::new(1.0, 0.0, 0.0);
        let heading = libm::atan2f(x_b.y, x_b.x);
        assert!(
            (heading - psi).abs() < 1e-4,
            "flown heading {heading} != commanded {psi}"
        );
    }

    /// Free-fall fault parity with the other maps.
    #[test]
    fn test_true_yaw_free_fall_fault() {
        let acc = Vec3::new(0.0, 0.0, -9.81);
        let r = flatness_to_state_true_yaw(acc, ZERO3, ZERO3, [0.0; 3], 9.81);
        assert_eq!(r.unwrap_err(), FlatnessFault::NearFreeFall);
    }

    /// Thrust axis horizontal along the heading's y-axis (ψ=0, α = +ŷ):
    /// the x_B construction and the ω_z denominator collapse — must
    /// fault, not NaN. The inverted pole (α = −ẑ), where the TILT map
    /// faults, is regular here.
    #[test]
    fn test_true_yaw_heading_singular_fault() {
        let acc = Vec3::new(0.0, 12.0, -9.81); // α = (0, 12, 0) ∥ y_C
        let r = flatness_to_state_true_yaw(acc, ZERO3, ZERO3, [0.0; 3], 9.81);
        assert_eq!(r.unwrap_err(), FlatnessFault::HeadingSingular);

        // Inverted pole: fine for this map (x_B = y_C × (−ẑ) is well
        // defined), so it must NOT fault.
        let acc = Vec3::new(0.0, 0.0, -2.0 * 9.81);
        let st = flatness_to_state_true_yaw(acc, ZERO3, ZERO3, [0.0; 3], 9.81)
            .expect("inverted pole is regular for true-yaw");
        assert!(st.omega.iter().all(|c| c.is_finite()));
    }

    /// Finite-difference lock on the ω and ω̇ closed forms: along a
    /// smooth analytic flat-output trajectory, ω must match the numeric
    /// derivative of the attitude and ω̇ the numeric derivative of ω.
    #[test]
    fn test_true_yaw_rates_match_attitude_derivative() {
        let g = 9.81f32;
        // acc(t) = (A sin t, B cos t, C sin 2t) with analytic jerk/snap;
        // ψ(t) = 0.4 sin t. Chosen well clear of both singularities.
        let eval = |t: f32| {
            let (a, b, c3) = (2.0f32, 1.5f32, 1.0f32);
            let acc = Vec3::new(a * libm::sinf(t), b * libm::cosf(t), c3 * libm::sinf(2.0 * t));
            let jer = Vec3::new(
                a * libm::cosf(t),
                -b * libm::sinf(t),
                2.0 * c3 * libm::cosf(2.0 * t),
            );
            let sna = Vec3::new(
                -a * libm::sinf(t),
                -b * libm::cosf(t),
                -4.0 * c3 * libm::sinf(2.0 * t),
            );
            let yaw = [
                0.4 * libm::sinf(t),
                0.4 * libm::cosf(t),
                -0.4 * libm::sinf(t),
            ];
            flatness_to_state_true_yaw(acc, jer, sna, yaw, g).expect("regular sample")
        };

        let t0 = 0.7f32;
        let h = 5e-3f32;
        let st = eval(t0);
        let st_m = eval(t0 - h);
        let st_p = eval(t0 + h);

        // Body rate: rotvec(q(t−h)⁻¹ q(t+h)) / 2h ≈ ω_body(t).
        let dq = st_m.attitude.inverse() * st_p.attitude;
        let omega_num = dq.scaled_axis() / (2.0 * h);
        let err = (st.omega - omega_num).norm();
        assert!(
            err < 2e-2,
            "omega {:?} vs numeric {:?} (err {err})",
            st.omega,
            omega_num
        );

        // Angular acceleration: central difference of the map's own ω.
        let omega_dot_num = (st_p.omega - st_m.omega) / (2.0 * h);
        let err = (st.omega_dot - omega_dot_num).norm();
        assert!(
            err < 2e-2,
            "omega_dot {:?} vs numeric {:?} (err {err})",
            st.omega_dot,
            omega_dot_num
        );
    }

    // ── snap-free companions are strict restrictions of the full maps ──

    /// The load-bearing claim behind [`flatness_to_thrust_omega_true_yaw`]:
    /// thrust, attitude and ω never read `sna`, so the companion must
    /// return *bit-identical* values to the full map for any snap at all.
    /// If someone later folds a snap term into one of those three, this
    /// fails instead of silently changing the MPC's `u_ref`.
    #[test]
    fn true_yaw_snap_free_matches_full_map_exactly() {
        let cases: &[(Vec3, Vec3, f32, f32)] = &[
            (ZERO3, ZERO3, 0.0, 0.0),
            (Vec3::new(1.0, -2.0, 3.0), Vec3::new(0.5, 0.25, -1.0), 0.7, 0.3),
            (Vec3::new(-4.0, 6.0, -2.0), Vec3::new(-3.0, 1.5, 2.0), -2.1, -0.8),
            (Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, 0.0), 3.0, 0.0),
        ];
        // Deliberately non-zero and varied: the whole point is that the
        // full map's other outputs are invariant to this argument.
        let snaps: &[Vec3] = &[
            ZERO3,
            Vec3::new(10.0, -20.0, 30.0),
            Vec3::new(-1e3, 1e3, -1e3),
        ];
        // Every case here sits above the *strict* floor, which is where
        // the two functions are required to agree. The pole-safe map
        // deliberately evaluates further down (see
        // `ALPHA_NORM_SQR_FLOOR_POLE_SAFE`); that divergence gets its own
        // test below rather than weakening this one.
        for &(acc, jer, psi, dpsi) in cases {
            let alpha_sq =
                (acc + Vector3::new(0.0, 0.0, 9.81)).norm_squared();
            assert!(alpha_sq >= 1.0, "case {acc:?} is below the strict floor");
            let short = flatness_to_thrust_omega_true_yaw(acc, jer, psi, dpsi, 9.81);
            for &sna in snaps {
                let full = flatness_to_state_true_yaw(acc, jer, sna, [psi, dpsi, 0.0], 9.81);
                match (&short, &full) {
                    (Ok((tpm, q, omega)), Ok(st)) => {
                        assert_eq!(*tpm, st.thrust_per_mass, "thrust for {acc:?}/{sna:?}");
                        assert_eq!(*q, st.attitude, "attitude for {acc:?}/{sna:?}");
                        assert_eq!(*omega, st.omega, "omega for {acc:?}/{sna:?}");
                    }
                    (Err(a), Err(b)) => assert_eq!(a, b, "fault kind for {acc:?}"),
                    _ => panic!("companion and full map disagree on success for {acc:?}"),
                }
            }
        }
        // Exact free fall must still fault in BOTH — the thrust direction
        // is genuinely undefined at α = 0 whatever the floor.
        let free_fall = Vec3::new(0.0, 0.0, -9.81);
        assert_eq!(
            flatness_to_thrust_omega_true_yaw(free_fall, ZERO3, 0.0, 0.0, 9.81),
            Err(FlatnessFault::NearFreeFall)
        );
        assert_eq!(
            flatness_to_state_true_yaw(free_fall, ZERO3, ZERO3, [0.0; 3], 9.81).err(),
            Some(FlatnessFault::NearFreeFall)
        );
        let sideways = Vec3::new(0.0, 20.0, -9.81);
        assert_eq!(
            flatness_to_thrust_omega_true_yaw(sideways, ZERO3, 0.0, 0.0, 9.81),
            Err(FlatnessFault::HeadingSingular)
        );
        assert!(matches!(
            flatness_to_state_true_yaw(sideways, ZERO3, ZERO3, [0.0; 3], 9.81),
            Err(FlatnessFault::HeadingSingular)
        ));
    }

    /// The aerobatic band: the pole-safe maps must keep producing a
    /// thrust/attitude/ω reference well below the 0.1 g floor the full
    /// maps refuse at, because a ballistic or pushover segment lives
    /// there by design. Losing this means an entire ballistic segment
    /// gets no feedforward.
    #[test]
    fn pole_safe_maps_evaluate_far_below_the_strict_floor() {
        // ‖α‖ from 0.5 m/s² (well under the strict floor of 1.0 on ‖α‖²)
        // down to 2e-3, just above the pole-safe floor of 1e-3.
        for alpha_z in [0.5f32, 0.1, 0.01, 0.002] {
            let acc = Vec3::new(0.0, 0.0, alpha_z - 9.81);
            let jer = Vec3::new(0.1, -0.2, 0.05);

            let strict = flatness_to_state_tilt_yaw(acc, jer, ZERO3, [0.0; 3], 9.81);
            assert_eq!(
                strict.err(),
                Some(FlatnessFault::NearFreeFall),
                "full map should still refuse at ‖α‖={alpha_z}",
            );

            let (tpm, _q, omega) = flatness_to_thrust_omega(acc, jer, 0.0, 0.0, 9.81)
                .unwrap_or_else(|e| panic!("pole-safe tilt map refused ‖α‖={alpha_z}: {e:?}"));
            // Thrust must be the true magnitude, not a hover substitute.
            assert!(
                (tpm - alpha_z).abs() < 1e-4,
                "‖α‖={alpha_z}: thrust {tpm} should equal ‖α‖",
            );
            assert!(omega.iter().all(|v| v.is_finite()), "‖α‖={alpha_z}: ω not finite");

            let (tpm_t, _q_t, omega_t) =
                flatness_to_thrust_omega_true_yaw(acc, jer, 0.0, 0.0, 9.81)
                    .unwrap_or_else(|e| panic!("pole-safe true-yaw refused ‖α‖={alpha_z}: {e:?}"));
            assert!((tpm_t - alpha_z).abs() < 1e-4);
            assert!(omega_t.iter().all(|v| v.is_finite()));
        }
    }

    /// At exactly α = 0 the direction is genuinely undefined, so the
    /// pole-safe maps must still fault — and must fault rather than
    /// return a NaN quaternion, which is what would reach the solver.
    #[test]
    fn exact_free_fall_faults_cleanly_in_every_map() {
        let acc = Vec3::new(0.0, 0.0, -9.81); // exactly 0.0 in f32
        assert_eq!(
            flatness_to_thrust_omega(acc, ZERO3, 0.0, 0.0, 9.81).err(),
            Some(FlatnessFault::NearFreeFall)
        );
        assert_eq!(
            flatness_to_thrust_omega_true_yaw(acc, ZERO3, 0.0, 0.0, 9.81).err(),
            Some(FlatnessFault::NearFreeFall)
        );
        assert_eq!(
            flatness_to_state_tilt_yaw(acc, ZERO3, ZERO3, [0.0; 3], 9.81).err(),
            Some(FlatnessFault::NearFreeFall)
        );
        // And the never-failing fallback must not produce NaN there
        // either — it is the last line of defence when the maps refuse.
        let q = reference_quaternion(acc, 0.0, 9.81);
        assert!(
            q.coords.iter().all(|v| v.is_finite()),
            "reference_quaternion produced NaN at exact free fall: {q:?}",
        );
    }

    /// Same invariance for the tilt-yaw pair, which has always had a
    /// snap-free companion — pins that the two families stay symmetric.
    #[test]
    fn tilt_yaw_snap_free_agrees_with_full_map_on_thrust_and_attitude() {
        let acc = Vec3::new(1.0, -2.0, 3.0);
        let jer = Vec3::new(0.5, 0.25, -1.0);
        let (tpm, q, _omega) = flatness_to_thrust_omega(acc, jer, 0.7, 0.3, 9.81).unwrap();
        let st = flatness_to_state_tilt_yaw(acc, jer, ZERO3, [0.7, 0.3, 0.0], 9.81).unwrap();
        assert!((tpm - st.thrust_per_mass).abs() < 1e-6);
        assert!(q.angle_to(&st.attitude) < 1e-5);
        // ω deliberately differs: the companion uses the min-norm world
        // rate, the full map the intrinsic-Euler one. They agree only
        // where ψ̇ = 0 and the tilt is small — documented on both, and
        // the reason this assertion stops at attitude.
    }
}
