//! MPC outer-loop task: position/attitude controller using
//! `SimpleSqpSolver` over `QuadModel` (or, with `outer_mpc_full`,
//! `FullSqpSolver` over the 13-state `FullQuadModel` publishing
//! (T_d, τ_d) for the INDI α inner loop), ticking at `mpc_rate_hz`
//! (default 50 Hz, reboot-flagged). Publishes body-rate +
//! collective-thrust commands to `super::RATE_COMMAND` for the INDI
//! inner loop to consume.
//!
//! Lives in its own embassy task to avoid blocking the 8 kHz INDI loop.
//! Stack-allocated locals are tiny (~1 KB); the ~32 KB SQP workspace lives
//! in BSS via `static_cell::StaticCell`.
//!
//! Gated on `cfg(feature = "outer_mpc")`.

#![cfg(feature = "outer_mpc")]

use cybflight_core::mpc::model_utils;
#[cfg(not(feature = "outer_mpc_full"))]
use cybflight_core::mpc::cost_adapt::{
    situation_obs, CostNominal, CostPolicy, SituationInputs, NZ as COST_NZ,
    OBS_DIM as COST_OBS_DIM,
};
#[cfg(not(feature = "outer_mpc_full"))]
use cybflight_core::mpc::quad_model::{N as MPC_N, NU as MPC_NU, NX as MPC_NX, PosCostMode};
#[cfg(not(feature = "outer_mpc_full"))]
use cybflight_core::mpc::{QuadModel, SimpleQuadProblem, SimpleSqpSolver};
#[cfg(feature = "outer_mpc_full")]
use cybflight_core::mpc::full_quad_model::{N as MPC_N, NU as MPC_NU, NX as MPC_NX};
#[cfg(feature = "outer_mpc_full")]
use cybflight_core::mpc::{FullQuadModel, FullQuadProblem, FullSqpSolver};
use cybflight_core::params::FirmwareConfig;
use cybflight_core::rotation::quaternion_from_zb_and_yaw;
use cybflight_core::trajectory_planning::flatness::{
    flatness_to_thrust_omega_tilt_yaw, flatness_to_thrust_omega_true_yaw, FlatnessFault,
};
use cybflight_core::trajectory_planning::lookahead_yaw::{chord_heading, slew_heading};
use cybflight_core::trajectory_planning::sampler::{
    PositionSampler, Sampler, SamplerInputs, SamplerKind, SamplerNode, TimeSampler,
};
use embassy_time::{Duration, Instant, Ticker};
use nalgebra::{SVector, UnitQuaternion, Vector3};
use static_cell::StaticCell;

type MpcStateVec = SVector<f32, MPC_NX>;
type MpcInputVec = SVector<f32, MPC_NU>;

// The outer-loop solver family is a compile-time choice
// (`build: outer_loop: mpc` vs `mpc_full`); everything downstream of these
// aliases is family-agnostic except the sites marked with
// `#[cfg(feature = "outer_mpc_full")]`.
#[cfg(not(feature = "outer_mpc_full"))]
type OuterSolver = SimpleSqpSolver;
#[cfg(not(feature = "outer_mpc_full"))]
type OuterProblem = SimpleQuadProblem;
#[cfg(feature = "outer_mpc_full")]
type OuterSolver = FullSqpSolver;
#[cfg(feature = "outer_mpc_full")]
type OuterProblem = FullQuadProblem;

/// Per-stage input reference from (collective thrust [N], body-rate ref).
///
/// - Reduced model: inputs are `[T, ωx, ωy, ωz]` — thrust + rate command.
/// - Full model: inputs are per-motor thrusts; the reference is an even
///   split of the collective (the rate reference has no input slot — body
///   rates are *states* there, costed against `x_refs`).
#[cfg(not(feature = "outer_mpc_full"))]
#[inline]
fn u_ref_from_thrust_omega(thrust_n: f32, omega: Vector3<f32>) -> MpcInputVec {
    MpcInputVec::from_row_slice(&[thrust_n, omega.x, omega.y, omega.z])
}
#[cfg(feature = "outer_mpc_full")]
#[inline]
fn u_ref_from_thrust_omega(thrust_n: f32, _omega: Vector3<f32>) -> MpcInputVec {
    MpcInputVec::from_element(thrust_n * 0.25)
}

use crate::msgs;
use crate::sensors::VEHICLE_ODOMETRY;

/// Static-allocated SQP workspace (~32 KB in BSS for the reduced model,
/// ~60 KB for `outer_mpc_full`; init-once at task startup).
static MPC_SOLVER: StaticCell<OuterSolver> = StaticCell::new();

/// Maximum age of an odometry sample (against its own timestamp) we will use
/// as the MPC initial state. ESKF divergence often produces valid-looking
/// (finite) but stale odometry; without this gate the MPC would happily plan
/// from an ancient pose.
///
/// Tightened from 100 ms to 50 ms to support the position-sampler path: at
/// the planner's 4 m/s cap, 100 ms of stale `state_pos` translates to up to
/// 0.4 m of position error fed into PositionSampler's closest-point search,
/// which on a tight curve or near a self-intersection can lock the search
/// onto the wrong τ. 50 ms caps the worst-case input error at ~0.2 m.
/// TimeSampler doesn't read `state_pos` and is unaffected by this tighter
/// gate. Deliberately NOT scaled with `mpc_rate_hz`: this bounds ESKF
/// output staleness (odometry arrives at the estimator's rate regardless
/// of the outer tick), so the freshness requirement is rate-independent.
/// Fallback odometry-staleness window, used only if the configured
/// `mpc_odom_stale_s` is not usable. The live value comes from the `mpc`
/// group.
const ODOM_STALE_TIMEOUT_FALLBACK: Duration = Duration::from_millis(50);

/// Legal `mpc_rate_hz` range, mirrored from the schema metadata (the
/// derive macro exposes it only as runtime `ParamMeta`, not a const).
/// Floor: the INDI inner loop's 100 ms `CMD_STALE_TIMEOUT` must cover
/// ≥ 2.5 outer periods or a *healthy* outer loop reads as stale and
/// trips the failsafe in flight. Ceiling: bounds the SQP's share of
/// the shared executor (solve budget ≈ 8 ms).
const MPC_RATE_HZ_RANGE: (u16, u16) = (25, 200);

/// Below this `‖a + g·ẑ‖` [m/s²] the reference thrust *direction* carries
/// no information, so the attitude reference holds its previous value
/// instead of being constructed from `α/‖α‖`.
///
/// Matches `flatness::ALPHA_NORM_SQR_FLOOR_POLE_SAFE` (1e-6 on ‖α‖²), so
/// the fallback engages over exactly the band where the pole-safe maps
/// decline to produce a direction — no gap in which one path refuses and
/// the other builds a quaternion out of numerical noise.
const Q_REF_MIN_ALPHA_M_S2: f32 = 1e-3;

const POS_PUB_DECIMATION: u32 = 1;
const ATT_PUB_DECIMATION: u32 = 1;
const OCP_PUB_DECIMATION: u32 = 1;
const MISSION_PUB_DECIMATION: u32 = 5;

// Reference-quaternion / feedforward convention is selected **per
// mission** by `MissionTrajectory::flatness_map` (baked from the
// mission YAML's `flatness_map` key; `tilt_yaw` is the default):
//
// * `FlatnessMap::TiltYaw` — `flatness_to_thrust_omega` +
//   `quaternion_from_zb_and_yaw(z_b, yaw, true)`. Yaw input = intrinsic
//   tilt-then-yaw Euler angle. Singular only at full inversion (the
//   library substitutes a yaw-consistent 180° flip).
// * `FlatnessMap::TrueYaw` — `flatness_to_thrust_omega_true_yaw` (whose
//   attitude output is the reference; a fault holds the previous node's
//   reference, as the free-fall branch does). Yaw input = world-frame compass
//   heading of the body-x projection (what an RC stick or "point the
//   camera north" means by yaw). Singular when the thrust axis goes
//   horizontal along the heading frame's y-axis (≈ 90° tilt in one
//   azimuthal direction).
//
// See `offline_mission::FlatnessMap` for the full convention docs.

// The trajectory-derived `u_refs` feedforward is the `mpc_u_ref_ff`
// parameter (it used to be a `const bool` here, which is what its own
// comment asked to stop being):
//
// * `true` (the default): each Executing-tick non-past-end node has its
//   `u_refs[k] = [mass·‖α‖, ω_x, ω_y, ω_z]` populated from the
//   mission's flatness map (`flatness_to_thrust_omega` or
//   `flatness_to_thrust_omega_true_yaw`) applied to the trajectory's
//   (acc, jerk) sample. The MPC's input cost biases toward the
//   open-loop differential-flatness solution, so the SQP only has to
//   handle model error and disturbance — measurably tighter tracking
//   on aggressive (high-α, high-‖j‖) trajectories.
// * `false`: every horizon step is biased to `hover_u` regardless of
//   trajectory state. Equivalent to the pre-feedforward behaviour and
//   the A/B baseline when the feedforward needs debugging in flight.
//
// It controls only the *bias term* in the SQP's input cost. The
// trajectory-derived **state** references (`x_refs[k]` — position,
// velocity, attitude) stay active either way, so turning it off does
// not turn the trajectory into a hover.

/// Odometry-staleness window from `mpc_odom_stale_s`, degrading to
/// [`ODOM_STALE_TIMEOUT_FALLBACK`] if the configured value is not usable.
///
/// A non-finite or non-positive window would reject every odometry
/// sample, which reads downstream as a dead estimator and stalls the
/// outer loop — so this degrades rather than trusting the value.
fn odom_stale_from(p: &cybflight_core::params::FirmwareConfig) -> Duration {
    let raw = p.mpc.odom_stale_s;
    if raw.is_finite() && raw > 0.0 {
        Duration::from_micros((raw * 1.0e6) as u64)
    } else {
        defmt::warn!("outer_loop: mpc_odom_stale_s not usable — using default");
        ODOM_STALE_TIMEOUT_FALLBACK
    }
}

/// One-shot flag for the "reference at inverted pole" diagnostic warn
/// inside the per-node fan-out. Stays `true` after the first hit so the
/// log isn't flooded — the planner-side problem (or genuine acrobatic
/// intent) is the same on every subsequent node, and one warning per
/// firmware boot is enough to catch it. Only meaningful on the
/// `FlatnessMap::TiltYaw` path; the cross-product (`TrueYaw`) path has
/// a different singularity profile.
static INVERTED_REF_WARNED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// One-shot flag for the flatness-feedforward fault diagnostic. Same
/// rationale as `INVERTED_REF_WARNED`: the underlying condition (free-
/// fall α, infeasible trajectory sample) repeats on every subsequent
/// node, so one boot-level warning is enough to point a debugger at it.
static FLATNESS_U_REF_FAULT_WARNED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// One-shot flag for "learned cost withheld on a true-yaw mission" —
/// once per boot is enough to explain the `gain 0` in the trace.
#[cfg(not(feature = "outer_mpc_full"))]
static LEARNED_COST_TRUE_YAW_WARNED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Reject odometry with any non-finite component.
fn odom_is_valid(odom: &msgs::VehicleOdometry) -> bool {
    let fin = |v: &Vector3<f32>| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    let q = odom.pose.orientation.as_vector();
    // The full model consumes `twist.angular` as states 10..12, so it must
    // pass the same finiteness gate as everything else in `mpc_x0` — a NaN
    // rate would otherwise sail through here and NaN the solve. Gated to
    // `outer_mpc_full`: the reduced model never reads the field, and a
    // frame it can fly on must not be rejected (→ eventual silence) for a
    // component it does not consume.
    #[cfg(feature = "outer_mpc_full")]
    if !fin(&odom.twist.angular) {
        return false;
    }
    fin(&odom.pose.position)
        && q.x.is_finite()
        && q.y.is_finite()
        && q.z.is_finite()
        && q.w.is_finite()
        && fin(&odom.twist.linear)
}

/// Active SQP horizon from `mpc_horizon_n`, clamped to the compile-time
/// workspace capacity `MPC_N` (shortening only — see `MpcParams::horizon_n`).
#[inline]
fn active_horizon(vp: &cybflight_core::params::FirmwareConfig) -> usize {
    (vp.mpc.horizon_n as usize).clamp(1, MPC_N)
}

/// Clamp each component of `u0` into the model's per-channel control bounds.
fn clamp_mpc_output(u0: &mut MpcInputVec, u_bounds: &[[f32; 2]; MPC_NU]) {
    for i in 0..MPC_NU {
        u0[i] = u0[i].clamp(u_bounds[i][0], u_bounds[i][1]);
    }
}

/// Build the outer-loop `QuadModel` from flash params, enforcing two
/// MPCTC safety invariants:
///
/// 1. **Sampler-pairing.** `PosCostMode::Contouring` requires a sampler
///    that publishes a non-zero `xref[7..10]` along the path tangent. The
///    `PositionSampler` does so by construction; the `TimeSampler`'s
///    time-anchored samples produce near-zero `vel_ref` at stalls /
///    past-end, which would drag the contouring cost through its
///    `VEL_EPS` fallback inconsistently across the horizon. Runtime
///    guard (both selections are runtime params now): when
///    `sampler_kind != Position`, unconditionally clamp `pos_cost_mode`
///    to `Quadratic`.
///
/// 2. **Weight sanity.** In Contouring mode the cost reduces to
///    `M = w_pos[0]·I + (w_pos[2] − w_pos[0])·t̂t̂ᵀ`. The 3×3 Hessian is
///    PSD only when `w_pos[0] > 0` and `w_pos[2] ≥ 0`. A misconfigured
///    `w_pos[0] = 0` collapses the contour-direction restoring force —
///    the drone can drift orthogonally off the path indefinitely. Negative
///    weights would invert the cost and are catastrophic. Runtime guard:
///    if either condition is violated, clamp back to `Quadratic`.
///
/// In all clamp cases a `defmt::warn!` fires (once at boot, once per
/// disarmed hot-reload) so a misconfigured flash is visibly surfaced.
#[cfg(not(feature = "outer_mpc_full"))]
fn build_outer_quad_model(vp: &FirmwareConfig) -> QuadModel {
    let mut model = QuadModel::from_vehicle_params(vp);
    if vp.trajectory.sampler.kind != SamplerKind::Position
        && model.pos_cost_mode != PosCostMode::Quadratic
    {
        defmt::warn!(
            "MPC: sampler_kind != Position — \
             forcing PosCostMode::Quadratic (Contouring requires the position sampler)"
        );
        model.pos_cost_mode = PosCostMode::Quadratic;
        model.w_pos[1] = model.w_pos[0];
        model.w_pos[2] = model.w_pos[0];
    }
    if model.pos_cost_mode == PosCostMode::Contouring
        && (model.w_pos[0] <= 0.0 || model.w_pos[2] < 0.0)
    {
        defmt::warn!(
            "MPC: Contouring mode with w_pos[0]={=f32} w_pos[2]={=f32} is unsafe \
             (need w_pos[0]>0, w_pos[2]>=0) — forcing PosCostMode::Quadratic",
            model.w_pos[0],
            model.w_pos[2]
        );
        model.pos_cost_mode = PosCostMode::Quadratic;
        model.w_pos[0] = 200.0;
        model.w_pos[1] = 200.0;
        model.w_pos[2] = 200.0;
    }
    model
}

/// Build the outer-loop `FullQuadModel` from flash params
/// (`outer_mpc_full`). The MPCTC sampler-pairing invariants above do not
/// apply (the full model has no contouring mode); the one safety check
/// here is the body-rate constraint: with the barrier off (τ = 0) the
/// full model has NO rate-limit enforcement anywhere — the reduced
/// model's input bounds don't exist in this formulation — so a zero τ
/// gets a loud boot/hot-reload warning.
#[cfg(feature = "outer_mpc_full")]
fn build_outer_quad_model(vp: &FirmwareConfig) -> FullQuadModel {
    let model = FullQuadModel::from_vehicle_params(vp);
    // Inertia consistency guard: the MPC model, the (T_d, τ_d) reduction,
    // and indi_task's α = I⁻¹(τ − ω×Iω) all consume the FULL tensor via
    // `inertia_from_array`, matching INDI's G1 (which inverts the same
    // 3×3). The one remaining divergence is the validation fallback: a
    // non-SPD/non-finite tensor silently degrades the MPC side to its
    // floored diagonal while INDI's G1 still inverts whatever was pinned
    // — surface that instead of flying it.
    let (_, _, inertia_fell_back) =
        cybflight_core::mpc::full_quad_model::inertia_from_array(
            &vp.airframe.body.inertia_kg_m2,
        );
    if inertia_fell_back {
        defmt::warn!(
            "MPC full: airframe inertia tensor is not symmetric              positive-definite (or non-finite) — the MPC model and the α              inner loop fell back to the floored DIAGONAL, while INDI's G1              inverts the full matrix as pinned. Fix the vehicle YAML              inertia before flying this airframe."
        );
    }
    if model.rate_barrier_tau <= 0.0 {
        defmt::warn!(
            "MPC full: mpc_rate_barrier_tau = 0 — body-rate limits are NOT              enforced (rates are states here, not bounded inputs). Set              mpc_rate_barrier_tau (0.5 is the validated default) unless this              is a deliberate unconstrained experiment."
        );
    }
    model
}

#[embassy_executor::task]
pub async fn control_loop_task() {
    // ── Construct MPC ──────────────────────────────────────────────────
    let params = crate::params::get();

    // Outer tick rate from `mpc_rate_hz`. Reboot-flagged: read once here;
    // the disarmed hot-reload below deliberately does not rebuild the
    // ticker. Every write path (shell, YAML bake, flash replay) is
    // range-checked against the schema metadata, but a programmatic
    // `params::set()` is not — clamp defensively (degrade, don't brick)
    // instead of trusting the value into a divide.
    let rate_hz = {
        let raw = params.mpc.rate_hz;
        let clamped = raw.clamp(MPC_RATE_HZ_RANGE.0, MPC_RATE_HZ_RANGE.1);
        if clamped != raw {
            defmt::warn!(
                "MPC outer loop: mpc_rate_hz {} out of range — clamped to {}",
                raw,
                clamped
            );
        }
        clamped as u32
    };
    let tick_period = Duration::from_micros(1_000_000 / rate_hz as u64);
    // Node-0 slew step for the lookahead yaw follower: node 0 advances
    // from last tick's ψ₀ once per tick, not once per `mpc_dt`.
    let tick_dt_s = tick_period.as_micros() as f32 * 1e-6;
    // Overrun threshold: half the tick period. One warning per second at
    // most — the bench-visible signal that the configured rate does not
    // fit the CPU budget (the SQP shares its executor with USB, blackbox,
    // and the planner).
    let solve_warn_us: u64 = tick_period.as_micros() / 2;
    let mut last_overrun_warn_tick: u32 = 0u32.wrapping_sub(rate_hz);

    let mpc_solver: &mut OuterSolver = MPC_SOLVER.init(OuterSolver::new());
    // Active horizon from `mpc_horizon_n`, clamped to the workspace
    // capacity `MPC_N` (the arrays below stay `MPC_N`-sized; the solver
    // only visits stages `0..mpc_n`). `mut`: reassigned by the disarmed
    // hot-reload.
    let mut mpc_n = active_horizon(&params);
    let mut mpc_problem = OuterProblem::with_rk4(build_outer_quad_model(&params), mpc_n);
    // SQP iteration budget + KKT tolerance from the param plane
    // (`mpc_max_iters` / `mpc_kkt_tol`; defaults 1 / 1e-3 = the RTI
    // configuration). `mut`: reassigned by the disarmed hot-reload.
    let mut mpc_max_iters = params.mpc.max_iters as usize;
    let mut mpc_kkt_tol = params.mpc.kkt_tol;

    // ── Learned cost adaptation (docs/learned_mpc_cost_deploy.md) ─────
    //
    // The nominal is captured from the freshly built model (post
    // Contouring-safety clamps) so repeated per-tick modulation never
    // compounds, and re-captured on every hot-reload so the nominal the
    // policy modulates follows a `mpc_w_*` retune. The policy borrows
    // the baked flash statics; a vehicle without `mpc_cost_policy:` in
    // its YAML bakes `None` and this whole path is dead code.
    #[cfg(not(feature = "outer_mpc_full"))]
    let mut cost_nominal = CostNominal::from_model(&mpc_problem.model);
    #[cfg(not(feature = "outer_mpc_full"))]
    let cost_policy: Option<CostPolicy<'static>> = match crate::vehicle::BAKED_COST_POLICY {
        Some((weights, shapes)) => match CostPolicy::new(weights, shapes) {
            Ok(p) => {
                defmt::info!(
                    "MPC outer loop: cost policy loaded ({} weights)",
                    weights.len()
                );
                Some(p)
            }
            Err(_) => {
                // Unreachable if the bake's checks hold; degrade loudly.
                defmt::warn!("MPC outer loop: baked cost policy failed validation — disabled");
                None
            }
        },
        None => None,
    };
    // Live-tunable via the disarmed hot-reload (which also rebuilds the
    // model, so a disabled policy can never leave stale weights behind).
    #[cfg(not(feature = "outer_mpc_full"))]
    let mut learned_cost_en = params.mpc.learned_cost;
    #[cfg(not(feature = "outer_mpc_full"))]
    let mut learned_gain = params.mpc.learned_gain;
    // The residual step 9b computed for the NEXT solve, as `(raw, gained)`
    // — raw is what the policy emitted, gained is what gets applied —
    // plus the effective gain that relates them, carried alongside so the
    // record can report the gain that was actually in force for the solve
    // it describes rather than whatever `learned_gain` reads a tick later.
    //
    // Taken (and reset to nominal) at the top of every tick, so a tick
    // that never reaches its solve leaves nothing behind: the residual is
    // consumed exactly once, by the solve it was computed for, and any
    // aborted tick in between silently drops it. That is what makes
    // "stale modulation can never persist" structural rather than a
    // property of remembering to reset on each early-exit path.
    #[cfg(not(feature = "outer_mpc_full"))]
    let mut pending_z: ([f32; COST_NZ], [f32; COST_NZ], f32) =
        ([0.0; COST_NZ], [0.0; COST_NZ], 0.0);

    // ── Reference + warm-start trajectories ────────────────────────────
    // Site gravity, not a 9.81 literal: the model's `grav` already honours
    // `site.gravity_m_s2`, and a hover feedforward computed against a
    // different g would bias the input cost at every site that pins it.
    let mut hover_thrust = params.airframe.body.mass_kg * params.site.gravity_m_s2;
    let identity_x = {
        let mut x = MpcStateVec::zeros();
        x[6] = 1.0; // qw = 1 (identity quaternion in xyzw layout)
        x
    };
    let mut x_refs: [MpcStateVec; MPC_N + 1] = [identity_x; MPC_N + 1];
    // `mut`: reassigned (NOT shadowed) by the disarmed hot-reload so the
    // top-of-tick hover feedforward tracks a `param set mass`.
    let mut hover_u = u_ref_from_thrust_omega(hover_thrust, Vector3::zeros());
    let mut u_refs: [MpcInputVec; MPC_N] = [hover_u; MPC_N];
    let mut u_warm: [MpcInputVec; MPC_N] = u_refs;

    // ── Publishers / Subscribers ──────────────────────────────────────
    let pos_ctrl_pub = super::POSITION_CONTROL_SETPOINT.immediate_publisher();
    let att_ctrl_pub = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();
    let ocp_pub = super::OCP_SOLVER_OUTPUT.immediate_publisher();
    #[cfg(not(feature = "outer_mpc_full"))]
    let mpc_cost_pub = super::MPC_COST_ADAPT.immediate_publisher();
    let tracking_err_pub = super::TRACKING_ERROR.immediate_publisher();
    let mission_status_pub = super::MISSION_STATUS.immediate_publisher();
    let ctrl_sp_pub = super::CONTROL_SETPOINT_TELEM
        .publisher()
        .expect("outer_loop: CONTROL_SETPOINT_TELEM publisher");
    let mut odom_sub = VEHICLE_ODOMETRY
        .subscriber()
        .expect("outer_loop: VEHICLE_ODOMETRY subscriber");

    // ── Wait for the shared setpoint cell to be seeded + ESKF ready ────
    //
    // `rc_interpreter_task` fires `ACTIVE_SETPOINT_READY` once it has
    // written the init value (after ESKF convergence + first finite
    // origin). We are the sole waiter under `est_eskf + outer_mpc`.
    super::ACTIVE_SETPOINT_READY.wait().await;
    while !crate::estimation::ESTIMATOR_READY.load(core::sync::atomic::Ordering::Acquire) {
        embassy_time::Timer::after_millis(100).await;
    }
    defmt::info!("MPC outer loop task started ({} Hz)", rate_hz);

    // ── Param hot-reload bookkeeping (mirror of indi_task pattern) ────
    let mut local_param_ver =
        crate::params::PARAM_VERSION.load(core::sync::atomic::Ordering::Acquire);

    // Live `mpc` knobs that used to be source constants. Both are
    // reloaded in the disarmed block below, so a bench A/B needs no
    // reflash.
    let mut odom_stale_timeout = odom_stale_from(&params);
    let mut use_u_ref_ff = params.mpc.u_ref_feedforward;

    // ── Reference-state envelope (hard clamp on trajectory samples) ──
    //
    // The trajectory optimizer enforces velocity/thrust/tilt/body-rate
    // limits as **soft** penalties. A MaxIterations solve may leave
    // penalties only partly resolved, so we clamp reference components
    // to hard limits before handing them to the MPC — preventing a
    // marginally-feasible trajectory from commanding saturating
    // references that then cause tracking error → failsafe trip.
    //
    // Position reference is not clamped (the planner's head/tail are
    // algebraic boundary conditions, and the circle lives inside a
    // bounded region by construction).
    // let mut max_vel_m_s = params.planner.max_vel_m_s;
    // let mut max_tilt_rad = params.planner.max_tilt_rad;

    // ── Reference sampler ─────────────────────────────────────────────
    //
    // Runtime-selected via the `sampler_kind` param (both variants always
    // compiled). The sampler is pure — it never touches mission state,
    // the active setpoint cell, or the trajectory slot. The outer loop
    // owns the Idle ↔ Executing transition and calls `sampler.reset()`
    // on entry so a stateful variant starts each mission fresh.
    //
    // Sampler params (kind included) come from VehicleParams.sampler so
    // flash updates land via the existing PARAM_VERSION hot-reload below.
    let build_sampler = |sp: &cybflight_core::params::SamplerParams| match sp.kind {
        SamplerKind::Time => Sampler::Time(TimeSampler::new()),
        SamplerKind::Position => {
            Sampler::Position(PositionSampler::new(sp.to_position_sampler_params()))
        }
    };
    let mut sampler = build_sampler(&params.trajectory.sampler);
    let mut sample_buf: [SamplerNode; MPC_N + 1] = [SamplerNode::default(); MPC_N + 1];
    // Local mirror of the mission state observed at the END of the last
    // tick. Used purely to detect Idle→Executing edges for sampler reset;
    // never read for control decisions (those use `mission_state` which
    // is the authoritative atomic load each tick).
    let mut prev_mission_state = super::MissionState::Idle;
    // Cross-tick hemisphere anchor for the MPC's reference quaternion.
    // The SQP's per-pair sign canonicalisation (in
    // `mpc::model_utils::attitude_error`) is computed independently each
    // tick. If this tick's `q_ref[0]` lands on the opposite S³
    // hemisphere from the previous tick's, every per-node `ea[k]`
    // discontinuously flips, producing a step in the body-rate command
    // even though SO(3) is smooth. Caching the previous tick's
    // `q_ref[0]` and negating the entire horizon when the dot product
    // is negative pins the reference's hemisphere choice across ticks.
    // `None` until the first Executing tick fills it.
    let mut prev_qref_q0: Option<[f32; 4]> = None;

    // ── Tick loop (`mpc_rate_hz`, default 50 Hz) ──────────────────────
    let mut ticker = Ticker::every(tick_period);
    let mut tick: u32 = 0;
    loop {
        ticker.next().await;
        tick = tick.wrapping_add(1);

        // Take this tick's cost residual, leaving the nominal behind for
        // whatever tick reaches the next solve. See `pending_z`.
        #[cfg(not(feature = "outer_mpc_full"))]
        let (tick_z_raw, tick_z, tick_gain) =
            core::mem::replace(&mut pending_z, ([0.0; COST_NZ], [0.0; COST_NZ], 0.0));

        // 1. Snapshot the current tracked position + yaw setpoint for
        //    this tick. `ACTIVE_POSITION_SETPOINT` is the single source
        //    of truth: rc_interpreter writes it during Idle, we write it
        //    during Executing. Reading it once at the top of the tick
        //    gives a consistent view for the hover-reference path below.
        //
        //    Defensive: if somehow None (should not happen after the
        //    startup handshake), skip this tick. Liveness is preserved
        //    because rc_interpreter will seed the cell on its next
        //    frame and we'll resume.
        let (pos_setpoint, yaw_setpoint_rad): (Vector3<f32>, f32) =
            match super::read_active_setpoint() {
                Some(sp) => (sp.position, sp.yaw_rad),
                None => {
                    defmt::warn!("MPC outer loop: ACTIVE_POSITION_SETPOINT empty, skipping tick");
                    continue;
                }
            };

        // Build the yaw-only reference attitude from `yaw_setpoint_rad`.
        // This fills Idle/hover ticks and past-end horizon nodes; during
        // a mission the Executing block below overwrites non-past-end
        // nodes with per-node yaw from the mission's `MissionYawMode`
        // (and writes ψ(τ₀) back into the cell each tick, so this value
        // tracks the mission and holds its final yaw afterwards).
        let half = 0.5 * yaw_setpoint_rad;
        let (sin_h, cos_h) = (libm::sinf(half), libm::cosf(half));
        // let att_setpoint: UnitQuaternion<f32> = UnitQuaternion::new_normalize(
        //     nalgebra::Quaternion::new(cos_h, 0.0, 0.0, sin_h), // (w, x, y, z)
        // );

        // Fill the MPC attitude reference (qx, qy, qz, qw at indices 3..7)
        // for every horizon node with this yaw-only quaternion. Position
        // and velocity slots are overwritten below per-state branch.
        for k in 0..=MPC_N {
            x_refs[k][3] = 0.0;
            x_refs[k][4] = 0.0;
            x_refs[k][5] = sin_h;
            x_refs[k][6] = cos_h;
            // Full model: body rates are states 10..13, costed against
            // `x_refs` (w_rate). Zero rate is the correct reference for
            // hover, past-end nodes, and flatness-fault nodes; the
            // Executing fan-out overrides non-past-end nodes with the
            // flatness ω.
            #[cfg(feature = "outer_mpc_full")]
            {
                x_refs[k][10] = 0.0;
                x_refs[k][11] = 0.0;
                x_refs[k][12] = 0.0;
            }
        }

        // Default the input feedforward to hover at the top of every
        // tick. This is the load-bearing invariant for `MissionState ==
        // Idle`: if no Executing branch ever overrides `u_refs[k]`
        // (Idle, Planning with empty slot, abort race, etc.), the SQP
        // sees hover thrust and zero body rate as the reference target.
        // The Executing trajectory-sample block below overrides
        // per-node where appropriate; faults inside that block keep the
        // hover default that's already in place, so a partial fan-out
        // can never leave a stale Executing-tick u_refs in place.
        for k in 0..MPC_N {
            u_refs[k] = hover_u;
        }

        // 2. Hot-reload params when disarmed (mirrors indi_task's pattern).
        let armed = crate::motors::IS_ARMED.load(core::sync::atomic::Ordering::Acquire);
        if !armed {
            let cur = crate::params::PARAM_VERSION.load(core::sync::atomic::Ordering::Acquire);
            if cur != local_param_ver {
                local_param_ver = cur;
                let mut np = crate::params::get();
                // Pin the boot snapshot of the reboot-flagged identity
                // groups (airframe mass/inertia/geometry, site gravity)
                // before rebuilding. Every airframe/site field the model
                // reads is schema-flagged `reboot`, and the inner loop
                // honours that flag: INDI's G1 thrust row (t/mass), its
                // T_d → spf conversion, and the KF all capture the
                // airframe once at task start. Applying a `param set
                // mass` here mid-session would fly an MPC commanding T_d
                // scaled for a mass the inner loop does not have — the
                // exact split-brain the reboot flag exists to prevent.
                // Live-tunable groups (`mpc`, `trajectory`) reload
                // normally below.
                np.airframe = params.airframe.clone();
                np.site = params.site.clone();
                mpc_n = active_horizon(&np);
                mpc_problem = OuterProblem::with_rk4(build_outer_quad_model(&np), mpc_n);
                mpc_max_iters = np.mpc.max_iters as usize;
                mpc_kkt_tol = np.mpc.kkt_tol;
                odom_stale_timeout = odom_stale_from(&np);
                use_u_ref_ff = np.mpc.u_ref_feedforward;
                // Mass and gravity are boot-pinned (see above), so
                // hover_thrust/hover_u are stable across reloads;
                // recomputed for consistency with the model rebuild
                // rather than because they change.
                hover_thrust = np.airframe.body.mass_kg * np.site.gravity_m_s2;
                hover_u = u_ref_from_thrust_omega(hover_thrust, Vector3::zeros());
                u_refs = [hover_u; MPC_N];
                u_warm = u_refs;
                // max_vel_m_s = np.planner.max_vel_m_s;

                // Rebuild the sampler from the new flash params — this is
                // also how a `sampler_kind` change takes effect (no reboot
                // needed). Disarmed-only reload guarantees we don't swap a
                // sampler mid-mission. A fresh PositionSampler starts with
                // no `prev_query_tau` — equivalent to a `reset()` — which
                // is the right semantics: any tunable change invalidates
                // the last tick's converged search base.
                sampler = build_sampler(&np.trajectory.sampler);

                // Re-capture the nominal the cost policy modulates from
                // the freshly rebuilt model, so a `mpc_w_*` retune moves
                // the policy's anchor with it; reload the enable/gain
                // knobs (the ramp-in path: `param set mpc_learned_gain`
                // between flights, disarmed).
                #[cfg(not(feature = "outer_mpc_full"))]
                {
                    cost_nominal = CostNominal::from_model(&mpc_problem.model);
                    learned_cost_en = np.mpc.learned_cost;
                    learned_gain = np.mpc.learned_gain;
                }

                defmt::info!("MPC outer loop: params reloaded (ver {})", cur);
            }
        }

        // 3. Drain latest valid odometry (skip the tick if none arrived).
        //    Validity = finite components AND timestamp within
        //    `mpc_odom_stale_s` of now (and not future-dated). The timestamp
        //    gate is the C3 fix: an ESKF that hangs while still publishing
        //    finite values must not feed the MPC an ancient initial state.
        let now_for_odom = Instant::now();
        let mut latest = None;
        while let Some(o) = odom_sub.try_next_message_pure() {
            if !odom_is_valid(&o) {
                continue;
            }
            if o.timestamp > now_for_odom {
                // Future-dated → clock skew or corruption. Reject.
                continue;
            }
            if now_for_odom.duration_since(o.timestamp) > odom_stale_timeout {
                continue;
            }
            latest = Some(o);
        }
        let Some(odom) = latest else {
            defmt::warn!("MPC outer loop: no fresh odometry, skipping tick");
            continue;
        };

        // 4. Build MPC initial state from odometry.
        // QuadModel state = [px, py, pz, qx, qy, qz, qw, vx, vy, vz];
        // FullQuadModel appends the body rates [wx, wy, wz] from the
        // estimator's twist (bias-corrected body rates).
        let q = odom.pose.orientation;
        #[cfg(not(feature = "outer_mpc_full"))]
        let mpc_x0 = MpcStateVec::from_row_slice(&[
            odom.pose.position.x,
            odom.pose.position.y,
            odom.pose.position.z,
            q.i,
            q.j,
            q.k,
            q.w,
            odom.twist.linear.x,
            odom.twist.linear.y,
            odom.twist.linear.z,
        ]);
        #[cfg(feature = "outer_mpc_full")]
        let mpc_x0 = MpcStateVec::from_row_slice(&[
            odom.pose.position.x,
            odom.pose.position.y,
            odom.pose.position.z,
            q.i,
            q.j,
            q.k,
            q.w,
            odom.twist.linear.x,
            odom.twist.linear.y,
            odom.twist.linear.z,
            odom.twist.angular.x,
            odom.twist.angular.y,
            odom.twist.angular.z,
        ]);

        // 5. Refresh reference state per horizon node.
        //
        //    If a mission is Executing, sample the trajectory at
        //    τ_k = τ₀ + k · MPC_DT with τ₀ = now − t_start. Each node gets
        //    its OWN position + velocity reference so the MPC tracks the
        //    trajectory's time profile rather than a single moving target.
        //
        //    Past-end samples are clamped to the final pose (zero velocity)
        //    so the terminal cost drives a clean hover at the landing point.
        //    When node 0 (τ₀) has itself passed the end, we transition
        //    Executing → Idle and clear the slot.
        //
        //    If Idle or Planning (no trajectory), all nodes get the current
        //    hover setpoint (`pos_setpoint`, last RC stick value) with zero
        //    velocity — same behavior as before the planner existed.
        let mpc_dt: f32 = mpc_problem.model.dt;
        let mut mission_state = super::MissionState::from_u8(
            super::MISSION_STATE.load(core::sync::atomic::Ordering::Acquire),
        );

        // Graceful abort path (user released AUX switch before mission end).
        //
        // We own the transition so the hover fallback point is
        // deterministic and bounded: we capture the **trajectory's**
        // reference position at the moment of abort, not the live
        // odometry. Rationale (same logic as rc_interpreter's passive
        // tracking):
        //   - trajectory samples are finite-by-construction from a
        //     polynomial with finite coefficients,
        //   - the MPC was actively driving the drone toward that
        //     reference, so its pose is close,
        //   - a spike in ESKF output at the abort instant cannot
        //     poison the hover target.
        //
        // Zero velocity and identity attitude are the natural hover
        // setpoint — the drone will roll/pitch back to level and arrest
        // whatever velocity the mission had induced.
        //
        // If state was Planning (slot empty), there is no trajectory to
        // sample; leave `pos_setpoint` at whatever pre-mission value it
        // held. The drone is still near that point because it never
        // started moving.
        if mission_state != super::MissionState::Idle
            && super::MISSION_ABORT_REQUESTED.swap(false, core::sync::atomic::Ordering::AcqRel)
        {
            let captured: Option<Vector3<f32>> = super::MISSION_TRAJECTORY_SLOT.lock(|slot| {
                let mut out = None;
                {
                    let cell = slot.borrow();
                    if let Some(traj) = cell.as_ref() {
                        let now = Instant::now();
                        let tau = if now >= traj.t_start {
                            (now.duration_since(traj.t_start).as_micros() as f32) * 1e-6
                        } else {
                            0.0
                        };
                        let tau_c = tau.clamp(0.0, traj.total_duration_s);
                        let p = traj.traj.get_pos(tau_c);
                        if p[0].is_finite() && p[1].is_finite() && p[2].is_finite() {
                            out = Some(p);
                        }
                    }
                }
                // Clear slot AND flip state to Idle under the same lock so
                // the (state, slot) pair stays consistent for any
                // concurrent observer. See mission_planner.rs publish path.
                *slot.borrow_mut() = None;
                super::MISSION_STATE.store(
                    super::MissionState::Idle as u8,
                    core::sync::atomic::Ordering::Release,
                );
                out
            });

            // Invariant (a): ACTIVE_POSITION_SETPOINT must be refreshed
            // BEFORE MISSION_STATE flips to Idle, so that rc_interpreter's
            // very first Idle tick reads the abort-point (not a stale
            // pre-mission value) as its stick-integration base.
            let now = Instant::now();
            if let Some(p) = captured {
                super::ACTIVE_POSITION_SETPOINT.lock(|cell| {
                    cell.set(Some(super::ActiveSetpoint {
                        timestamp: now,
                        position: Vector3::new(p[0], p[1], p[2]),
                        // Hold the yaw being flown at abort — snapping
                        // the setpoint to 0 would command a spin on top
                        // of an emergency stop.
                        yaw_rad: yaw_setpoint_rad,
                    }));
                });
                defmt::info!("outer_loop: mission abort honored — hovering at trajectory ref");
            } else {
                // Slot was empty (Planning phase, or race with completion).
                // Refresh the timestamp on the existing value so the
                // liveness stamp stays monotonic, but leave the position
                // unchanged — the drone hasn't moved from it yet.
                super::ACTIVE_POSITION_SETPOINT.lock(|cell| {
                    if let Some(mut sp) = cell.get() {
                        sp.timestamp = now;
                        cell.set(Some(sp));
                    }
                });
                defmt::info!(
                    "outer_loop: mission abort honored (no trajectory ref, holding prior setpoint)"
                );
            }
            // State already flipped to Idle inside the slot lock above;
            // mirror it locally so the rest of the tick takes the hover
            // branch.
            mission_state = super::MissionState::Idle;
        }

        let mut sampled_from_trajectory = false;
        // Per-tick outputs of the trajectory-sample block, produced under
        // the slot lock and consumed below to (a) refresh
        // `ACTIVE_POSITION_SETPOINT` with the τ₀ sample and (b) decide
        // whether to end the mission.
        let mut tau0_sample: Option<Vector3<f32>> = None;
        // Desired yaw at node 0 this tick — written back into
        // ACTIVE_POSITION_SETPOINT alongside the τ₀ position sample.
        // Because the per-node yaw evaluation clamps τ to the trajectory
        // end, the final Executing tick leaves the mission's *last*
        // desired yaw here, which is exactly what post-mission hover
        // must inherit as its yaw setpoint.
        let mut tau0_yaw: f32 = yaw_setpoint_rad;
        let mut mission_done_final: Option<Vector3<f32>> = None;
        // Captured for MISSION_STATUS telemetry publish below.
        let mut tau_and_duration: Option<(f32, f32)> = None;
        let mut solve_diag: Option<msgs::SolveDiagnostics> = None;
        // This tick's mission flatness convention — the learned-cost gate
        // (step 9b) withholds the policy's output off tilt-yaw.
        #[cfg_attr(feature = "outer_mpc_full", allow(unused_variables, unused_assignments))]
        let mut mission_flatness_map: Option<super::offline_mission::FlatnessMap> = None;
        if mission_state == super::MissionState::Executing {
            // Reset the sampler's per-mission state on the Idle→Executing
            // transition. TimeSampler is stateless so this is a no-op
            // today, but the hook keeps PositionSampler's `prev_query_tau`
            // honest when it lands. Detection is purely local (no atomic
            // reads): the sampler doesn't need to know about MissionState.
            if prev_mission_state != super::MissionState::Executing {
                sampler.reset();
                // Drop the cross-tick hemisphere anchor on a fresh
                // mission — the previous mission's q_ref[0] is unrelated
                // to this one, and a stale anchor could spuriously
                // negate the new horizon on the first tick.
                prev_qref_q0 = None;
            }
            // Hold the mutex across all horizon samples to avoid cloning
            // the ~2 KB polynomial. The slot is written at most once per
            // mission by the planner task, so there is no contention.
            super::MISSION_TRAJECTORY_SLOT.lock(|slot| {
                let cell = slot.borrow();
                let Some(traj) = cell.as_ref() else {
                    return; // Race: slot was cleared. Fall through to hover.
                };

                // tau0 is computed via u64 `Instant::duration_since`
                // BEFORE crossing into f32, then divided. Going to f32
                // first and subtracting would lose microsecond precision
                // on the difference and produce a controller with
                // measurably different tracking error — see the sim
                // regression snapshot.
                // Phase from the state's own epoch, not the wall clock:
                // `odom` is what `mpc_x0` was built from, so sampling the
                // reference at any other instant poses the tracking problem
                // across two different times and turns estimator scheduling
                // jitter into reference-phase jitter. The drain in step 3
                // rejects future-dated samples and anything older than
                // `mpc_odom_stale_s`, so this is bounded to
                // [now - mpc_odom_stale_s, now].
                let tau0_s = if odom.timestamp >= traj.t_start {
                    odom.timestamp.duration_since(traj.t_start).as_micros() as f32 * 1e-6
                } else {
                    0.0
                };
                let inputs = SamplerInputs {
                    traj: &traj.traj,
                    total_duration_s: traj.total_duration_s,
                    tau0_s,
                    state_pos: odom.pose.position,
                    horizon_dt: mpc_dt,
                };
                // Only the active horizon is sampled; nodes past `mpc_n`
                // keep stale data the solver never reads.
                let result = sampler.sample(&inputs, &mut sample_buf[..=mpc_n]);

                // Per-node fan-out into x_refs. Quaternion construction
                // stays here because it depends on the per-node desired
                // yaw from the mission's `MissionYawMode` and the
                // past-end identity-tilt rule (the pre-loop yaw-only fill
                // already wrote the right quaternion for those nodes, so
                // we leave it untouched).
                //
                // u_refs feedforward is populated alongside x_refs in the
                // same node loop. Each non-past-end node gets its own
                // `[T, ω_x, ω_y, ω_z]` from the pole-safe flatness map
                // applied to (acc, jerk) plus the per-node yaw pair
                // (ψ_k, ψ̇_k) — ψ̇ is nonzero only for `Schedule` missions;
                // past-end nodes bias to hover (zero body rate, mass·g
                // thrust). On a `FlatnessFault` — reachable in aerobatics,
                // where a ballistic segment drives α → 0 — that node keeps
                // the trajectory's true thrust magnitude `mass·‖α‖` with a
                // zero rate, and holds the previous attitude reference.
                // Every `u_refs` entry is projected onto the model's
                // `u_bounds` before it can bias the cost.
                let grav = mpc_problem.model.grav;
                let mass = mpc_problem.model.mass;
                let mpc_n_inputs = mpc_n; // u_refs stages 0..mpc_n (no terminal input)
                // Lookahead follower seed. Each node slews from the
                // previous node's yaw toward its look-at heading (or holds
                // it below the minimum look-at displacement); node 0 slews
                // from the live setpoint yaw (= last tick's ψ₀ via the
                // write-back below, or the entry yaw on the first
                // Executing tick). So the mission starts from the entry
                // yaw and path reversals become sweeps — never a step.
                let mut prev_psi = yaw_setpoint_rad;
                // Lookahead slew cap: the mission's rate, never above the
                // model's own yaw-rate bound (symmetric part).
                #[cfg(not(feature = "outer_mpc_full"))]
                let yaw_rate_bound = {
                    let b = mpc_problem.model.u_bounds[3];
                    b[1].min(-b[0])
                };
                #[cfg(feature = "outer_mpc_full")]
                let yaw_rate_bound = {
                    let b = mpc_problem.model.rate_bounds[2];
                    b[1].min(-b[0])
                };
                // Attitude reference to hold when a node's thrust
                // direction is degenerate (α ≈ 0 — a ballistic segment).
                // Seeded from the live estimate so even node 0 has a
                // real answer, then tracks the previous node so the
                // held value stays continuous along the horizon.
                let mut prev_q_ref = odom.pose.orientation;
                for (k, n) in sample_buf[..=mpc_n].iter().enumerate() {
                    x_refs[k][0] = n.pos[0];
                    x_refs[k][1] = n.pos[1];
                    x_refs[k][2] = n.pos[2];
                    x_refs[k][7] = n.vel[0];
                    x_refs[k][8] = n.vel[1];
                    x_refs[k][9] = n.vel[2];

                    // Per-node desired yaw (ψ_k, ψ̇_k) at τ_k = τ₀ + k·dt,
                    // clamped to the trajectory end — `YawTrajectory`
                    // extrapolates its last cubic past the end, and the
                    // clamp also makes the final tick's ψ₀ equal the
                    // mission's terminal desired yaw.
                    let tau_k =
                        (result.tau0_s + k as f32 * mpc_dt).min(traj.total_duration_s);
                    let (psi_k, dpsi_k) = match &traj.yaw {
                        super::MissionYawMode::Constant(psi0) => (*psi0, 0.0),
                        super::MissionYawMode::Schedule(yt) => {
                            let s = yt.sample(tau_k);
                            (s[0], s[1])
                        }
                        super::MissionYawMode::Lookahead { dt_s, max_rate_rad_s } => {
                            // Look at the reference point `dt_s` ahead,
                            // with the chord's analytic ψ̇ so the rate
                            // feedforward matches the per-node attitude
                            // sweep. Past the end the ahead point is
                            // pinned to the terminal pose (zero velocity).
                            let tau_ahead = tau_k + dt_s;
                            let (p_ahead, v_ahead) = if tau_ahead < traj.total_duration_s {
                                (traj.traj.get_pos(tau_ahead), traj.traj.get_vel(tau_ahead))
                            } else {
                                (traj.traj.get_pos(traj.total_duration_s), Vector3::zeros())
                            };
                            let raw = chord_heading(&n.pos, &n.vel, &p_ahead, &v_ahead);
                            let step_dt = if k == 0 { tick_dt_s } else { mpc_dt };
                            slew_heading(
                                prev_psi,
                                raw,
                                max_rate_rad_s.min(yaw_rate_bound),
                                step_dt,
                            )
                        }
                    };
                    prev_psi = psi_k;
                    if k == 0 {
                        tau0_yaw = psi_k;
                    }

                    // Single-call flatness per the mission's `flatness_map`:
                    // both maps produce the same (thrust-per-mass, attitude,
                    // body-rate) triple, so the attitude reference and the
                    // u_refs feedforward share one call per node. TrueYaw's
                    // ω needs no snap (snap only enters ω̇, which the MPC
                    // input vector `[T, ω]` doesn't carry), so the sampler's
                    // (acc, jerk) suffices for both maps.
                    //
                    // Per-node `u_refs` population. There are three
                    // sub-cases here, and the `map_attitude` cache for
                    // the `q_ref` path below depends on which one fires:
                    //
                    //  (a) Feedforward enabled, non-terminal, non-past-
                    //      end node: call the flatness map and use both
                    //      outputs (u_refs + cached attitude).
                    //  (b) Feedforward disabled OR past-end node:
                    //      `u_refs[k]` stays at the top-of-tick `hover_u`
                    //      default (we deliberately do *not* re-write
                    //      it). When the feedforward is disabled we also
                    //      need the attitude reference, so we still call
                    //      the flatness map for non-past-end nodes purely
                    //      to populate `map_attitude` — saves a redundant
                    //      quaternion construction further down. Past-end
                    //      nodes have no tilt reference (the pre-loop
                    //      yaw-only fill is the right answer), so we skip
                    //      the call.
                    //  (c) Terminal node (k == MPC_N): no `u_refs[k]`
                    //      slot exists. The `q_ref` path below falls
                    //      back to a direct per-convention construction.
                    let mut map_attitude: Option<UnitQuaternion<f32>> = None;
                    let needs_flatness_call = !n.past_end;
                    if needs_flatness_call {
                        let flat = match traj.flatness_map {
                            super::offline_mission::FlatnessMap::TiltYaw => {
                                // Closed form: ω is the angular velocity
                                // of the tilt `q_ref` itself (incl. the
                                // (1−cos θ)·φ̇ body-z term). The min-norm
                                // `flatness_to_thrust_omega` has body-z
                                // ≡ 0, which contradicts `q_ref` on any
                                // curved path under tilt and made the
                                // reduced MPC diverge on the time-optimal
                                // missions in sim (omega_ref_ab.rs).
                                flatness_to_thrust_omega_tilt_yaw(
                                    n.acc, n.jerk, psi_k, dpsi_k, grav,
                                )
                            }
                            super::offline_mission::FlatnessMap::TrueYaw => {
                                // Snap-free twin of the tilt-yaw arm above.
                                // This used to call the full
                                // `flatness_to_state_true_yaw` with a
                                // fabricated `sna = 0` and drop the
                                // resulting (therefore wrong, not merely
                                // absent) `omega_dot`. The thrust /
                                // attitude / ω it returns are identical —
                                // none of the three read snap.
                                flatness_to_thrust_omega_true_yaw(
                                    n.acc, n.jerk, psi_k, dpsi_k, grav,
                                )
                            }
                        };
                        match flat {
                            Ok((tpm, attitude, omega)) => {
                                map_attitude = Some(attitude);
                                // Full model: the flatness ω is a *state*
                                // reference (slots 10..12), not an input —
                                // written unconditionally (it is a state
                                // ref like position/attitude, not part of
                                // the `mpc_u_ref_ff`
                                // input bias) and for every non-past-end
                                // node including the terminal k == MPC_N.
                                // Clamped to `rate_bounds` for the same
                                // reason u_refs is projected onto
                                // `u_bounds` below: ω is unbounded as
                                // α → 0, and a reference is a cost bias
                                // nothing downstream would otherwise
                                // limit.
                                #[cfg(feature = "outer_mpc_full")]
                                for i in 0..3 {
                                    x_refs[k][10 + i] = omega[i].clamp(
                                        mpc_problem.model.rate_bounds[i][0],
                                        mpc_problem.model.rate_bounds[i][1],
                                    );
                                }
                                if use_u_ref_ff && k < mpc_n_inputs {
                                    u_refs[k] =
                                        u_ref_from_thrust_omega(mass * tpm, omega);
                                    // The flatness ω is unbounded as α → 0
                                    // (‖ω‖ ≈ ‖j⊥‖/‖α‖), and the pole-safe
                                    // maps now evaluate far closer to zero
                                    // α than they used to. u_refs is a cost
                                    // *bias*, not a constrained variable, so
                                    // nothing downstream would otherwise
                                    // stop an out-of-envelope rate from
                                    // dominating the input cost. Project
                                    // onto the same bounds the solver output
                                    // is projected onto.
                                    clamp_mpc_output(
                                        &mut u_refs[k],
                                        &mpc_problem.model.u_bounds,
                                    );
                                }
                            }
                            Err(fault) => {
                                if !FLATNESS_U_REF_FAULT_WARNED
                                    .swap(true, core::sync::atomic::Ordering::Relaxed)
                                {
                                    let kind: &'static str = match fault {
                                        FlatnessFault::NearFreeFall => "NearFreeFall",
                                        FlatnessFault::InvertedTilt => "InvertedTilt",
                                        FlatnessFault::HeadingSingular => "HeadingSingular",
                                    };
                                    defmt::warn!(
                                        "outer_loop: flatness fault ({=str}, node={=usize}) — holding \
                                         last attitude reference and feeding thrust=mass*||a+g|| with \
                                         zero rate. Expected on a ballistic/pushover segment; on a \
                                         trajectory that should never approach free fall it means \
                                         the sample is infeasible",
                                        kind,
                                        k,
                                    );
                                }
                                // Thrust-preserving fallback. A fault means
                                // the thrust *direction* is undefined; the
                                // *magnitude* ‖α‖ is still exact. Biasing
                                // to hover_u here would ask the SQP for 1 g
                                // at the very node where the trajectory
                                // commands a ballistic ~0 g, actively
                                // fighting the maneuver. Feed the real
                                // magnitude with zero rate instead, and let
                                // the attitude hold (below) supply the
                                // direction the map could not.
                                if use_u_ref_ff && k < mpc_n_inputs {
                                    let alpha_norm = Vector3::new(
                                        n.acc[0],
                                        n.acc[1],
                                        n.acc[2] + grav,
                                    )
                                    .norm();
                                    u_refs[k] = u_ref_from_thrust_omega(
                                        mass * alpha_norm,
                                        Vector3::zeros(),
                                    );
                                    clamp_mpc_output(
                                        &mut u_refs[k],
                                        &mpc_problem.model.u_bounds,
                                    );
                                }
                                // map_attitude stays None → q_ref path
                                // falls back to direct construction.
                            }
                        }
                    }
                    if !n.past_end {
                        // Reuse the attitude already produced by the
                        // flatness call (convention-correct for either
                        // map). For the terminal horizon node (k == MPC_N)
                        // we never entered the u_refs branch above, and on
                        // a flatness fault the cache is empty — fall back
                        // to a direct per-convention construction. The
                        // pre-loop inverted-pole diagnostic is preserved on
                        // the *fallback* tilt call only; the in-loop cached
                        // value already came from the same closed form, so
                        // re-checking its z_b would be redundant noise.
                        let acc_cmd =
                            Vector3::new(n.acc[0], n.acc[1], n.acc[2] + grav);
                        let alpha_norm = acc_cmd.norm();
                        let q_ref = if let Some(q) = map_attitude {
                            q
                        } else if alpha_norm < Q_REF_MIN_ALPHA_M_S2 {
                            // Degenerate thrust direction — hold the last
                            // reference rather than construct one.
                            //
                            // The old `acc_cmd * 1/norm.max(1e-8)` did NOT
                            // cover this. It rescues a *tiny* α (1e-9 still
                            // normalizes to a unit vector), but at EXACTLY
                            // zero it yields the zero vector, and
                            // `quaternion_from_zb_and_yaw` then calls
                            // `normalize_mut()` on it → 0/0 → a NaN
                            // quaternion. That NaN lands in `x_refs`, NaNs
                            // the solve, and trips the non-finite guard at
                            // step 7, which skips the publish. Held past
                            // INDI's 100 ms `CMD_STALE_TIMEOUT` the inner
                            // loop goes silent and `fs_ctrl_timeout_s`
                            // disarms — i.e. a ballistic reference could
                            // disarm the vehicle mid-maneuver.
                            //
                            // Exactly zero is not exotic: a planner emitting
                            // `acc.z = -9.81` against `gravity_m_s2: 9.81`
                            // sums to precisely 0.0 in f32.
                            prev_q_ref
                        } else if matches!(
                            traj.flatness_map,
                            super::offline_mission::FlatnessMap::TiltYaw
                        ) {
                            let inv_norm = 1.0 / alpha_norm;
                            let z_b = acc_cmd * inv_norm;
                            if z_b.z < -1.0 + 1e-3
                                && !INVERTED_REF_WARNED
                                    .swap(true, core::sync::atomic::Ordering::Relaxed)
                            {
                                defmt::warn!(
                                    "outer_loop: reference attitude at the inverted pole \
                                     (z_b.z={=f32}, node={=usize}, τ₀={=f32}s) — using \
                                     fallback 180° flip; trajectory may demand acrobatic flight",
                                    z_b.z,
                                    k,
                                    result.tau0_s,
                                );
                            }
                            quaternion_from_zb_and_yaw(&z_b, psi_k, true)
                        } else {
                            // TrueYaw fault (the only way to get here: the
                            // map ran for every non-past-end node). Hold,
                            // exactly like the free-fall branch. Rebuilding
                            // via `reference_quaternion` would normalize
                            // the near-zero ẑ_B × x_C cross product at the
                            // heading singularity, so body-x — and the
                            // flown heading — lands anywhere, a jump of up
                            // to 180° that the hemisphere alignment below
                            // cannot undo (it fixes sign, not heading).
                            prev_q_ref
                        };
                        prev_q_ref = q_ref;
                        x_refs[k][3] = q_ref.i; // qx
                        x_refs[k][4] = q_ref.j; // qy
                        x_refs[k][5] = q_ref.k; // qz
                        x_refs[k][6] = q_ref.w; // qw (scalar-last)

                        // Hemisphere-align this node's q_ref with the
                        // previous node's. Without this, a sign flip in
                        // the q_ref construction (either parameterisation
                        // can produce one near a pole or near a cross-
                        // product singularity) lands adjacent horizon
                        // nodes on opposite S³ hemispheres. The SQP's
                        // per-pair sign canonicalisation in
                        // `mpc::model_utils::attitude_error` then
                        // computes inconsistent ea[k] for adjacent k,
                        // and the resulting body-rate command is a
                        // non-geodesic compromise between conflicting
                        // per-node gradients. Aligning here preempts the
                        // problem at the source — the canonicalisation
                        // becomes identity (it never has to flip).
                        if k > 0 {
                            let dot = x_refs[k][3] * x_refs[k - 1][3]
                                + x_refs[k][4] * x_refs[k - 1][4]
                                + x_refs[k][5] * x_refs[k - 1][5]
                                + x_refs[k][6] * x_refs[k - 1][6];
                            if dot < 0.0 {
                                x_refs[k][3] = -x_refs[k][3];
                                x_refs[k][4] = -x_refs[k][4];
                                x_refs[k][5] = -x_refs[k][5];
                                x_refs[k][6] = -x_refs[k][6];
                            }
                        }
                    } else if k > 0 {
                        // past_end nodes inherit the pre-loop yaw-only
                        // fill (qx=qy=0, qz=sin_h, qw=cos_h). Align them
                        // too so the horizon stays on one hemisphere
                        // across the trajectory→past_end boundary.
                        let dot = x_refs[k][3] * x_refs[k - 1][3]
                            + x_refs[k][4] * x_refs[k - 1][4]
                            + x_refs[k][5] * x_refs[k - 1][5]
                            + x_refs[k][6] * x_refs[k - 1][6];
                        if dot < 0.0 {
                            x_refs[k][3] = -x_refs[k][3];
                            x_refs[k][4] = -x_refs[k][4];
                            x_refs[k][5] = -x_refs[k][5];
                            x_refs[k][6] = -x_refs[k][6];
                        }
                    }
                }

                sampled_from_trajectory = true;
                mission_flatness_map = Some(traj.flatness_map);
                tau_and_duration = Some((result.tau0_s, traj.total_duration_s));
                solve_diag = Some(traj.solve);

                // Snapshot the τ₀ sample — this is the "currently tracked
                // point" we must publish to ACTIVE_POSITION_SETPOINT each
                // tick (invariant b: every Executing tick refreshes the
                // shared cell, so its timestamp is a live liveness proof).
                // The sampler already past-end-clamped node 0 to the
                // terminal pose when `mission_done`, so we can read it
                // straight out of the buffer.
                tau0_sample = Some(sample_buf[0].pos);

                // Wall-clock mission timeout. `PositionSampler`'s
                // `mission_done` can stay false indefinitely when the
                // drone settles offset from the trajectory: the
                // forward-only closest-point search stalls at a τ short
                // of `end`, and the radius-of-acceptance check fails by
                // the same offset that caused the stall. Without an
                // external escape the mission would stay in Executing
                // forever — the drone hovers at whatever past-end
                // setpoint the sampler is feeding, but rc_interpreter
                // never gets stick control back. The grace factor is
                // multiplicative on `total_duration_s` so long missions
                // get proportionally more slack; 1.5× past the nominal
                // end is conservative enough that a well-tracked
                // mission never hits it.
                const MISSION_GRACE_FACTOR: f32 = 0.5;
                let timeout_s =
                    traj.total_duration_s + traj.total_duration_s * MISSION_GRACE_FACTOR;
                let wall_clock_timeout = tau0_s > timeout_s;

                if result.mission_done {
                    mission_done_final = Some(sample_buf[mpc_n].pos);
                } else if wall_clock_timeout {
                    // Override tau0_sample so ACTIVE_POSITION_SETPOINT
                    // lands at the trajectory's terminal pose — same
                    // hover anchor as a normal mission_done. Without
                    // this override, rc_interpreter's first Idle tick
                    // would base stick integration on the search's
                    // last-known tracking point, which on a stuck
                    // mission is mid-trajectory.
                    let end_pos = traj.traj.get_pos(traj.total_duration_s);
                    tau0_sample = Some(end_pos);
                    mission_done_final = Some(end_pos);
                    defmt::warn!(
                        "outer_loop: mission wall-clock timeout (tau0={=f32} > {=f32} s) → Idle at terminal pose",
                        tau0_s,
                        timeout_s
                    );
                }
            });

            // Cross-tick hemisphere alignment of the MPC reference
            // quaternion. If this tick's q_ref[0] is on the opposite S³
            // hemisphere from the previous tick's, negate the entire
            // horizon so the SQP's quaternion-error gradients are
            // continuous across ticks. This complements the in-horizon
            // alignment above; together they keep the SQP's per-pair
            // sign canonicalisation as a no-op.
            if let Some(prev_q0) = prev_qref_q0 {
                let dot = x_refs[0][3] * prev_q0[0]
                    + x_refs[0][4] * prev_q0[1]
                    + x_refs[0][5] * prev_q0[2]
                    + x_refs[0][6] * prev_q0[3];
                if dot < 0.0 {
                    for k in 0..=MPC_N {
                        x_refs[k][3] = -x_refs[k][3];
                        x_refs[k][4] = -x_refs[k][4];
                        x_refs[k][5] = -x_refs[k][5];
                        x_refs[k][6] = -x_refs[k][6];
                    }
                }
            }
            prev_qref_q0 = Some([x_refs[0][3], x_refs[0][4], x_refs[0][5], x_refs[0][6]]);

            // Write the shared cell with this tick's tracked reference.
            // Invariant (a): this happens BEFORE any MISSION_STATE
            // transition, so rc_interpreter's first Idle tick reads a
            // value consistent with the mission's endpoint.
            if let Some(p) = tau0_sample {
                if p[0].is_finite()
                    && p[1].is_finite()
                    && p[2].is_finite()
                    && tau0_yaw.is_finite()
                {
                    let now = Instant::now();
                    super::ACTIVE_POSITION_SETPOINT.lock(|cell| {
                        cell.set(Some(super::ActiveSetpoint {
                            timestamp: now,
                            position: Vector3::new(p[0], p[1], p[2]),
                            // Node-0 desired yaw. On the final Executing
                            // tick this is the mission's last desired yaw
                            // (τ is end-clamped), so post-mission hover
                            // and rc_interpreter's carry-forward inherit
                            // it as the standing yaw setpoint.
                            yaw_rad: tau0_yaw,
                        }));
                    });
                }
            }

            if let Some(_final_pos) = mission_done_final {
                // Cell was already refreshed to the terminal pose above.
                // Clear slot AND flip state under the same lock to keep
                // the (state, slot) pair consistent for any concurrent
                // observer (mirrors mission_planner.rs's publish path).
                super::MISSION_TRAJECTORY_SLOT.lock(|slot| {
                    *slot.borrow_mut() = None;
                    super::MISSION_STATE.store(
                        super::MissionState::Idle as u8,
                        core::sync::atomic::Ordering::Release,
                    );
                });
                defmt::info!("outer_loop: mission complete → Idle");
            }
        }

        if !sampled_from_trajectory {
            // Hover reference: single pos_setpoint, zero velocity.
            // (Hover `u_refs` was already filled at the top of the tick;
            // nothing to do for inputs here.)
            for k in 0..=MPC_N {
                x_refs[k][0] = pos_setpoint.x;
                x_refs[k][1] = pos_setpoint.y;
                x_refs[k][2] = pos_setpoint.z;
                x_refs[k][7] = 0.0;
                x_refs[k][8] = 0.0;
                x_refs[k][9] = 0.0;
            }
            // The sampler nodes are the other half of the horizon the cost
            // observation reads (step 9b's preview block). The solve does
            // not use them off-trajectory, but a stale buffer left by the
            // last mission would pair this hover `x_refs` with that
            // mission's accelerations and hand the policy an observation
            // describing no situation at all. Reset to rest so the idle
            // observation means "hover, reference at rest" — which is what
            // `x_refs` above says.
            #[cfg(not(feature = "outer_mpc_full"))]
            sample_buf[..=mpc_n].fill(SamplerNode::default());
        }

        // The one place the cost weights are written. The policy itself
        // runs AFTER the command is published (step 9b) so its ~0.2 ms
        // never delays the inner loop; what it produced last tick is
        // applied here, on the only path that reaches a solve. Applying
        // at the choke point rather than at each producer is what lets an
        // aborted tick — no odometry, no setpoint, a diverged solve —
        // cost nothing more than the residual it discarded.
        #[cfg(not(feature = "outer_mpc_full"))]
        cost_nominal.apply(&tick_z, &mut mpc_problem.model);

        // 6. Solve (`mpc_max_iters` SQP iterations, default 1 = RTI; the
        //    solve-overrun warning below is the guard rail when raising it).
        let solve_start = Instant::now();
        let result = mpc_solver.solve(
            &mpc_problem,
            &mpc_x0,
            &x_refs,
            &u_refs,
            &u_warm,
            mpc_max_iters,
            mpc_kkt_tol,
        );
        let solve_time_us = Instant::now().duration_since(solve_start).as_micros();
        // Latency-to-command budget: solve only — the cost policy runs
        // after the publish (step 9b) and its wall time rides in the
        // `/mpc_cost` record instead.
        if solve_time_us > solve_warn_us
            && tick.wrapping_sub(last_overrun_warn_tick) >= rate_hz
        {
            last_overrun_warn_tick = tick;
            defmt::warn!(
                "MPC outer loop: solve took {} us — over half the {} Hz tick period; \
                 the configured mpc_rate_hz may not fit the CPU budget",
                solve_time_us,
                rate_hz
            );
        }
        // 6b. Divergence guard — the solver detected non-finite Riccati
        //     state and rejected the step, so this tick produced no new
        //     command (u_bar holds the previous iterate, which may look
        //     plausibly finite — that is exactly why the flag exists:
        //     checking u0 alone cannot see this failure). Skip publish
        //     (silence protocol) and drop the warm start back to the
        //     reference feedforward: the divergence cause (bad x0, cost
        //     overflow) usually persists, and re-warm-starting from the
        //     rejected landscape re-runs the same failure.
        if result.diverged {
            defmt::warn!("MPC outer loop: solver diverged, skipping tick");
            u_warm = u_refs;
            // `continue` skips step 9b, so `pending_z` stays at the
            // nominal this tick's take left there and the retry solves at
            // the exact nominal — the same reason the warm start is
            // dropped.
            continue;
        }
        u_warm = *mpc_solver.u_bar();
        let mut u0 = mpc_solver.u_bar()[0];

        // 7. Non-finite guard — skip publishing on NaN/Inf. The inner loop's
        //    last_mpc_cmd_time staleness check will eventually trip the
        //    failsafe if this persists. (With the solver's sweep-level
        //    divergence check upstream this should be unreachable; kept as
        //    the last boundary before the command leaves the task.)
        if !u0.iter().all(|v| v.is_finite()) {
            defmt::warn!("MPC outer loop: non-finite output, skipping tick");
            // Reset warm-start so a transient NaN does not poison the next
            // iteration via u_warm. The next solve runs at the nominal for
            // the same reason (see the divergence guard above).
            u_warm = u_refs;
            continue;
        }

        // 7b. Bounds clamp (C1) — even with a finite solve, a diverged or
        //     numerically degraded interior can emit values outside the
        //     physical envelope. The SQP penalty (rho) makes constraint
        //     violation costly but does not enforce hard feasibility, so we
        //     project explicitly onto the model's `u_bounds` before letting
        //     the command reach INDI.
        clamp_mpc_output(&mut u0, &mpc_problem.model.u_bounds);
        // Post-clamp u0 is what INDI receives — the cost-policy
        // observation's "most recent command" (step 9b, this tick).
        #[cfg(not(feature = "outer_mpc_full"))]
        let last_u0 = u0;

        // 8. Reference attitude/velocity for telemetry come from the
        //    current reference state x_refs[0] — the trajectory sample
        //    (or hover fallback) we just handed to the SQP. Publishing the
        //    reference rather than the MPC's one-step prediction makes the
        //    downlink show what we *asked* the controller to track. Yaw is
        //    `yaw_setpoint_rad` by construction (both the flatness map and
        //    the hover fill build the reference attitude from it), so we
        //    skip the `quaternion_to_yaw` round-trip.
        let xr0 = &x_refs[0];
        let ref_att = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            xr0[6], // qw (scalar-last in state, scalar-first in nalgebra ctor)
            xr0[3], // qx
            xr0[4], // qy
            xr0[5], // qz
        ));
        let ref_vel = Vector3::new(xr0[7], xr0[8], xr0[9]);

        let publish_time = Instant::now();

        // 9. Publish to the inner loop.
        //
        // Reduced model: (collective thrust, body-rate setpoint) — INDI's
        // rate-error stage converts to angular acceleration.
        //
        // Full model (`outer_mpc_full`): the first control is reduced to
        // (T_d, τ_d) per Sun et al. T-RO 2022 eq. (32): T_d = Σu0 and the
        // RAW model torque τ(u0) from the allocation — deliberately
        // WITHOUT the gyroscopic correction. `indi_task` derives the α
        // pseudo-control at IMU rate as α = I⁻¹·(τ_d − ω×Iω) with the
        // FRESH gyro, matching the paper's inner-loop placement of
        // eq. (32) (Fig. 3) instead of freezing ω×Iω at the solve
        // instant. No rate gains anywhere in that path.
        // `body_rate_rad_s` is telemetry-only there (measured rates, so
        // the downlink still shows the rate channel).
        #[cfg(not(feature = "outer_mpc_full"))]
        let setpoint = msgs::AttitudeControlSetpoint {
            timestamp: publish_time,
            collective_thrust_n: u0[0],
            attitude_quaternion: ref_att,
            body_rate_rad_s: Vector3::new(u0[1], u0[2], u0[3]),
            torque_n_m: Vector3::zeros(),
        };
        #[cfg(feature = "outer_mpc_full")]
        let setpoint = {
            let omega_meas = Vector3::new(mpc_x0[10], mpc_x0[11], mpc_x0[12]);
            let (t_d, tau_x, tau_y, tau_z) = mpc_problem.model.alloc(&u0);
            msgs::AttitudeControlSetpoint {
                timestamp: publish_time,
                collective_thrust_n: t_d,
                attitude_quaternion: ref_att,
                body_rate_rad_s: omega_meas,
                torque_n_m: Vector3::new(tau_x, tau_y, tau_z),
            }
        };
        super::RATE_COMMAND.signal(setpoint.clone());
        ctrl_sp_pub.publish_immediate(setpoint.clone());

        // 9b. Learned cost adaptation — deliberately AFTER the command is
        //     signalled so the ~0.2 ms observation + MLP never sits in the
        //     sensor→command path. The residual produced here is consumed
        //     by the NEXT tick's solve, at the choke point before step 6:
        //     one tick (10 ms at 100 Hz) of weight staleness, measured at
        //     ≤0.2 % of the policy's gain in sim
        //     (docs/learned_mpc_cost_deploy.md; the observation's preview
        //     spans the 1 s horizon and the reward's Δz² term trained the
        //     policy smooth). `last_u0` is THIS tick's clamped command —
        //     the training env's "most recent command" semantics.
        //
        //     `mpc_learned_cost` decides whether the policy RUNS; the
        //     effective gain decides whether its output is APPLIED, and
        //     the two are deliberately separate. Running unconditionally
        //     is what makes the path observable on the bench: `param set
        //     mpc_learned_cost 1` disarmed now moves `policy_time_us` off
        //     zero and puts a live `z` in the trace, instead of leaving
        //     every knob unverifiable until the first mission is already
        //     executing. It also matches the sim controller, which has no
        //     mission state and runs the policy before every solve.
        //
        //     The output is withheld (`gain_eff = 0`, exact nominal
        //     weights) unless the tick both fanned out a trajectory and
        //     found the mission Executing: the preview is meaningless
        //     without a trajectory and the policy was never trained on
        //     hover, so off-trajectory `z` is an out-of-distribution
        //     number — worth logging, never worth flying.
        #[allow(unused_mut)]
        let mut policy_time_us: u32 = 0;
        #[cfg(not(feature = "outer_mpc_full"))]
        {
            if let Some(policy) = cost_policy.as_ref()
                && learned_cost_en
            {
                let policy_start = Instant::now();
                let inputs = SituationInputs {
                    x0: &mpc_x0,
                    body_rate: Vector3::new(
                        odom.twist.angular.x,
                        odom.twist.angular.y,
                        odom.twist.angular.z,
                    ),
                    x_refs: &x_refs[..=mpc_n],
                    nodes: &sample_buf[..=mpc_n],
                    last_u: &last_u0,
                    u_bounds: mpc_problem.model.u_bounds,
                    mass: params.airframe.body.mass_kg,
                    grav: params.site.gravity_m_s2,
                };
                let mut obs = [0.0f32; COST_OBS_DIM];
                situation_obs(&inputs, &mut obs);
                let mut z = [0.0f32; COST_NZ];
                // `act` leaves z = 0 on a non-finite forward pass.
                policy.act(&obs, &mut z);
                policy_time_us =
                    Instant::now().saturating_duration_since(policy_start).as_micros() as u32;

                let executing = super::MissionState::from_u8(
                    super::MISSION_STATE.load(core::sync::atomic::Ordering::Acquire),
                ) == super::MissionState::Executing;
                // The one gate on application. `gain_eff` — not
                // `learned_gain` — is what the record reports, so the
                // schema's "multiply `z` by `gain` for the applied
                // residual" stays literally true on every tick, and an
                // off-trajectory tick reads as "policy ran (non-zero
                // `policy_time_us`, live `z`), output withheld (gain 0)".
                //
                // Tilt-yaw only: the baked policies were trained solely on
                // tilt-yaw references, and its observation carries the
                // yaw attitude error, whose distribution true-yaw changes.
                // Same withhold (gain 0, policy still runs and logs).
                let tilt_yaw = matches!(
                    mission_flatness_map,
                    Some(super::offline_mission::FlatnessMap::TiltYaw)
                );
                if sampled_from_trajectory
                    && executing
                    && !tilt_yaw
                    && learned_gain != 0.0
                    && !LEARNED_COST_TRUE_YAW_WARNED
                        .swap(true, core::sync::atomic::Ordering::Relaxed)
                {
                    defmt::warn!(
                        "outer_loop: learned cost withheld — mission uses flatness_map \
                         true_yaw; the cost policy was trained on tilt_yaw only"
                    );
                }
                let gain_eff = if sampled_from_trajectory && executing && tilt_yaw {
                    learned_gain
                } else {
                    0.0
                };
                // Raw and gained side by side: the raw vector is what the
                // record carries (the schema declares `z` pre-gain, and
                // the sim probes log the same thing, so a flight can be
                // compared against them at any point on the ramp-in),
                // while the gained one is what the next solve applies.
                let mut gained = z;
                for v in gained.iter_mut() {
                    *v *= gain_eff;
                }
                pending_z = (z, gained, gain_eff);
            }

            // The record describes the solve that just ran, not the one
            // being set up: `tick_z_raw`/`tick_gain` are the residual and
            // the gain that were in force for it and the model still holds
            // the weights it used, since the next apply does not happen
            // until the top of the next tick. Published on every
            // commanding tick — with the policy absent or disabled that is
            // a zero residual and the nominal cost; off-trajectory it is
            // the policy's live output at `gain` 0, i.e. the same nominal
            // cost with the withheld number next to it. Either way the
            // weights a reader needs to interpret the solve sit alongside
            // it. `policy_time_us` (this tick's, not the residual's — the
            // two are the same measurement one tick apart) is the "did the
            // policy run at all" signal.
            let m = &mpc_problem.model;
            mpc_cost_pub.publish_immediate(msgs::MpcCostAdapt {
                timestamp: publish_time,
                z: tick_z_raw,
                w_contour: m.w_pos[0],
                w_lag: m.w_pos[2],
                w_rate: [m.w_input[1], m.w_input[2], m.w_input[3]],
                gain: tick_gain,
                policy_time_us,
            });
        }

        // 10. Publish telemetry for downlink (fulfils the promise in indi_task's
        //     comment that the MPC path delegates these to outer_loop).
        if tick % POS_PUB_DECIMATION == 0 {
            pos_ctrl_pub.publish_immediate(msgs::PositionControlSetpoint {
                timestamp: publish_time,
                position: pos_setpoint,
                velocity: ref_vel,
                yaw: yaw_setpoint_rad,
                // collective_thrust_n: u0[0],
            });
        }
        if tick % ATT_PUB_DECIMATION == 0 {
            // Re-publish the same setpoint step 9 handed to the inner
            // loop. Reconstructing it from `u0` here would mislabel the
            // full model's per-motor thrusts as (collective, body rates)
            // on the downlink; step 9's cfg-gated construction is the
            // single source of truth for the vector's interpretation.
            att_ctrl_pub.publish_immediate(setpoint);
        }
        if tick % OCP_PUB_DECIMATION == 0 {
            ocp_pub.publish_immediate(msgs::OcpSolverOutput {
                timestamp: publish_time,
                command: u0,
                iterations: result.iters as i32,
                converged: result.converged,
                solve_time_us,
                // This tick's cost-policy wall time (step 9b, off the
                // command path) — 0 with the policy disabled/absent.
                policy_time_us,
            });

            // Tracking error against the τ₀ reference, in the same
            // parameterization the SQP cost uses. `attitude_error`
            // returns the tilt-prioritizing 3-vec `ea` plus jacobians
            // we ignore here. Sign convention: `error = reference -
            // actual`, so we negate `(x - xref)`.
            let pos_err = xr0.fixed_rows::<3>(0) - mpc_x0.fixed_rows::<3>(0);
            let vel_err = xr0.fixed_rows::<3>(7) - mpc_x0.fixed_rows::<3>(7);
            // `model_utils::attitude_error(x, xref)` builds
            // `qa = conj(q) ⊗ qref` (the body-frame rotation from
            // actual to reference) and returns its tilt-prio 3-vec
            // parameterization. That already matches our
            // `error = reference − actual` convention — no flip.
            let (ea, _, _) = model_utils::attitude_error(&mpc_x0, xr0);
            // Reduced model: body rates are inputs, not states — there is
            // no rate reference to difference against, so publish zero.
            // Full model: rates are states 10..13; same `reference −
            // actual` convention as pos/vel above.
            #[cfg(not(feature = "outer_mpc_full"))]
            let body_rate_err = Vector3::zeros();
            #[cfg(feature = "outer_mpc_full")]
            let body_rate_err = xr0.fixed_rows::<3>(10) - mpc_x0.fixed_rows::<3>(10);
            tracking_err_pub.publish_immediate(super::TrackingError {
                timestamp: publish_time,
                pos_err,
                vel_err,
                attitude_err: ea,
                body_rate_err,
                source: super::TRACKING_ERROR_SOURCE_MPC,
            });
        }

        // Mission status heartbeat. Read the authoritative state (may have
        // been flipped to Idle above on completion or abort). For non-Executing
        // ticks, tau/duration are zero; target_position is whatever the
        // outer loop is currently tracking (pos_setpoint for hover, τ₀
        // sample for Executing).
        let final_state = super::MissionState::from_u8(
            super::MISSION_STATE.load(core::sync::atomic::Ordering::Acquire),
        );
        let (tau_pub, dur_pub) = tau_and_duration.unwrap_or((0.0, 0.0));
        let target_pub = match tau0_sample {
            Some(p) => Vector3::new(p[0], p[1], p[2]),
            None => pos_setpoint,
        };
        if tick % MISSION_PUB_DECIMATION == 0 {
            mission_status_pub.publish_immediate(msgs::MissionStatus {
                timestamp: publish_time,
                state: final_state as u8,
                tau_s: tau_pub,
                total_duration_s: dur_pub,
                target_position: target_pub,
                solve: solve_diag.unwrap_or(msgs::SolveDiagnostics::NONE),
            });
        }

        // Snapshot the authoritative mission state for the next tick's
        // edge detector. `final_state` already reflects any in-tick
        // transition (mission_done or abort), so an Executing→Idle flip
        // this tick will look like Idle→Executing on the *next* mission's
        // first Executing tick — exactly when sampler.reset() should fire.
        prev_mission_state = final_state;
    }
}
