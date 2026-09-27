//! Persistent firmware configuration: schema, registry, and flash blob.
//!
//! The tunable-parameter schema is defined by the structs in this module
//! (plus `RigidBodyParams`/`MotorParams` in `mixer.rs`), each annotated
//! with `#[derive(Params)]` — see `cybflight-params-derive`. The derive
//! generates the [`ParamGroup`] implementation: a flat, name-addressed
//! registry of typed scalar parameters. Adding a parameter is one struct
//! field with an attribute; the registry, shell surface, and flash
//! serialization all follow from it.
//!
//! # On-flash layout
//!
//! ```text
//! [0x00]  magic:   u32 = 0x43594250 ("CYBP")
//! [0x04]  version: u32 = VERSION
//! [0x08]  length:  u32 = PAYLOAD_SIZE
//! [0x0C]  crc32:   u32 (over payload only)
//! [0x10]  payload: PARAM_COUNT × 4 bytes — every parameter as one
//!         little-endian f32 slot (typed values project losslessly via
//!         `ParamValue::as_f32`), in registry index order (declaration
//!         order, depth-first).
//! ```
//!
//! The payload layout is therefore *derived from the schema*: reordering,
//! adding, or removing a field changes the layout, and `VERSION` must be
//! bumped whenever the schema changes shape. Images whose
//! magic/version/length/CRC don't match are rejected and the firmware
//! falls back to compile-time defaults.
//!
//! v31: registry-driven uniform-f32 blob replaced the hand-maintained
//! per-field serialization; `VehicleParams` became [`FirmwareConfig`]
//! with per-subsystem groups.
//! v32: added the `sensors` group and `mpc_thrust_frac`.
//! v33–v37: loop rates, battery, rpm-notch, sampler-kind, GPS-fusion and
//! peer-pose params (see git history).
//! v38: the `eskf` group — the full `EskfConfig` / `GpsGuardConfig` /
//! `MocapGuardConfig` surface, so no estimator threshold or measurement
//! σ is a hidden compile-time constant. The five `eskf_*` noise keys
//! moved out of `sensors` into `eskf.filter` (key strings unchanged).
//! v51: `mpc_drag_*` / `mpc_bodydrag_*` — the identified rotor/body drag
//! model in the MPC prediction model (docs/learned_mpc_cost_deploy.md).
//! v52: `mpc_learned_cost` / `mpc_learned_gain` — enable + ramp-in gain
//! for the baked situation-conditioned cost policy (`mpc::cost_adapt`).
//! v54: the "invisible constant" sweep — values that shaped flight
//! behaviour from source literals rather than from the registry. New:
//! the `mahony` group (the attitude reference the arming gate
//! cross-checks was entirely untunable); `pos_err_max_*` /
//! `vel_err_max_*` / `cascade_odom_stale_s`; `mpc_odom_stale_s` /
//! `mpc_u_ref_ff`; `eskf_inflation_cap` / `eskf_mag_norm_gate` /
//! `eskf_max_predict_dt_s`; `eskf_mocap_reanchor_frames` (the radius was
//! already a param, its frame count was not); `indi_nan_rampdown`;
//! `rpm_est_omega_ref` / `rpm_est_init_omega_var` /
//! `rpm_est_escape_rejects` / `rpm_est_plausible_frac`;
//! `plan_dur_min_s` / `plan_dur_max_s` / `plan_waypoint_radius` /
//! `plan_bfgs_delta_collapse`; `batt_settle_ticks` / `batt_max_cells`.
//! Same release also wired three params that existed but reached
//! nothing: `rc_min_us` / `rc_mid_us` / `rc_max_us` (the flown stick
//! calibration was a source constant) and `eskf_init_att_var_rp` on the
//! GPS guard's re-init path.
//! v55: `rc_land_lead_m` — the landing reference is now bounded relative
//! to the vehicle's own measured altitude instead of to an absolute one,
//! so landing works over terrain below the ENU origin.
//!
//! Note: the firmware's persistent storage is the name-keyed override log
//! in [`crate::param_store`], which is insensitive to schema reshapes.
//! This blob format remains for tests and host-side export tooling.

use crate::eskf::{EskfConfig, GpsGuardConfig, MocapGuardConfig};
use crate::mixer::{MotorParams, RigidBodyParams};
use crate::mpc::quad_model::PosCostMode;
use crate::param_registry::ParamGroup;
use crate::trajectory_planning::sampler::{PositionSamplerParams, SamplerKind};
use crate::trajectory_planning::types::Vec3;
use cybflight_params_derive::Params;

const MAGIC: u32 = 0x4359_4250; // "CYBP"
/// Schema-shape version, bumped whenever the registry layout changes
/// (fields added/removed/reordered — i.e. whenever `PARAM_COUNT` or index
/// order moves). The KV store is name-keyed and insensitive to reshapes;
/// this stamps the fixed-layout blob header used by tests and host export.
const VERSION: u32 = 56;
const HEADER_SIZE: usize = 16; // magic + version + length + crc

/// The `fs_ctrl_timeout_s` schema minimum, mirrored as a `const` so the
/// firmware can assert the failsafe-ordering invariant at compile time.
///
/// The derive macro takes only numeric literals in `#[param(min = ...)]`
/// and exposes the value as runtime `ParamMeta`, so this cannot be the
/// single definition; the parity test `fs_ctrl_timeout_min_matches_schema`
/// keeps the two from drifting.
pub const FS_CTRL_TIMEOUT_MIN_S: f32 = 0.05;

/// Number of scalar parameters in the schema.
pub const PARAM_COUNT: usize = <FirmwareConfig as ParamGroup>::COUNT;
/// One 4-byte f32 slot per parameter.
const PAYLOAD_SIZE: usize = 4 * PARAM_COUNT;
/// Padded to the 32-byte flash word boundary.
pub const PADDED_SIZE: usize = (HEADER_SIZE + PAYLOAD_SIZE).div_ceil(32) * 32;

// ---------------------------------------------------------------------------
// Leaf parameter groups
// ---------------------------------------------------------------------------

/// MPC tuning parameters: cost weights, discretization, and constraint penalty.
#[derive(Clone, Debug, Params)]
pub struct MpcParams {
    /// Position tracking weights [x, y, z].
    ///
    /// In `PosCostMode::Quadratic` (default) this is the per-axis position
    /// cost. In `PosCostMode::Contouring` (MPCTC) the same array is reused:
    /// `pos_weight[0]` = contour (orthogonal-to-path) weight,
    /// `pos_weight[2]` = lag (along-path) weight, `pos_weight[1]` is unused.
    #[param(keys = "mpc_w_pos_x,mpc_w_pos_y,mpc_w_pos_z", min = 0.0, max = 1e6)]
    pub pos_weight: [f32; 3],
    /// Velocity tracking weights [x, y, z].
    #[param(keys = "mpc_w_vel_x,mpc_w_vel_y,mpc_w_vel_z", min = 0.0, max = 1e6)]
    pub vel_weight: [f32; 3],
    /// Attitude tracking weights [roll, pitch, yaw].
    #[param(keys = "mpc_w_att_r,mpc_w_att_p,mpc_w_att_y", min = 0.0, max = 1e6)]
    pub att_weight: [f32; 3],
    /// Body-rate tracking weights [roll, pitch, yaw].
    #[param(keys = "mpc_w_rate_r,mpc_w_rate_p,mpc_w_rate_y", min = 0.0, max = 1e6)]
    pub rate_weight: [f32; 3],
    /// Control effort weight (uniform across motors).
    #[param(key = "mpc_w_thrust", min = 0.0, max = 1e6)]
    pub thrust_weight: f32,
    /// Integration timestep [s] for the prediction horizon.
    #[param(key = "mpc_dt", unit = "s", min = 0.005, max = 0.5)]
    pub dt: f32,
    /// Cubic constraint penalty weight (input bound enforcement).
    #[param(key = "mpc_rho", min = 0.001, max = 1e9)]
    pub rho: f32,
    /// Fraction of the summed per-motor max thrust available as the
    /// collective ceiling, applied identically by the MPC model and the
    /// trajectory planner (previously a hardcoded 0.75 in the MPC only,
    /// with the planner assuming the full sum).
    #[param(key = "mpc_thrust_frac", min = 0.1, max = 1.0)]
    pub thrust_frac: f32,
    /// Rotor-drag coefficients `c = −m·k` per body axis [N·s²/(m·rad)]
    /// for the prediction model's identified drag term
    /// (`analysis/sysid_mcap.py` emits these as `mpc_drag_*`, the same
    /// values as `sim: aero_drag`). **All-zero = drag term off** — the
    /// schema default, so a vehicle opts in via its YAML. Consumed by the
    /// reduced `QuadModel` only (`outer_mpc`); `FullQuadModel` ignores
    /// it. `Σω` is reconstructed through `c_T = max_thrust_n/ω_max²`, so
    /// adopt these together with the airframe motor block from the SAME
    /// sysid fit or the drag force is mis-scaled.
    #[param(keys = "mpc_drag_x,mpc_drag_y,mpc_drag_z", min = 0.0, max = 0.01)]
    pub drag_coeff: [f32; 3],
    /// Quadratic body-drag coefficients `½ρC_dA` per body axis [N·s²/m²]
    /// (`mpc_bodydrag_*`, the `v_b|v_b|` regressor of the same fit;
    /// dominant above ~30 m/s). **All-zero = off.** Same consumer and
    /// caveats as `mpc_drag_*`.
    #[param(keys = "mpc_bodydrag_x,mpc_bodydrag_y,mpc_bodydrag_z", min = 0.0, max = 1.0)]
    pub body_drag: [f32; 3],
    /// Run the baked situation-conditioned cost policy before every solve
    /// (`mpc::cost_adapt`, docs/learned_mpc_cost_deploy.md). Inert unless
    /// the vehicle YAML bakes a policy (`mpc_cost_policy:`); with no
    /// policy baked or `learned_gain` 0 the weights stay at the nominal.
    #[param(key = "mpc_learned_cost")]
    pub learned_cost: bool,
    /// Scale on the cost policy's output, `z_eff = gain·z` — the ramp-in
    /// knob (0 = nominal weights even with the policy running, 1 = as
    /// trained). Live-tunable; either this at 0 or `mpc_learned_cost` 0
    /// returns bit-identical nominal behaviour.
    #[param(key = "mpc_learned_gain", min = 0.0, max = 1.0)]
    pub learned_gain: f32,
    /// Position-cost formulation (`mpc_pos_cost_mode`: 0 = Quadratic,
    /// 1 = Contouring/MPCTC).
    #[param(enum_u8, key = "mpc_pos_cost_mode")]
    pub pos_cost_mode: PosCostMode,
    /// MPC outer-loop solve/tick rate, independent of the `mpc_dt` horizon spacing.
    ///
    /// Independent of `mpc_dt` (the prediction discretization) because the
    /// loop re-derives τ₀ from wall clock every tick, so the horizon
    /// spacing need not equal the tick period. The 25 Hz floor keeps the
    /// INDI inner loop's 100 ms command-staleness failsafe covering
    /// ≥ 2.5 outer periods; the 200 Hz ceiling bounds the SQP's share of
    /// the executor (solve budget ≈ 8 ms). Watch the solve-overrun
    /// warning on the bench when raising this.
    #[param(key = "mpc_rate_hz", unit = "Hz", min = 25.0, max = 200.0, reboot)]
    pub rate_hz: u16,
    /// SQP iterations per solve. 1 = RTI (real-time iteration): one
    /// warm-started Newton step per tick — the flight configuration.
    /// Higher values iterate toward `mpc_kkt_tol` within a single tick at
    /// proportionally higher CPU cost (the solve-overrun warning applies).
    #[param(key = "mpc_max_iters", min = 1.0, max = 30.0)]
    pub max_iters: u8,
    /// Active prediction-horizon length in stages (`mpc_dt` apart). The
    /// solver workspace is sized for the compile-time maximum
    /// (`quad_model::N` / `full_quad_model::N` = 20), so this can only
    /// **shorten** the horizon: values above the capacity are clamped to
    /// it at the call site. Solve time scales ~linearly with the horizon;
    /// the sim shows a 20 → 15 cut costs ~10 % tracking on the contouring
    /// cost, so shorten for CPU headroom, not accuracy. Hot-reloadable.
    #[param(key = "mpc_horizon_n", min = 1.0, max = 20.0)]
    pub horizon_n: u8,
    /// SQP convergence tolerance on the KKT residual (max |feedforward|
    /// element). With `mpc_max_iters` = 1 it only labels the result
    /// converged/not; with more iterations it enables early exit.
    #[param(key = "mpc_kkt_tol", min = 1e-6, max = 0.1)]
    pub kkt_tol: f32,
    /// Body-rate state-constraint barrier weight τ (`FullQuadModel` only —
    /// the reduced `QuadModel` bounds rates as *inputs* via `u_bounds`).
    /// **0 = state constraints off.** τ must be commensurate with the cost
    /// weights it competes against (the yaw attitude weight in
    /// particular); see docs/mpc_runtime_comparison.md for the τ sweep —
    /// 0.5 balances enforcement against task aggression on the stress
    /// suite.
    #[param(key = "mpc_rate_barrier_tau", min = 0.0, max = 100.0)]
    pub rate_barrier_tau: f32,
    /// Relaxed-barrier margin δ [rad/s]: below this distance-to-bound the
    /// log barrier switches to its quadratic extension (which also covers
    /// infeasible iterates).
    #[param(key = "mpc_rate_barrier_delta", unit = "rad/s", min = 0.01, max = 2.0)]
    pub rate_barrier_delta: f32,
    /// Maximum-tilt state-constraint limit θ_max (both MPC models —
    /// attitude is a state in each). An envelope *fence* keeping the
    /// predicted trajectory away from deep tilt, where the local SQP has a
    /// free-fall stationary point (all motors corner at zero beyond
    /// ~100°); it cannot recover states that disturbances push past it
    /// (see docs/mpc_full_indi_plan.md).
    #[param(key = "mpc_tilt_max_deg", unit = "deg", min = 10.0, max = 178.0)]
    pub tilt_max_deg: f32,
    /// Tilt-constraint barrier weight τ. **0 = tilt constraint off.**
    /// Commensurability rule as for the rate barrier: τ competes with the
    /// position/attitude weights.
    #[param(key = "mpc_tilt_barrier_tau", min = 0.0, max = 100.0)]
    pub tilt_barrier_tau: f32,
    /// Relaxed-barrier margin δ for the tilt constraint, in **cos units**
    /// (the constraint is `1 − 2(qx²+qy²) ≥ cos θ_max`; 0.05 ≈ 3.3° of
    /// margin at a 60° limit).
    #[param(key = "mpc_tilt_barrier_delta", min = 0.005, max = 0.5)]
    pub tilt_barrier_delta: f32,
    /// Oldest odometry [s] the outer loop will accept as the MPC's
    /// initial state. Beyond it the tick is skipped rather than solved
    /// from a stale `x0`.
    ///
    /// Bounds *estimator* staleness, not solver rate, so it is
    /// deliberately independent of `mpc_rate_hz`. The default 50 ms is
    /// sized for `PositionSampler`: at the planner's 4 m/s cap it caps
    /// the closest-point search's position error near 0.2 m, tight
    /// enough that the search cannot lock onto the wrong arc-length on
    /// a tight curve. `TimeSampler` does not read position and is
    /// unaffected. Raise it only with a slower vehicle.
    #[param(key = "mpc_odom_stale_s", unit = "s", min = 0.005, max = 1.0)]
    pub odom_stale_s: f32,
    /// Bias each horizon step's input cost toward the trajectory's own
    /// differentially-flat input (`true`) or toward hover (`false`).
    ///
    /// Only the input *bias* changes: the trajectory-derived state
    /// references stay active either way, so turning this off does not
    /// turn the trajectory into a hover. It exists as an in-flight A/B
    /// knob for the feedforward path, which previously required a
    /// firmware reflash to toggle.
    #[param(key = "mpc_u_ref_ff")]
    pub u_ref_feedforward: bool,
}

impl Default for MpcParams {
    fn default() -> Self {
        Self {
            pos_weight: [500.0, 500.0, 100.0],
            vel_weight: [10.0, 10.0, 10.0],
            att_weight: [5.0, 5.0, 200.0],
            rate_weight: [20.0, 20.0, 20.0],
            thrust_weight: 1.0,
            dt: 0.05,
            rho: 1e4,
            thrust_frac: 0.75,
            // Drag model off by default: the frozen sim baseline must not
            // drift, and a flying vehicle enables the identified values
            // deliberately in its YAML (docs/learned_mpc_cost_deploy.md).
            drag_coeff: [0.0; 3],
            body_drag: [0.0; 3],
            // Learned cost adaptation off by default; a vehicle with a
            // baked policy enables it at flight time via `param set` and
            // ramps `mpc_learned_gain` (docs/learned_mpc_cost_deploy.md).
            learned_cost: false,
            learned_gain: 0.0,
            pos_cost_mode: PosCostMode::Contouring,
            rate_hz: 50,
            // RTI defaults — byte-identical to the previously hardcoded
            // solve(…, 1, 1e-3) call sites.
            max_iters: 1,
            horizon_n: 20,
            kkt_tol: 1e-3,
            // State constraints off by default: only FullQuadModel consumes
            // them, and the frozen sim baseline must not drift. Flyable
            // vehicle YAMLs opt in (mpc_rate_barrier_tau: 0.5).
            rate_barrier_tau: 0.0,
            rate_barrier_delta: 0.5,
            // Tilt fence off by default (same freeze rationale). Flyable
            // vehicle YAMLs opt in (mpc_tilt_barrier_tau).
            tilt_max_deg: 60.0,
            tilt_barrier_tau: 0.0,
            tilt_barrier_delta: 0.05,
            odom_stale_s: 0.05,
            u_ref_feedforward: true,
        }
    }
}

impl MpcParams {
    /// Explicit constructor. Pass every weight literally — useful for
    /// the host simulation, which freezes its tuning baseline against
    /// future `Default` retunes (see
    /// `crates/cybflight_sim/src/plant.rs::VehicleParamsBuilder`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pos_weight: [f32; 3],
        vel_weight: [f32; 3],
        att_weight: [f32; 3],
        rate_weight: [f32; 3],
        thrust_weight: f32,
        dt: f32,
        rho: f32,
        thrust_frac: f32,
        pos_cost_mode: PosCostMode,
        rate_hz: u16,
    ) -> Self {
        Self {
            pos_weight,
            vel_weight,
            att_weight,
            rate_weight,
            thrust_weight,
            dt,
            rho,
            thrust_frac,
            pos_cost_mode,
            rate_hz,
            // Solver knobs keep their schema defaults; callers freezing a
            // sim baseline override the struct fields directly if needed.
            ..Default::default()
        }
    }
}

/// BFGS trust-region parameters.
#[derive(Clone, Debug, Params)]
pub struct BfgsTrustParams {
    /// Initial trust-region radius (decision-vector units).
    #[param(key = "plan_bfgs_delta_init", min = 1e-4, max = 1e3)]
    pub delta_init: f32,
    /// Cap on the trust-region radius.
    #[param(key = "plan_bfgs_delta_max", min = 1e-4, max = 1e4)]
    pub delta_max: f32,
    /// Acceptance threshold: accept step if actual/predicted > eta.
    #[param(key = "plan_bfgs_eta", min = 0.0, max = 0.5)]
    pub eta: f32,
    /// Gradient convergence test: ‖g‖∞ / max(1, ‖x‖∞) < g_epsilon.
    #[param(key = "plan_bfgs_g_eps", min = 1e-10, max = 1e-2)]
    pub g_epsilon: f32,
    /// Outer-iteration cap. 0 disables the cap.
    #[param(as_u16, key = "plan_bfgs_max_iter", min = 0, max = 10000)]
    pub max_iterations: usize,
    /// Cost-stagnation lookback in accepted iterations (0 disables; clamped to 15).
    ///
    /// Bounded by the solver's ring buffer (`MAX_PAST - 1`).
    #[param(as_u16, key = "plan_bfgs_past", min = 0, max = 15)]
    pub past: usize,
    /// Relative cost change over `past` iterations that triggers `Stop`.
    #[param(key = "plan_bfgs_delta_conv", min = 0.0, max = 1e-2)]
    pub delta_conv: f32,
    /// Trust radius below which the solve is declared collapsed and
    /// stops.
    ///
    /// Distinct from `plan_bfgs_g_eps`: that is convergence on a small
    /// gradient, this is failure to make progress at any step size.
    /// Raising it abandons hard problems sooner; lowering it lets the
    /// solver keep shrinking against the wall-clock budget instead.
    #[param(key = "plan_bfgs_delta_collapse", min = 1e-12, max = 1e-2)]
    pub delta_collapse: f32,
}

impl Default for BfgsTrustParams {
    fn default() -> Self {
        Self {
            delta_init: 1.0,
            delta_max: 100.0,
            eta: 0.1,
            // Tight enough to avoid premature termination in multi-waypoint
            // trajectory optimization problems (empirically the loose 1e-5
            // caused non-monotonic weight-time behavior).
            g_epsilon: 1.0e-7,
            max_iterations: 500,
            past: 3,
            delta_conv: 1.0e-8,
            delta_collapse: 1.0e-7,
        }
    }
}

impl BfgsTrustParams {
    pub fn new(
        delta_init: f32,
        delta_max: f32,
        eta: f32,
        g_epsilon: f32,
        max_iterations: usize,
        past: usize,
        delta_conv: f32,
    ) -> Self {
        Self {
            delta_init,
            delta_max,
            eta,
            g_epsilon,
            max_iterations,
            past,
            delta_conv,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Params)]
pub struct PlannerParams {
    #[param(key = "plan_max_vel", unit = "m/s", min = 0.1, max = 200.0)]
    pub max_vel_m_s: f32,
    #[param(key = "plan_max_tilt", unit = "rad", min = 0.05, max = 1.55)]
    pub max_tilt_rad: f32,
    /// Weight on total trajectory time Σ T_i (higher → faster trajectories).
    #[param(key = "plan_w_time", min = 0.0, max = 1e6)]
    pub weight_time: f32,
    /// Weight on energy (∫‖jerk‖² or ∫‖snap‖²) — controls smoothness.
    #[param(key = "plan_w_energy", min = 0.0, max = 1e6)]
    pub weight_energy: f32,
    /// Weight on velocity constraint penalty (soft ‖v‖ ≤ max_vel).
    #[param(key = "plan_w_vel", min = 0.0, max = 1e6)]
    pub weight_vel: f32,
    /// Weight on tilt angle penalty. Set to 0 to disable.
    #[param(key = "plan_w_tilt", min = 0.0, max = 1e6)]
    pub weight_tilt: f32,
    /// Weight on body rate penalty. Set to 0 to disable.
    #[param(key = "plan_w_body_rate", min = 0.0, max = 1e6)]
    pub weight_body_rate: f32,
    /// Weight on thrust constraint penalty (soft thrust bounds).
    #[param(key = "plan_w_thrust", min = 0.0, max = 1e6)]
    pub weight_thrust: f32,
    /// Smoothing width ε of the smoothed-L1 penalty; must be > 0.
    ///
    /// The cubic blend divides by it.
    #[param(key = "plan_smooth_eps", min = 1e-6, max = 10.0)]
    pub smoothing_eps: f32,
    /// Trapezoidal sub-intervals per piece for constraint evaluation (≥ 1).
    ///
    /// `n+1` uniform samples with end weights ½; 0 would make the sample
    /// step `T/0`.
    #[param(as_u16, key = "plan_num_check", min = 1, max = 64)]
    pub num_check_per_piece: usize,
    /// Shortest trajectory [s] the planner will publish.
    ///
    /// A solve that collapses every segment time reports "converged"
    /// with a duration near zero, which the tracker would fly as an
    /// instantaneous jump. Rejecting it keeps the vehicle hovering
    /// instead.
    #[param(key = "plan_dur_min_s", unit = "s", min = 0.05, max = 60.0)]
    pub duration_min_s: f32,
    /// Longest trajectory [s] the planner will publish. Belt-and-braces
    /// against a solve that never compressed its seed allocation.
    #[param(key = "plan_dur_max_s", unit = "s", min = 1.0, max = 3600.0)]
    pub duration_max_s: f32,
    /// Radius [m] of the ball each intermediate waypoint is pinned
    /// inside.
    ///
    /// The waypoints are soft: the optimizer may move one anywhere
    /// within this ball, which is what keeps the energy term from
    /// collapsing the path. Widen it to let the planner cut corners,
    /// tighten it to fly the waypoints literally. It cannot go to zero
    /// — the stereographic chart the solver optimizes over is singular
    /// at r = 0 — so the floor is enforced structurally.
    #[param(key = "plan_waypoint_radius", unit = "m", min = 1e-4, max = 10.0)]
    pub waypoint_radius_m: f32,
    /// Nonlinear solver backend tuning.
    #[param(nested)]
    pub bfgs_trust: BfgsTrustParams,
}

impl Default for PlannerParams {
    fn default() -> Self {
        Self {
            max_vel_m_s: 100.0,
            max_tilt_rad: core::f32::consts::FRAC_PI_3,
            weight_time: 1.0,
            // A small amount of jerk-integral regularization conditions the
            // BFGS landscape and typically speeds convergence by an order of
            // magnitude vs we=0. The ball-shape waypoint parameterization
            // prevents this from collapsing the path.
            weight_energy: 0.01,
            weight_vel: 0.0,
            weight_tilt: 0.0,
            weight_body_rate: 10.0,
            weight_thrust: 10.0,
            smoothing_eps: 0.01,
            num_check_per_piece: 8,
            duration_min_s: 0.5,
            duration_max_s: 120.0,
            waypoint_radius_m:
                crate::trajectory_planning::planner::DEFAULT_WAYPOINT_RADIUS,
            bfgs_trust: BfgsTrustParams::default(),
        }
    }
}

impl PlannerParams {
    /// Explicit constructor. `bfgs_trust` is a structural sub-config —
    /// callers that don't tune it can pass `BfgsTrustParams::default()`
    /// or `BfgsTrustParams::new(...)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_vel_m_s: f32,
        max_tilt_rad: f32,
        weight_time: f32,
        weight_energy: f32,
        weight_vel: f32,
        weight_tilt: f32,
        weight_body_rate: f32,
        weight_thrust: f32,
        smoothing_eps: f32,
        num_check_per_piece: usize,
        bfgs_trust: BfgsTrustParams,
    ) -> Self {
        Self {
            max_vel_m_s,
            max_tilt_rad,
            weight_time,
            weight_energy,
            weight_vel,
            weight_tilt,
            weight_body_rate,
            weight_thrust,
            smoothing_eps,
            num_check_per_piece,
            bfgs_trust,
            ..Default::default()
        }
    }
}

/// Geometric-cascade outer-loop gains (position PD + attitude→rate P).
///
/// Renamed from `ControlGains`: these tune the `outer_geometric` cascade
/// specifically, not "control" in general. Dead data under `outer_mpc`,
/// but kept in the schema in every build so the flash image stays
/// portable across feature sets. All vector gains are dimension-major:
/// `[roll/x, pitch/y, yaw/z]`.
#[derive(Clone, Debug, Params)]
pub struct CascadeParams {
    /// Position proportional gains [x, y, z].
    #[param(keys = "pos_kp_x,pos_kp_y,pos_kp_z")]
    pub pos_kp: [f32; 3],
    /// Position derivative (velocity) gains [x, y, z].
    #[param(keys = "pos_kd_x,pos_kd_y,pos_kd_z")]
    pub pos_kd: [f32; 3],
    /// Attitude error to body-rate gains [roll, pitch, yaw].
    /// Keyed (`att_k_*`) since v31 — previously persisted but untunable.
    #[param(keys = "att_k_r,att_k_p,att_k_y")]
    pub att_k_rate: [f32; 3],
    /// Body-rate error to torque gains [roll, pitch, yaw]. The second
    /// stage of the geometric attitude controller; previously hardcoded
    /// in `cascade_task.rs` on the line after the params-sourced
    /// `att_k_rate`, which made one half of the pair tunable and the
    /// other invisible.
    #[param(keys = "att_kt_r,att_kt_p,att_kt_y")]
    pub att_k_torque: [f32; 3],
    /// Geometric-cascade outer-loop tick rate.
    ///
    /// The 25 Hz floor keeps the INDI inner loop's 100 ms
    /// command-staleness failsafe covering ≥ 2.5 outer periods.
    #[param(key = "cascade_rate_hz", unit = "Hz", min = 25.0, max = 500.0)]
    pub rate_hz: u16,
    /// Per-axis clamp on the position error the PD stage acts on [m].
    ///
    /// Bounds the acceleration a large step demand can request, so a
    /// re-arm far from the setpoint ramps in instead of commanding a
    /// full-authority lunge. Previously a `Default`-impl literal that
    /// `PositionController::new` copied and nothing could reach.
    #[param(keys = "pos_err_max_x,pos_err_max_y,pos_err_max_z", unit = "m", min = 0.01, max = 100.0)]
    pub pos_err_max: [f32; 3],
    /// Per-axis clamp on the velocity error the PD stage acts on [m/s].
    /// Same rationale and history as [`Self::pos_err_max`].
    #[param(
        keys = "vel_err_max_x,vel_err_max_y,vel_err_max_z",
        unit = "m/s",
        min = 0.01,
        max = 100.0
    )]
    pub vel_err_max: [f32; 3],
    /// Oldest odometry [s] the cascade will accept as its state.
    ///
    /// Looser than the MPC's `mpc_odom_stale_s` because the cascade
    /// reads no trajectory arc-length and so has no closest-point search
    /// to mislead. It must stay well under the INDI inner loop's 100 ms
    /// command-staleness failsafe.
    #[param(key = "cascade_odom_stale_s", unit = "s", min = 0.005, max = 1.0)]
    pub odom_stale_s: f32,
}

/// Identified override of the geometry-derived G1 effectiveness matrix,
/// per-motor and per-axis. Set from an offline system identification, the
/// shell (`param set`) or the vehicle YAML.
///
/// This group holds *only* G1, and deliberately so: G1 is not an
/// independent fact but a **derived** quantity — the controller computes
/// it from mass, inertia and motor geometry, and these keys exist to
/// supersede that computation with a measured one. The genuinely
/// independent actuator facts that used to sit here (`tau`, `omega_max`,
/// `g2_*`, `nonlin`) moved to [`crate::mixer::MotorParams`], where they
/// belong: they describe the motor, not the controller reading it.
///
/// Semantics (see `IndiController::apply_effectiveness_params`): an
/// all-zero block is the "not configured" sentinel — the controller
/// derives G1 geometrically from the airframe identity. A partially-set
/// or invalid block degrades back to geometric, never to zero authority.
#[derive(Clone, Debug, Params)]
pub struct IndiEffectivenessParams {
    /// G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values.
    /// Range mirrors the controller's `G_MAG_MAX` (±1e4, ~30× a typical
    /// geometric entry) so an implausible magnitude is rejected at the
    /// write, not first discovered by the apply-path degrade.
    #[param(
        keys = "g1_fx_m0,g1_fy_m0,g1_fz_m0,g1_fx_m1,g1_fy_m1,g1_fz_m1,g1_fx_m2,g1_fy_m2,g1_fz_m2,g1_fx_m3,g1_fy_m3,g1_fz_m3",
        min = -1e4,
        max = 1e4
    )]
    pub g1_force: [[f32; 3]; 4],
    /// G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values
    #[param(
        keys = "g1_rr_m0,g1_rp_m0,g1_ry_m0,g1_rr_m1,g1_rp_m1,g1_ry_m1,g1_rr_m2,g1_rp_m2,g1_ry_m2,g1_rr_m3,g1_rp_m3,g1_ry_m3",
        min = -1e4,
        max = 1e4
    )]
    pub g1_torque: [[f32; 3]; 4],
}

impl Default for IndiEffectivenessParams {
    fn default() -> Self {
        Self {
            g1_force: [[0.0; 3]; 4],
            g1_torque: [[0.0; 3]; 4],
        }
    }
}

/// INDI controller tuning parameters.
#[derive(Clone, Debug, Params)]
pub struct IndiControllerParams {
    /// Rate error -> angular acceleration gains [roll, pitch, yaw] (rad/s^2 per rad/s).
    #[param(keys = "indi_rate_r,indi_rate_p,indi_rate_y", unit = "1/s", min = 0.0, max = 1000.0)]
    pub rate_gains: [f32; 3],
    /// Biquad low-pass cutoff for synchronized filters (Hz).
    #[param(key = "indi_sync_hz", unit = "Hz", min = 1.0, max = 500.0)]
    pub sync_filter_hz: f32,
    /// INDI steps once per this many IMU samples (control rate = IMU ODR / this).
    ///
    /// The IMU keeps its full ODR (filter margin, sysid logging); this
    /// sets how often the controller actually runs. 4 on 8 kHz vehicles (2 kHz control — an INDI step
    /// costs well over the 125 µs an 8 kHz tick allows, and 2 kHz is far
    /// above the 10–30 ms motor dynamics it closes the loop around);
    /// 1 on `imu_1khz` vehicles. Every INDI filter, notch bank, SG window
    /// and decimation counter is designed from the resulting rate at task
    /// start, hence reboot. Skipped samples are already band-limited by
    /// the reader's biquad, so this is decimation, not aliasing.
    #[param(as_u16, key = "indi_ctrl_div", min = 1.0, max = 16.0, reboot)]
    pub ctrl_decimation: usize,
    /// WLS pseudo-control weights [fx, fy, fz, roll, pitch, yaw].
    #[param(keys = "wls_wv_fx,wls_wv_fy,wls_wv_fz,wls_wv_rr,wls_wv_rp,wls_wv_ry", min = 0.001, max = 1e6)]
    pub wls_wv: [f32; 6],
    /// WLS actuator penalty weights [m0, m1, m2, m3].
    #[param(keys = "wls_wu_m0,wls_wu_m1,wls_wu_m2,wls_wu_m3", min = 0.001, max = 1e6)]
    pub wls_wu: [f32; 4],
    /// Per-motor idle throttle (normalized 0..1) held while armed and
    /// pre-launch, so props keep spinning rather than restarting under
    /// load. Prop/ESC coupled — Betaflight's `motor_idle` analogue.
    #[param(key = "indi_idle_norm", min = 0.0, max = 0.3)]
    pub idle_normalized: f32,
    /// Ground-contact detection: gyro magnitude below this counts as
    /// "not flying" [deg/s]. Used to freeze RPM-derived G2 adaptation
    /// while the airframe is sitting on its skids.
    #[param(key = "indi_ground_gyro_dps", unit = "deg/s", min = 1.0, max = 2000.0)]
    pub ground_gyro_dps: f32,
    /// Ground-contact detection: specific-force magnitude within this
    /// fraction of 1 g counts as "resting".
    #[param(key = "indi_ground_accel_g", unit = "g", min = 0.1, max = 2.0)]
    pub ground_accel_g: f32,
    /// Ground-contact detection: vertical thrust setpoint below this
    /// [m/s²] counts as "not commanding flight".
    #[param(key = "indi_ground_thrust_sp", min = 0.0, max = 50.0)]
    pub ground_thrust_sp_m_s2: f32,
    /// RPM staleness gate: how many consecutive *expected* telemetry
    /// intervals a motor may go without a usable ω before its speed is
    /// treated as stale. One stale motor disables G2 for that motor; all
    /// four stale makes INDI fall back to its internal du-based ω̇ model.
    ///
    /// The expected interval is derived from physics rather than fixed.
    /// A bidirectional-DShot ESC answers every frame, but the payload only
    /// carries a *new* speed once it has measured another 60° electrical
    /// step — between those it reports "no new commutation", which is
    /// expected traffic, not a fault. That interval scales as
    /// `π / (3 · pole_pairs · ω)`: ~120 µs at hover but milliseconds
    /// during spin-up. A fixed frame count cannot express both ends —
    /// 50 frames is 8 ms of genuine fault at hover yet ordinary traffic
    /// at 1000 rpm — so this multiplies the expected interval instead,
    /// which keeps one number meaningful across the whole RPM range.
    ///
    /// Set it from measured data, not by feel: it must exceed the longest
    /// consecutive `no_reply` run your ESC link produces in normal flight,
    /// with margin. See `DshotMotorHealth` in the firmware for the counters
    /// that measure it.
    #[param(key = "indi_rpm_stale_gaps", min = 2.0, max = 200.0)]
    pub rpm_stale_gaps: f32,
    /// Source of the ω / ω̇ that INDI's incremental law consumes.
    ///
    /// `false` — the DShot value held through dropouts (zero-order hold),
    /// low-pass filtered. This is the faithful indiflight port and depends
    /// on no model at all.
    ///
    /// `true` — the RPM Kalman filter's estimate, through the *same*
    /// low-pass. Dropouts are then bridged by the motor model instead of a
    /// constant hold, and the single-step commutation noise this ESC sends
    /// is attenuated before the derivative amplifies it. The filter runs
    /// either way; this only chooses what reaches the controller.
    ///
    /// Both paths share the low-pass, so the group delay INDI depends on
    /// (`indi_sync_hz`) is identical and the two are directly comparable
    /// in an A/B flight. The trade is that `true` makes the inner loop
    /// depend on `m*_tau` and `m*_omega_max` being *right* — an estimator
    /// bridging a gap with a wrong model produces a confidently wrong ω,
    /// where a hold merely produces a stale one. Identify those two per
    /// motor before enabling it.
    #[param(key = "indi_omega_kf")]
    pub use_kf_omega: bool,
    /// Per-tick multiplier on the held actuator state while the WLS
    /// allocator is returning NaN.
    ///
    /// The NaN path ramps the motors down rather than holding the last
    /// good command or cutting them dead, buying the `indi_nan_limit`
    /// window for the allocator to recover before the failsafe fires.
    /// The decay is per *control* tick, so the same fraction falls
    /// faster at a higher rate — retune it alongside `indi_ctrl_div`.
    /// 1.0 reproduces a plain hold.
    #[param(key = "indi_nan_rampdown", min = 0.5, max = 1.0)]
    pub nan_rampdown: f32,
}

impl Default for IndiControllerParams {
    fn default() -> Self {
        Self {
            rate_gains: [80.0, 80.0, 80.0],
            sync_filter_hz: 12.0,
            ctrl_decimation: 1,
            wls_wv: [1.0, 1.0, 50.0, 50.0, 50.0, 5.0],
            wls_wu: [1.0, 1.0, 1.0, 1.0],
            idle_normalized: 0.055,
            ground_gyro_dps: 100.0,
            ground_accel_g: 0.8,
            ground_thrust_sp_m_s2: 3.0,
            // 10 missed expected intervals. At hover the expected interval
            // floors at the DShot frame period (~200 µs), so this is ~2 ms
            // — 10 consecutive undecodable replies, which at any sane link
            // error rate is a real fault rather than chance. Re-derive from
            // measured `no_reply` runs before trusting it in flight.
            rpm_stale_gaps: 10.0,
            use_kf_omega: true,
            nan_rampdown: 0.95,
        }
    }
}

/// Transmitter and stick-input configuration.
///
/// Every value here is a property of *your* radio and its channel map,
/// not of the airframe: endpoints come from the TX's servo travel, the
/// channel indices from its mixer, the switch thresholds from where the
/// physical detents land. A different transmitter mis-scales every
/// stick and can put the arm switch on the wrong channel, so none of it
/// belongs in firmware source.
#[derive(Clone, Debug, Params)]
pub struct RcParams {
    /// Channel pulse width at full-low travel [µs].
    #[param(key = "rc_min_us", unit = "us", min = 500.0, max = 1500.0)]
    pub min_us: u16,
    /// Channel pulse width at centre detent [µs].
    #[param(key = "rc_mid_us", unit = "us", min = 800.0, max = 2200.0)]
    pub mid_us: u16,
    /// Channel pulse width at full-high travel [µs].
    #[param(key = "rc_max_us", unit = "us", min = 1500.0, max = 2500.0)]
    pub max_us: u16,
    /// Zero-based channel index of the arm switch (AETR order, so 4 =
    /// AUX1, 5 = AUX2 …).
    #[param(key = "rc_arm_channel", min = 0.0, max = 15.0)]
    pub arm_channel: u8,
    /// Arm switch reads "armed" above this pulse width [µs].
    #[param(key = "rc_arm_threshold_us", unit = "us", min = 900.0, max = 2100.0)]
    pub arm_threshold_us: u16,
    /// Throttle must be below this to permit arming [µs]
    /// (Betaflight `mincheck`).
    #[param(key = "rc_throttle_mincheck_us", unit = "us", min = 900.0, max = 1500.0)]
    pub throttle_mincheck_us: u16,
    /// Zero-based channel index of the mission trigger.
    #[param(key = "rc_mission_channel", min = 0.0, max = 15.0)]
    pub mission_channel: u8,
    /// Mission trigger asserts above this [µs] (Schmitt upper edge).
    #[param(key = "rc_mission_high_us", unit = "us", min = 900.0, max = 2100.0)]
    pub mission_high_us: u16,
    /// Mission trigger releases below this [µs] (Schmitt lower edge).
    /// Must sit below `rc_mission_high_us`; the gap is the hysteresis band.
    #[param(key = "rc_mission_low_us", unit = "us", min = 900.0, max = 2100.0)]
    pub mission_low_us: u16,
    /// Throttle below this [µs] commands a descent-to-land.
    #[param(key = "rc_throttle_land_us", unit = "us", min = 900.0, max = 1500.0)]
    pub throttle_land_us: u16,
    /// Throttle above this [µs] starts the launch latch counting.
    #[param(key = "rc_launch_us", unit = "us", min = 1000.0, max = 2100.0)]
    pub launch_us: u16,
    /// Consecutive RC frames above `rc_launch_us` before launch latches.
    #[param(key = "rc_launch_confirm_frames", min = 1.0, max = 100.0)]
    pub launch_confirm_frames: u8,
    /// Deadband on the horizontal position sticks (normalized).
    #[param(key = "rc_xy_deadband", min = 0.0, max = 0.5)]
    pub xy_deadband: f32,
    /// Deadband around throttle centre (normalized). Sized for the TX's
    /// spring slop — too small and the vehicle drifts vertically at rest.
    #[param(key = "rc_throttle_deadband", min = 0.0, max = 0.5)]
    pub throttle_deadband: f32,
    /// Deadband on the rate sticks in `outer_rate` mode (normalized).
    #[param(key = "rc_rate_deadband", min = 0.0, max = 0.5)]
    pub rate_deadband: f32,
    /// Full-stick horizontal setpoint slew rate [m/s].
    #[param(key = "rc_xy_rate_m_s", unit = "m/s", min = 0.05, max = 20.0)]
    pub xy_rate_m_s: f32,
    /// Full-stick vertical setpoint slew rate [m/s].
    #[param(key = "rc_z_rate_m_s", unit = "m/s", min = 0.05, max = 20.0)]
    pub z_rate_m_s: f32,
    /// Descent rate while landing [m/s].
    #[param(key = "rc_land_rate_m_s", unit = "m/s", min = 0.05, max = 5.0)]
    pub land_rate_m_s: f32,
    /// How far below the vehicle's own *measured* altitude the landing
    /// reference is allowed to sit [m].
    ///
    /// The landing descent is bounded relative to the vehicle rather
    /// than to an absolute altitude, because `z = 0` is the ENU origin —
    /// the mocap anchor, or wherever the first RTK fix landed — and not
    /// the ground. An absolute floor stops the descent in mid-air over
    /// any terrain below the takeoff point; a relative one follows the
    /// vehicle down over any terrain at all.
    ///
    /// It is a leash, not just a floor: it also pulls the reference back
    /// *up* if the vehicle stops descending, so a stuck airframe cannot
    /// wind the reference metres below itself and then dump the whole
    /// error when it breaks free.
    ///
    /// Sizing has one non-local coupling worth knowing. On the geometric
    /// cascade the standing thrust command at touchdown is
    /// `gravity − pos_kp_z · lead`, and it has to fall below
    /// `indi_ground_thrust_sp` for INDI to recognise ground contact.
    /// With the shipped `pos_kp_z` of 8.0 and a 3.0 threshold that wants
    /// `lead ≥ ~0.85 m`. The lower bound comes from the other side: the
    /// leash-limited descent rate is about `(pos_kp_z / pos_kd_z) · lead`,
    /// which must stay above `rc_land_rate_m_s` or the leash rather than
    /// the rate limiter governs the descent.
    #[param(key = "rc_land_lead_m", unit = "m", min = 0.2, max = 5.0)]
    pub land_lead_m: f32,
    /// Full-stick roll/pitch body-rate command in `outer_rate` mode
    /// [rad/s]. This is *stick scaling* — a pilot preference, the
    /// Betaflight "rates" analogue — and is deliberately separate from
    /// `max_rate_r/p` in `airframe`, which is the vehicle's physical
    /// rate capability used as a planner/MPC constraint. Keep it at or
    /// below the airframe limit.
    #[param(key = "rc_max_rate_rp", unit = "rad/s", min = 0.1, max = 50.0)]
    pub max_rate_rp_rad_s: f32,
    /// Full-stick yaw-rate command [rad/s]. Same stick-scaling
    /// semantics as `rc_max_rate_rp`. In `outer_rate` this is the
    /// commanded body yaw rate; in the position modes it is the slew
    /// rate of the *heading reference* the stick integrator drives —
    /// full stick turns the yaw setpoint at this rate.
    #[param(key = "rc_max_rate_yaw", unit = "rad/s", min = 0.1, max = 50.0)]
    pub max_rate_yaw_rad_s: f32,
}

impl Default for RcParams {
    fn default() -> Self {
        Self {
            min_us: 988,
            mid_us: 1500,
            max_us: 2012,
            arm_channel: 5,
            arm_threshold_us: 1500,
            throttle_mincheck_us: 1050,
            mission_channel: 4,
            mission_high_us: 1700,
            mission_low_us: 1300,
            throttle_land_us: 1100,
            launch_us: 1600,
            launch_confirm_frames: 5,
            xy_deadband: 0.05,
            throttle_deadband: 0.12,
            rate_deadband: 0.05,
            xy_rate_m_s: 1.0,
            z_rate_m_s: 0.5,
            land_rate_m_s: 0.4,
            land_lead_m: 1.0,
            max_rate_rp_rad_s: 8.0,
            max_rate_yaw_rad_s: 4.0,
        }
    }
}

/// Properties of the *place* you fly, as opposed to the vehicle.
#[derive(Clone, Debug, Params)]
pub struct SiteParams {
    /// Local gravitational acceleration [m/s²]. Varies ~0.5% between the
    /// equator and the poles.
    ///
    /// Read by every consumer that has a `FirmwareConfig` in hand: both
    /// MPC models, the planner config, the cascade, the arming accel
    /// gate, the INDI ground-contact test, the RC hover feed-forward and
    /// the host sim's plant. The `Default` impls of the models keep a
    /// `9.81` literal because they construct a nominal vehicle without
    /// params; those are test scaffolding, not a flight path.
    #[param(key = "gravity_m_s2", unit = "m/s^2", min = 9.6, max = 10.0)]
    pub gravity_m_s2: f32,
    /// Enable the stick-integrator position envelope. Defaults **off**,
    /// which reproduces the historical behaviour on GPS builds; indoor
    /// vehicles pin it on. Only bounds the RC-driven setpoint — it is
    /// not a trajectory or failsafe geofence.
    #[param(key = "fence_enable")]
    pub fence_enable: bool,
    /// Envelope half-extent along world X [m] (±).
    #[param(key = "fence_x_m", unit = "m", min = 0.1, max = 1000.0)]
    pub fence_x_m: f32,
    /// Envelope half-extent along world Y [m] (±).
    #[param(key = "fence_y_m", unit = "m", min = 0.1, max = 1000.0)]
    pub fence_y_m: f32,
    /// Envelope ceiling [m].
    #[param(key = "fence_z_max_m", unit = "m", min = 0.1, max = 1000.0)]
    pub fence_z_max_m: f32,
    /// Lowest altitude the vehicle will ever command [m].
    ///
    /// Unlike the other three envelope bounds, this one applies whether
    /// or not `fence_enable` is set: it is also the floor of the landing
    /// integrator, which needs *some* bound or holding the land command
    /// on the ground integrates the setpoint downward forever.
    ///
    /// It is negative-capable on purpose. z = 0 is the ENU origin — the
    /// mocap anchor, or wherever the first RTK fix landed — not the
    /// ground. Landing on terrain below the takeoff point requires a
    /// negative floor; leaving it at 0 there stops the descent in
    /// mid-air. Set it to the lowest terrain in the flying area,
    /// measured relative to the origin.
    #[param(key = "fence_z_min_m", unit = "m", min = -100.0, max = 1000.0)]
    pub fence_z_min_m: f32,
}

impl Default for SiteParams {
    fn default() -> Self {
        Self {
            gravity_m_s2: 9.81,
            fence_enable: false,
            fence_x_m: 2.5,
            fence_y_m: 3.5,
            fence_z_max_m: 2.0,
            fence_z_min_m: 0.0,
        }
    }
}

/// Arming preconditions and RC-loss failsafe timing.
///
/// These decide when the vehicle may spin up and when it gives up on the
/// pilot. They are exposed so an operator can read and set them rather
/// than discover them by reading firmware source — but loosening them
/// widens the window in which a degraded link or a bad attitude estimate
/// still commands motors. Treat them as safety envelope, not as nuisance
/// suppression.
#[derive(Clone, Debug, Params)]
pub struct SafetyParams {
    /// Maximum roll/pitch tilt permitted at arming [deg].
    #[param(key = "arm_max_tilt_deg", unit = "deg", min = 1.0, max = 90.0)]
    pub arm_max_tilt_deg: f32,
    /// Minimum RC link quality permitted at arming [%].
    #[param(key = "arm_min_link_quality", unit = "%", min = 0.0, max = 100.0)]
    pub arm_min_link_quality: u8,
    /// Maximum ESKF↔Mahony attitude disagreement permitted at arming [deg].
    #[param(key = "arm_eskf_mahony_tol_deg", unit = "deg", min = 0.5, max = 90.0)]
    pub arm_eskf_mahony_tol_deg: f32,
    /// Maximum |‖accel‖ − g| for the attitude-health accel gate [m/s²].
    #[param(key = "arm_accel_tol_m_s2", unit = "m/s^2", min = 0.05, max = 20.0)]
    pub arm_accel_tol_m_s2: f32,
    /// Maximum per-axis gyro magnitude for the attitude-health gate [rad/s].
    #[param(key = "arm_gyro_limit_rad_s", unit = "rad/s", min = 0.01, max = 20.0)]
    pub arm_gyro_limit_rad_s: f32,
    /// Arm switch must be held this long before arming [s].
    #[param(key = "arm_switch_hold_s", unit = "s", min = 0.0, max = 5.0)]
    pub arm_switch_hold_s: f32,
    /// Link statistics older than this block arming [s].
    #[param(key = "arm_link_stats_max_age_s", unit = "s", min = 0.05, max = 10.0)]
    pub arm_link_stats_max_age_s: f32,
    /// No valid RC frame for this long enters the failsafe guard period [s].
    #[param(key = "fs_rxloss_trigger_s", unit = "s", min = 0.02, max = 5.0)]
    pub fs_rxloss_trigger_s: f32,
    /// Total RC-loss duration before disarm [s]. Must exceed
    /// `fs_rxloss_trigger_s`.
    #[param(key = "fs_guard_period_s", unit = "s", min = 0.05, max = 30.0)]
    pub fs_guard_period_s: f32,
    /// Continuous valid RC required to clear a failsafe [s].
    #[param(key = "fs_recovery_period_s", unit = "s", min = 0.0, max = 10.0)]
    pub fs_recovery_period_s: f32,
    /// Controller-heartbeat watchdog timeout before disarm [s].
    ///
    /// The second stage of the controller-silence degradation: the DShot
    /// task first drops the motors to idle after its own much shorter
    /// stale window, and this disarms. It must stay well above that
    /// window or the two collapse into one — see
    /// [`FS_CTRL_TIMEOUT_MIN_S`], which the firmware checks against the
    /// motor stale window at compile time.
    #[param(key = "fs_ctrl_timeout_s", unit = "s", min = 0.05, max = 10.0)]
    pub fs_ctrl_timeout_s: f32,
}

impl Default for SafetyParams {
    fn default() -> Self {
        Self {
            arm_max_tilt_deg: 30.0,
            arm_min_link_quality: 50,
            arm_eskf_mahony_tol_deg: 10.0,
            arm_accel_tol_m_s2: 1.5,
            arm_gyro_limit_rad_s: 0.5,
            arm_switch_hold_s: 0.1,
            arm_link_stats_max_age_s: 0.5,
            fs_rxloss_trigger_s: 0.15,
            fs_guard_period_s: 1.5,
            fs_recovery_period_s: 0.5,
            fs_ctrl_timeout_s: 0.5,
        }
    }
}

/// Estimator fault-annunciation windows. These do not gate flight
/// directly; they decide how long a transient shows up in the
/// `ESKF_FAULTS` bitfield and the attitude-health byte, which the
/// arming gate and the GCS both read.
// `Copy`: snapshotted at estimation-task start and passed by value into
// `evaluate_faults` on every predict tick — see `FaultEvalInputs`.
#[derive(Clone, Copy, Debug, Params)]
pub struct EskfFaultParams {
    /// No position/velocity/attitude update for this long asserts the
    /// corresponding `*_STALE` fault bit [s].
    #[param(key = "eskf_fault_pos_timeout_s", unit = "s", min = 0.05, max = 60.0)]
    pub pos_timeout_s: f32,
    /// Position covariance trace above this asserts `COV_TRACE_BLOWUP` [m²].
    #[param(key = "eskf_fault_cov_blowup_m2", unit = "m^2", min = 0.1, max = 1e4)]
    pub cov_blowup_m2: f32,
    /// How long a NaN re-init keeps its fault bit asserted [s].
    #[param(key = "eskf_fault_nan_hold_s", unit = "s", min = 0.1, max = 60.0)]
    pub nan_reset_hold_s: f32,
    /// How long a guard cascade keeps its fault bit asserted [s].
    #[param(key = "eskf_fault_cascade_hold_s", unit = "s", min = 0.1, max = 60.0)]
    pub cascade_hold_s: f32,
}

impl Default for EskfFaultParams {
    fn default() -> Self {
        Self {
            pos_timeout_s: 2.0,
            cov_blowup_m2: 25.0,
            nan_reset_hold_s: 3.0,
            cascade_hold_s: 3.0,
        }
    }
}

/// Reference-sampler tuning parameters. Mirrors the fields of
/// [`PositionSamplerParams`](crate::trajectory_planning::sampler::PositionSamplerParams)
/// so a single struct holds the runtime-tunable surface for the position
/// sampler. `TimeSampler` is stateless and ignores these values.
///
/// A flash-persisted change reaches the outer loop through the existing
/// `PARAM_VERSION` hot-reload path. Bumping any of these clears the
/// sampler's `prev_query_tau` because the outer loop rebuilds the sampler
/// from scratch on reload.
#[derive(Clone, Debug, PartialEq, Params)]
pub struct SamplerParams {
    /// Reference-sampler selection (`sampler_kind`: 0 = Time, 1 = Position).
    ///
    /// Applied by the outer loop's disarmed hot-reload (the sampler is
    /// rebuilt from scratch, so no reboot flag). `PosCostMode::Contouring`
    /// requires Position; the outer loop clamps the cost mode to Quadratic
    /// (with a warning) when the pairing is violated.
    #[param(enum_u8, key = "sampler_kind")]
    pub kind: SamplerKind,
    #[param(key = "sampler_max_lag_s", unit = "s", min = 0.0, max = 5.0)]
    pub max_lag_s: f32,
    #[param(
        keys = "sampler_axis_weights_sqrt_x,sampler_axis_weights_sqrt_y,sampler_axis_weights_sqrt_z"
    )]
    pub axis_weights_sqrt: [f32; 3],
    #[param(key = "sampler_search_dt", unit = "s", min = 0.001, max = 1.0)]
    pub search_dt: f32,
    #[param(key = "sampler_max_search_steps")]
    pub max_search_steps: u16,
    #[param(key = "sampler_radius_of_acceptance")]
    pub radius_of_acceptance: f32,
    #[param(key = "sampler_max_lead_s", unit = "s", min = 0.0, max = 5.0)]
    pub max_lead_s: f32,
}

impl Default for SamplerParams {
    fn default() -> Self {
        // Mirror PositionSamplerParams::defaults() exactly. If the two ever
        // drift, the parity test below will fail loudly.
        Self {
            // Position is the flight default (matches the old Justfile
            // SAMPLER=position build and the Contouring cost-mode default).
            kind: SamplerKind::Position,
            max_lag_s: 0.1,
            axis_weights_sqrt: [1.0, 1.0, 1.0],
            search_dt: 0.01,
            max_search_steps: 100,
            radius_of_acceptance: 0.15,
            max_lead_s: 0.1,
        }
    }
}

impl SamplerParams {
    pub fn new(
        kind: SamplerKind,
        max_lag_s: f32,
        axis_weights_sqrt: [f32; 3],
        search_dt: f32,
        max_search_steps: u16,
        radius_of_acceptance: f32,
        max_lead_s: f32,
    ) -> Self {
        Self {
            kind,
            max_lag_s,
            axis_weights_sqrt,
            search_dt,
            max_search_steps,
            radius_of_acceptance,
            max_lead_s,
        }
    }

    /// Project a `SamplerParams` into the sampler-side `PositionSamplerParams`
    /// shape. The outer loop calls this when constructing or rebuilding the
    /// `Sampler::Position` instance.
    pub fn to_position_sampler_params(&self) -> PositionSamplerParams {
        PositionSamplerParams {
            axis_weights_sqrt: Vec3::new(
                self.axis_weights_sqrt[0],
                self.axis_weights_sqrt[1],
                self.axis_weights_sqrt[2],
            ),
            search_dt: self.search_dt,
            max_search_steps: self.max_search_steps,
            radius_of_acceptance: self.radius_of_acceptance,
            max_lag_s: self.max_lag_s,
            max_lead_s: self.max_lead_s,
        }
    }
}

/// Sensor signal conditioning and measurement-source policy: values that
/// used to be compile-time constants scattered across `board_init`,
/// `vehicle.rs`, and the ESKF, promoted to params in v32 so a recompile is
/// no longer needed to retune an install.
///
/// The *physical* half of the old `SensorParams` — where the antennas and
/// magnetometer sit on the frame — is now [`InstallParams`] under
/// `airframe`. What is left here travels with the tune rather than with
/// the airframe.
///
/// Defaults are the previous hardcoded values. Boards whose IMU wants
/// different LPF cutoffs (e.g. FOXEERH743's historical 20/80 Hz) set them
/// in their vehicle YAML.
#[derive(Clone, Debug, Params)]
pub struct SensorParams {
    /// IMU software LPF cutoff for accel (Hz), applied by `ImuReader`.
    /// Clamped to 0.4·ODR at init — see `imu_gyro_lpf_hz`.
    #[param(key = "imu_accel_lpf_hz", unit = "Hz", min = 1.0, max = 2000.0)]
    pub imu_accel_lpf_hz: f32,
    /// IMU software LPF cutoff for gyro (Hz), applied by `ImuReader`.
    /// `ImuReader` clamps to 0.4·ODR, so the effective ceiling is 3200 Hz
    /// at 8 kHz, 1280 Hz on the 3.2 kHz BMI270 board, and 400 Hz on an
    /// `imu_1khz` build — the schema max is the loosest of the three.
    /// Not a free knob: the hardware AAF/UI chain in front of it differs
    /// per ODR, so the same cutoff is not the same decision at every rate
    /// (docs/imu_filtering.md).
    #[param(key = "imu_gyro_lpf_hz", unit = "Hz", min = 1.0, max = 2000.0)]
    pub imu_gyro_lpf_hz: f32,
    /// Fuse GNSS velocity into the ESKF (`update_vel`). Works with any
    /// receiver.
    ///
    /// Unlike the extrinsics in [`InstallParams`], this is an estimator
    /// *policy* rather than a hardware fact: nothing about the airframe
    /// decides it, and a site with heavy multipath is a reason to turn it
    /// off without touching anything physical.
    #[param(key = "gps_fuse_vel")]
    pub gps_fuse_vel: bool,
}

impl Default for SensorParams {
    fn default() -> Self {
        Self {
            imu_accel_lpf_hz: 80.0,
            imu_gyro_lpf_hz: 200.0,
            // Matches the old GPS_FUSE_VEL=on build default — asserted by
            // `behavior_preserving_defaults`.
            gps_fuse_vel: true,
        }
    }
}

/// Where the sensors sit on *this* airframe, and what is physically
/// bolted to it — the extrinsics and calibrations that a technician
/// re-measures after a rebuild.
///
/// Split out of `SensorParams`, which had come to mix two unrelated
/// things: signal conditioning (the IMU LPF cutoffs, which are a filter
/// choice and travel with the tune) and install geometry (which travels
/// with the frame). Moving the same flight controller to another airframe
/// invalidates everything here and nothing there.
///
/// GPS extrinsics default to a null lever arm (antenna at the IMU) — a
/// graceful degradation, unlike a default mass, since a few centimetres
/// of unmodelled lever arm is a small position bias rather than an
/// unflyable vehicle.
#[derive(Clone, Debug, Params)]
pub struct InstallParams {
    /// Magnetometer hard-iron offset, sensor frame (raw units). The
    /// magnetic signature of this frame's metal and current paths, so it
    /// is re-flown from scratch after any rewiring.
    #[param(keys = "mag_hi_x,mag_hi_y,mag_hi_z")]
    pub mag_hard_iron: [f32; 3],
    /// GPS ANT1 (position antenna) phase-centre lever arm from the IMU,
    /// body/FLU frame [m].
    #[param(keys = "gps_ant_x,gps_ant_y,gps_ant_z", unit = "m", min = -2.0, max = 2.0)]
    pub gps_ant1_offset_m: [f32; 3],
    /// ANT1→ANT2 baseline unit direction, body/FLU frame (dual-antenna
    /// heading). Keep unit-norm.
    #[param(keys = "gps_base_x,gps_base_y,gps_base_z", min = -1.0, max = 1.0)]
    pub gps_baseline_body: [f32; 3],
}

impl Default for InstallParams {
    fn default() -> Self {
        Self {
            mag_hard_iron: [0.0, 0.0, 0.0],
            gps_ant1_offset_m: [0.0, 0.0, 0.0],
            gps_baseline_body: [0.0, -1.0, 0.0],
        }
    }
}

/// ESKF core filter tuning — the full [`EskfConfig`] surface.
///
/// Every field of `EskfConfig` appears here: nothing about the filter's
/// noise model or its outlier gates is a hidden constant. The five
/// `eskf_{acc,gyro,...}` noise keys moved here from `SensorParams` (key
/// strings unchanged, so pinned YAML and flash overrides carry over) —
/// `SensorParams` keeps sensor *identity* (LPF cutoffs, extrinsics,
/// which channels to fuse), this group owns filter *tuning*.
#[derive(Clone, Debug, Params)]
pub struct EskfFilterParams {
    /// Accelerometer noise density (m/s² per √Hz).
    #[param(key = "eskf_acc_noise", min = 1e-6, max = 10.0)]
    pub accel_noise_density: f32,
    /// Gyroscope noise density (rad/s per √Hz).
    #[param(key = "eskf_gyro_noise", min = 1e-8, max = 1.0)]
    pub gyro_noise_density: f32,
    /// Accelerometer bias random walk (m/s² per √s).
    #[param(key = "eskf_acc_bias_rw", min = 1e-9, max = 1.0)]
    pub accel_bias_random_walk: f32,
    /// Gyroscope bias random walk (rad/s per √s).
    #[param(key = "eskf_gyro_bias_rw", min = 1e-9, max = 1.0)]
    pub gyro_bias_random_walk: f32,
    /// Barometer altitude measurement 1-σ (m).
    #[param(key = "eskf_baro_std", unit = "m", min = 0.01, max = 100.0)]
    pub baro_noise_std: f32,
    /// Magnetometer field measurement 1-σ (body-frame µT equivalent).
    #[param(key = "eskf_mag_std", min = 1e-4, max = 10.0)]
    pub mag_noise_std: f32,
    /// Mahalanobis outlier gate, in sigma units per dimension: a
    /// measurement is rejected when `zᵀS⁻¹z / dof > gate_sigma²`.
    /// Lower = stricter rejection.
    #[param(key = "eskf_gate_sigma", min = 1.0, max = 100.0)]
    pub gate_sigma: f32,
    /// Absolute position-innovation reject threshold (m), independent of
    /// filter covariance. Catches source faults (mocap rigid-body
    /// re-association, RTK ambiguity loss, multipath) that the
    /// Mahalanobis gate would soft-accept under R-inflation.
    ///
    /// **Position-source coupled — pin it per vehicle.** ~1 m suits mocap
    /// (10 m/s × 100 ms). GPS at 5–10 Hz needs ~3 m, since the legitimate
    /// inter-frame residual at speed is ~1 m by itself and a 1 m gate
    /// would hard-reject healthy fast-flight frames.
    #[param(key = "eskf_max_pos_jump_m", unit = "m", min = 0.05, max = 100.0)]
    pub max_pos_jump_m: f32,
    /// Absolute attitude-innovation reject threshold (rad), independent
    /// of filter covariance. Catches mocap orientation flips (~180°) and
    /// rigid-body re-associations. 0.7 rad (~40°) sits above the worst
    /// plausible inter-frame attitude change and below the smallest
    /// dangerous flip (90° axis swap).
    #[param(key = "eskf_max_att_jump_rad", unit = "rad", min = 0.05, max = 3.15)]
    pub max_att_jump_rad: f32,
    /// Initial position variance [m²]. Set it to the bootstrap source's
    /// own accuracy squared — 1 m² badly over-states an RTK-fixed or
    /// mocap anchor and makes the filter snap to early measurements.
    #[param(key = "eskf_init_pos_var", min = 1e-6, max = 1e4)]
    pub init_pos_var: f32,
    /// Initial velocity variance [(m/s)²].
    #[param(key = "eskf_init_vel_var", min = 1e-6, max = 1e4)]
    pub init_vel_var: f32,
    /// Initial accelerometer-bias variance [(m/s²)²].
    #[param(key = "eskf_init_acc_bias_var", min = 1e-9, max = 100.0)]
    pub init_accel_bias_var: f32,
    /// Initial gyro-bias variance [(rad/s)²]. **Coupled to the guards'
    /// `eskf_*_bias_cov_thresh`**: the arming gate compares the sum of
    /// this diagonal against that threshold, so the pair jointly sets
    /// time-to-arm. The 0.01 default is ≈5.7 °/s 1-σ — pessimistic for a
    /// modern MEMS gyro, so lowering it shortens arming.
    #[param(key = "eskf_init_gyro_bias_var", min = 1e-9, max = 100.0)]
    pub init_gyro_bias_var: f32,
    /// Initial roll/pitch orientation variance [rad²]. Source-independent
    /// (both axes are observable from gravity at bootstrap).
    ///
    /// Yaw depends on the position source. **GPS** seeds yaw separately
    /// via the guard's `eskf_gps_init_yaw_cov`, because a GPS-only build
    /// cannot observe yaw at bootstrap. **Mocap** seeds yaw from *this*
    /// value: the mocap task calls `Eskf::init`, which applies one
    /// variance to all three attitude axes, and the bootstrap pose
    /// observes yaw directly — so on a mocap build this is the
    /// roll/pitch *and* yaw initial variance.
    #[param(key = "eskf_init_att_var_rp", unit = "rad^2", min = 1e-6, max = 100.0)]
    pub init_att_var_rp: f32,
    /// Hard cap on the measurement-noise inflation factor. Past it a
    /// residual is Rejected as a true outlier rather than absorbed, so
    /// the failsafe counts it.
    ///
    /// **Read with `eskf_gate_sigma`**: residuals up to
    /// `sqrt(cap) · gate_sigma` σ are still partially absorbed, so
    /// retuning the gate silently changes what this cap means. Raising
    /// it makes the filter swallow larger glitches; lowering it makes
    /// the guards trip sooner.
    #[param(key = "eskf_inflation_cap", min = 1.0, max = 1e6)]
    pub inflation_cap: f32,
    /// Magnetometer norm gate: reject a sample whose field magnitude
    /// differs from the world reference by more than this fraction.
    ///
    /// The filter's only magnetometer quality check. Tighten it at a
    /// site with local ferrous distortion; loosen it if a correctly
    /// calibrated mag is being rejected near the airframe's own wiring.
    #[param(key = "eskf_mag_norm_gate", min = 0.01, max = 2.0)]
    pub mag_norm_gate: f32,
    /// Longest IMU gap [s] the estimator will integrate across.
    ///
    /// A predict step whose `dt` exceeds this is skipped rather than
    /// propagated, so an IMU hiccup or a starved task cannot advance the
    /// nominal state on a huge extrapolation. Too small and a loaded
    /// system silently stops predicting, so raise it only with evidence
    /// from the loop-timing counters.
    #[param(key = "eskf_max_predict_dt_s", unit = "s", min = 0.002, max = 1.0)]
    pub max_predict_dt_s: f32,
}

impl Default for EskfFilterParams {
    fn default() -> Self {
        // Mirrors EskfConfig::default() — the parity test below guards
        // against drift.
        Self {
            accel_noise_density: 0.01,
            gyro_noise_density: 1.0e-4,
            accel_bias_random_walk: 1.0e-3,
            gyro_bias_random_walk: 1.0e-5,
            baro_noise_std: 0.5,
            mag_noise_std: 0.05,
            gate_sigma: 10.0,
            max_pos_jump_m: 1.0,
            max_att_jump_rad: 0.7,
            init_pos_var: 1.0,
            init_vel_var: 1.0,
            init_accel_bias_var: 0.01,
            init_gyro_bias_var: 0.01,
            init_att_var_rp: 0.1,
            inflation_cap: crate::eskf::DEFAULT_INFLATION_CAP,
            mag_norm_gate: crate::eskf::DEFAULT_MAG_NORM_GATE,
            max_predict_dt_s: 0.05,
        }
    }
}

impl EskfFilterParams {
    /// Project onto the core filter's own config struct.
    pub fn to_eskf_config(&self) -> EskfConfig {
        EskfConfig {
            accel_noise_density: self.accel_noise_density,
            gyro_noise_density: self.gyro_noise_density,
            accel_bias_random_walk: self.accel_bias_random_walk,
            gyro_bias_random_walk: self.gyro_bias_random_walk,
            baro_noise_std: self.baro_noise_std,
            mag_noise_std: self.mag_noise_std,
            gate_sigma: self.gate_sigma,
            max_pos_jump_m: self.max_pos_jump_m,
            max_att_jump_rad: self.max_att_jump_rad,
            init_pos_var: self.init_pos_var,
            init_vel_var: self.init_vel_var,
            init_accel_bias_var: self.init_accel_bias_var,
            init_gyro_bias_var: self.init_gyro_bias_var,
            init_att_var_rp: self.init_att_var_rp,
            inflation_cap: self.inflation_cap,
            mag_norm_gate: self.mag_norm_gate,
        }
    }
}

/// GPS failsafe-guard tuning — the full [`GpsGuardConfig`] surface plus
/// the dual-antenna heading σ floor.
///
/// Timeouts are seconds here (`f32`) rather than the guard struct's
/// milliseconds (`u64`): the param schema carries only f32/u8/u16/bool,
/// and seconds match the rest of the schema (`sampler_max_lag_s`).
/// [`to_gps_guard_config`](Self::to_gps_guard_config) converts.
///
/// The cascade limits are deliberately exposed. They decide when a bad
/// PVT stream stops the vehicle, so an operator should be able to read
/// and set them rather than discover them by reading firmware source —
/// but raising them widens the window in which a faulted GPS keeps
/// steering the filter. Treat them as safety-relevant, not as nuisance
/// suppression.
#[derive(Clone, Debug, Params)]
pub struct EskfGpsGuardParams {
    /// Consecutive jump-gated PVTs before the guard disarms (and, when
    /// the carrier solution permits, re-inits at the offending fix).
    #[param(key = "eskf_gps_max_jumps", min = 1.0, max = 100.0)]
    pub max_consecutive_jumps: u8,
    /// Consecutive filter-rejected PVTs before the guard disarms.
    #[param(key = "eskf_gps_max_rejects", min = 1.0, max = 200.0)]
    pub max_consecutive_rejects: u8,
    /// No accepted PVT for this long ⇒ stale: odometry publishing stops
    /// and the arming gate drops.
    #[param(key = "eskf_gps_stale_s", unit = "s", min = 0.1, max = 60.0)]
    pub stale_s: f32,
    /// Sustained `carr_soln ≥ 2` for this long raises `rtk_quality_ok`.
    #[param(key = "eskf_gps_rtk_fix_debounce_s", unit = "s", min = 0.0, max = 60.0)]
    pub rtk_fix_debounce_s: f32,
    /// Sustained `carr_soln < 2` for this long clears `rtk_quality_ok`.
    #[param(key = "eskf_gps_rtk_loss_debounce_s", unit = "s", min = 0.0, max = 60.0)]
    pub rtk_loss_debounce_s: f32,
    /// Gyro-bias covariance xy-trace below which the filter counts as
    /// converged (the arming gate). Scales with gyro grade.
    #[param(key = "eskf_gps_bias_cov_thresh", min = 1e-6, max = 1.0)]
    pub gyro_bias_cov_trace_xy_thresh: f32,
    /// Yaw covariance seeded at init / re-init. GPS-only cannot observe
    /// yaw, so this starts deliberately large.
    #[param(key = "eskf_gps_init_yaw_cov", min = 1e-3, max = 100.0)]
    pub init_yaw_cov: f32,
    /// Minimum satellite count for a PVT to be usable.
    #[param(key = "eskf_gps_min_sv", min = 0.0, max = 60.0)]
    pub min_sv: u8,
    /// Maximum receiver-reported horizontal accuracy for a usable PVT.
    #[param(key = "eskf_gps_h_acc_max_m", unit = "m", min = 0.01, max = 1000.0)]
    pub h_acc_max_m: f32,
    /// Position measurement σ floor at `carr_soln = 2` (RTK-fixed).
    #[param(key = "eskf_gps_pos_sigma_fix_m", unit = "m", min = 1e-3, max = 100.0)]
    pub pos_sigma_floor_fix_m: f32,
    /// Position measurement σ floor at `carr_soln = 1` (RTK-float).
    #[param(key = "eskf_gps_pos_sigma_float_m", unit = "m", min = 1e-3, max = 100.0)]
    pub pos_sigma_floor_float_m: f32,
    /// Position measurement σ floor at `carr_soln = 0` (stand-alone).
    #[param(key = "eskf_gps_pos_sigma_none_m", unit = "m", min = 1e-3, max = 100.0)]
    pub pos_sigma_floor_none_m: f32,
    /// Velocity measurement σ floor (applies when `gps_fuse_vel` is set).
    #[param(key = "eskf_gps_vel_sigma_m_s", min = 1e-3, max = 100.0)]
    pub vel_sigma_floor_m_s: f32,
    /// Minimum `carr_soln` for a jump-cascade re-init to fire. 2 =
    /// RTK-fixed only; re-seeding from a stand-alone fix injects
    /// metre-scale bias relative to the origin anchor. **Receivers
    /// without RTK must set this to 0**, otherwise the cascade can never
    /// re-init and the guard only ever disarms.
    #[param(key = "eskf_gps_reinit_min_carr_soln", min = 0.0, max = 2.0)]
    pub reinit_min_carr_soln: u8,
    /// σ floor for the dual-antenna heading/pitch measurement (rad).
    /// Applied to the receiver-reported heading σ before `update_baseline`.
    /// Only read on a `build: gps_dual_antenna: yes` + `gps_model: unicore` build.
    #[param(key = "eskf_gps_heading_sigma_floor_rad", unit = "rad", min = 1e-4, max = 1.0)]
    pub heading_sigma_floor_rad: f32,
}

impl Default for EskfGpsGuardParams {
    fn default() -> Self {
        // Mirrors GpsGuardConfig::default() (plus the former
        // `eskf_imu_gps::HEADING_SIGMA_FLOOR`) — parity-tested below.
        Self {
            max_consecutive_jumps: 5,
            max_consecutive_rejects: 15,
            stale_s: 2.0,
            rtk_fix_debounce_s: 2.0,
            rtk_loss_debounce_s: 1.0,
            gyro_bias_cov_trace_xy_thresh: 0.002,
            init_yaw_cov: 10.0,
            min_sv: 6,
            h_acc_max_m: 50.0,
            pos_sigma_floor_fix_m: 0.05,
            pos_sigma_floor_float_m: 0.30,
            pos_sigma_floor_none_m: 2.0,
            vel_sigma_floor_m_s: 0.10,
            reinit_min_carr_soln: 2,
            heading_sigma_floor_rad: 0.025,
        }
    }
}

impl EskfGpsGuardParams {
    /// Project onto the guard's own config struct. `fuse_velocity` comes
    /// from `SensorParams::gps_fuse_vel` (channel selection lives with
    /// the sensor group, not the tuning group).
    pub fn to_gps_guard_config(&self, fuse_velocity: bool) -> GpsGuardConfig {
        GpsGuardConfig {
            max_consecutive_jumps: self.max_consecutive_jumps as u32,
            max_consecutive_rejects: self.max_consecutive_rejects as u32,
            gps_stale_ms: (self.stale_s * 1000.0) as u64,
            rtk_fix_debounce_ms: (self.rtk_fix_debounce_s * 1000.0) as u64,
            rtk_loss_debounce_ms: (self.rtk_loss_debounce_s * 1000.0) as u64,
            gyro_bias_cov_trace_xy_thresh: self.gyro_bias_cov_trace_xy_thresh,
            init_yaw_cov: self.init_yaw_cov,
            gps_min_sv: self.min_sv,
            gps_h_acc_max_mm: (self.h_acc_max_m * 1000.0) as u32,
            pos_sigma_floor_fix_m: self.pos_sigma_floor_fix_m,
            pos_sigma_floor_float_m: self.pos_sigma_floor_float_m,
            pos_sigma_floor_none_m: self.pos_sigma_floor_none_m,
            vel_sigma_floor_m_s: self.vel_sigma_floor_m_s,
            fuse_velocity,
            reinit_min_carr_soln: self.reinit_min_carr_soln,
        }
    }
}

/// Mocap failsafe-guard tuning — the full [`MocapGuardConfig`] surface.
/// Same seconds-vs-milliseconds note as [`EskfGpsGuardParams`].
///
/// The σ values here are the mocap *measurement* noise: unlike GPS
/// (which derives σ per fix from the receiver's `h_acc`), mocap σ is
/// static, so these two numbers are the only statement of how much the
/// filter trusts the volume. They are genuinely install-specific —
/// camera count, volume size and marker spread all move them.
#[derive(Clone, Debug, Params)]
pub struct EskfMocapGuardParams {
    /// Consecutive jump-gated poses before the guard disarms. Mocap does
    /// **not** re-init on cascade (unlike GPS): a jump usually means
    /// rigid-body re-association, where IMU dead-reckoning is the more
    /// trustworthy source.
    #[param(key = "eskf_mocap_max_jumps", min = 1.0, max = 100.0)]
    pub max_consecutive_jumps: u8,
    /// Consecutive filter-rejected poses before the guard disarms.
    #[param(key = "eskf_mocap_max_rejects", min = 1.0, max = 200.0)]
    pub max_consecutive_rejects: u8,
    /// No accepted pose for this long ⇒ stale. Couple this to the mocap
    /// stream rate: the 0.1 s default assumes 100–360 Hz.
    #[param(key = "eskf_mocap_stale_s", unit = "s", min = 0.01, max = 10.0)]
    pub stale_s: f32,
    /// Gyro-bias covariance trace below which the filter counts as
    /// converged. Mocap observes all three attitude axes, so this is a
    /// 3-axis trace (GPS-only uses the xy trace).
    #[param(key = "eskf_mocap_bias_cov_thresh", min = 1e-6, max = 1.0)]
    pub gyro_bias_cov_trace_thresh: f32,
    /// Mocap position measurement 1-σ (m).
    #[param(key = "eskf_mocap_pos_std", unit = "m", min = 1e-4, max = 10.0)]
    pub pos_std: f32,
    /// Mocap attitude measurement 1-σ (rad).
    #[param(key = "eskf_mocap_att_std", unit = "rad", min = 1e-4, max = 3.15)]
    pub att_std: f32,
    /// Mutual-agreement radius for the **disarmed** re-anchor escape hatch
    /// (m). Once the jump cascade has wedged the filter, a disarmed
    /// airframe whose pose stream agrees with itself to within this radius
    /// over `eskf_mocap_reanchor_frames` re-seeds the ESKF; without it the cascade is
    /// unrecoverable short of a power cycle. Keep it well above `pos_std`
    /// (so a stationary airframe clears it) and well below
    /// `eskf_max_pos_jump_m` (so a re-associated rigid body cannot).
    /// Never applies in flight.
    #[param(key = "eskf_mocap_reanchor_m", unit = "m", min = 0.001, max = 1.0)]
    pub reanchor_radius_m: f32,
    /// Consecutive mutually-agreeing poses required before the disarmed
    /// re-anchor fires.
    ///
    /// The other half of the re-anchor policy, whose radius is
    /// `eskf_mocap_reanchor_m`; exposing only the radius left the frame
    /// count invisible. It is rate-coupled: at a 100–360 Hz mocap stream
    /// the default 5 frames is 14–50 ms of agreement.
    #[param(as_u16, key = "eskf_mocap_reanchor_frames", min = 1.0, max = 100.0)]
    pub reanchor_frames: usize,
}

impl Default for EskfMocapGuardParams {
    fn default() -> Self {
        // Mirrors MocapGuardConfig::default() — parity-tested below.
        Self {
            max_consecutive_jumps: 3,
            max_consecutive_rejects: 8,
            stale_s: 0.1,
            gyro_bias_cov_trace_thresh: 0.003,
            pos_std: 0.01,
            att_std: 0.03,
            reanchor_radius_m: 0.10,
            reanchor_frames: crate::eskf::DEFAULT_REANCHOR_FRAMES,
        }
    }
}

impl EskfMocapGuardParams {
    /// Project onto the guard's own config struct.
    pub fn to_mocap_guard_config(&self) -> MocapGuardConfig {
        MocapGuardConfig {
            max_consecutive_jumps: self.max_consecutive_jumps as u32,
            max_consecutive_rejects: self.max_consecutive_rejects as u32,
            mocap_stale_ms: (self.stale_s * 1000.0) as u64,
            gyro_bias_cov_trace_thresh: self.gyro_bias_cov_trace_thresh,
            mocap_pos_std: self.pos_std,
            mocap_att_std: self.att_std,
            reanchor_radius_m: self.reanchor_radius_m,
            reanchor_frames: self.reanchor_frames as u32,
        }
    }
}

/// Battery/pack facts consumed by the INDI inner loop's voltage handling —
/// hardware identity (4S vs 6S), previously hardcoded consts in
/// `indi_task.rs`. The staleness/failsafe *timeouts* stay compile-time
/// consts: they are protocol behavior, not hardware.
#[derive(Clone, Debug, Params)]
pub struct BatteryParams {
    /// Bootstrap pack voltage before the first POWER_STATUS frame arrives.
    /// Seeds the thrust-table linearization at boot; pick the mid-range of
    /// your pack (6S ≈ 23 V, 4S ≈ 15 V).
    #[param(key = "batt_nominal_v", unit = "V", min = 6.0, max = 36.0)]
    pub nominal_v: f32,
    /// Plausibility floor — voltage frames below this are dropped as ADC
    /// glitches.
    #[param(key = "batt_min_v", unit = "V", min = 5.0, max = 30.0)]
    pub min_plausible_v: f32,
    /// Plausibility ceiling — voltage frames above this are dropped as ADC
    /// glitches.
    #[param(key = "batt_max_v", unit = "V", min = 10.0, max = 60.0)]
    pub max_plausible_v: f32,
    /// Per-cell voltage used to auto-detect the pack's cell count at
    /// boot (`cells = floor(v / this) + 1`). Chemistry-coupled: 4.30 V
    /// suits LiPo/LiHV, a Li-ion pack needs a lower value or the count
    /// comes out short.
    #[param(key = "batt_cell_detect_v", unit = "V", min = 3.0, max = 5.0)]
    pub cell_detect_v: f32,
    /// Pack voltage below this reads as "no battery connected", which
    /// suppresses cell detection and the thrust-table voltage update.
    #[param(key = "batt_no_battery_v", unit = "V", min = 0.5, max = 10.0)]
    pub no_battery_v: f32,
    /// Cutoff of the single-pole low-pass on the pack-voltage ADC, in Hz.
    /// The raw conversion carries both ADC noise and the switching ripple
    /// the motors put on the rail; unfiltered, a 1 Hz telemetry sample is
    /// one arbitrary point out of that spread rather than a measurement.
    ///
    /// This filters the value *every* consumer sees, INDI's thrust-table
    /// linearization included, so it is a trade rather than a free win:
    /// lower = steadier reading but laggier sag tracking under a throttle
    /// punch. 2 Hz removes the noise while still following real sag inside
    /// ~80 ms. Drop toward 0.3 Hz for a Betaflight-like rock-steady display
    /// if the thrust model doesn't need the transient.
    #[param(key = "batt_lpf_hz", unit = "Hz", min = 0.1, max = 50.0)]
    pub lpf_hz: f32,
    /// Power-task ticks discarded before the cell count is detected.
    ///
    /// The divider network and its low-pass need to settle before the
    /// first reading means anything; counting cells from a settling
    /// voltage picks the wrong pack. At the task's 100 Hz tick the
    /// default 10 is ~100 ms.
    #[param(key = "batt_settle_ticks", min = 1.0, max = 1000.0)]
    pub settle_ticks: u16,
    /// Largest cell count the detector will report.
    ///
    /// A clamp, not a plausibility test: a mis-scaled divider reads as a
    /// valid pack at the ceiling rather than as an error, so raise it
    /// only for a pack that genuinely has more cells.
    #[param(key = "batt_max_cells", min = 1.0, max = 24.0)]
    pub max_cells: u8,
}

impl Default for BatteryParams {
    fn default() -> Self {
        // The previous hardcoded consts (6S bench pack, 12–30 V window).
        Self {
            nominal_v: 23.0,
            min_plausible_v: 12.0,
            max_plausible_v: 30.0,
            cell_detect_v: 4.30,
            no_battery_v: 3.00,
            lpf_hz: 2.0,
            settle_ticks: 10,
            max_cells: 8,
        }
    }
}

/// RPM-tracking notch banks on gyro/accel (`indi_task`), previously
/// function-local consts. Reboot-flagged as a group: `RpmNotchBank` bakes
/// q/min/fade into its biquads at construction (no reconfigure), so every
/// value is read once at task start. No hot-reload by design — which also
/// rules out the enable-edge stale-delay-line hazard a runtime toggle
/// would have.
#[derive(Clone, Debug, Params)]
pub struct RpmNotchParams {
    /// Enable the RPM-notch path. The banks are always allocated
    /// (~10 KB); this gates the per-tick tracking + filtering work.
    /// Bench-check CPU headroom before first enabling on a flight vehicle.
    #[param(key = "rpm_notch_en")]
    pub enable: bool,
    /// Notch Q: higher = narrower = less off-band phase loss but worse
    /// rejection if motor-frequency tracking is off.
    #[param(key = "rpm_notch_q", min = 1.0, max = 20.0)]
    pub q: f32,
    /// Below this motor frequency the notch fades to passthrough.
    #[param(key = "rpm_notch_min_hz", unit = "Hz", min = 20.0, max = 500.0)]
    pub min_hz: f32,
    /// Fade-in window above `min_hz`.
    #[param(key = "rpm_notch_fade_hz", unit = "Hz", min = 1.0, max = 200.0)]
    pub fade_hz: f32,
    /// PT1 cutoff for the notch-frequency tracker (must track motor 1P
    /// through throttle transients — deliberately much faster than the
    /// INDI sync filter).
    #[param(key = "rpm_notch_lpf_hz", unit = "Hz", min = 20.0, max = 500.0)]
    pub freq_lpf_hz: f32,
}

impl Default for RpmNotchParams {
    fn default() -> Self {
        // The previous hardcoded consts (Betaflight rpm_filter defaults);
        // enable=false preserves the shipped behavior bit-for-bit.
        Self {
            enable: false,
            q: 5.0,
            min_hz: 100.0,
            fade_hz: 50.0,
            freq_lpf_hz: 150.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Subsystem containers
// ---------------------------------------------------------------------------

/// Physical airframe identity: rigid body, motor geometry and the
/// identified actuator dynamics. The one group that genuinely describes
/// "the vehicle" — everything here would read the same under any
/// controller, which is the test for whether a key belongs in it.
///
/// Reboot-flagged **per field, not as a group**: the geometry (mass,
/// inertia, `m*_px`, `m*_spin`, …) is captured once when the INDI task
/// builds its effectiveness model, but the identified dynamics
/// (`m*_tau`, `m*_omega_max`, `m*_g2_*`, `m*_nonlin`) are re-applied by
/// the disarmed hot-reload. Flagging the container would mark all of them
/// "(reboot)" and send an operator rebooting between every step of a
/// bench G2 identification — see the leaf `reboot` attributes in
/// [`crate::mixer::MotorParams`].
#[derive(Clone, Debug, Params)]
pub struct AirframeParams {
    #[param(nested)]
    pub body: RigidBodyParams,
    #[param(nested_array, prefix = "m")]
    pub motors: [MotorParams; 4],
    /// Sensor extrinsics and calibrations for this physical install.
    /// Reboot-flagged: the estimation task and the board init read them
    /// once at start.
    #[param(nested, reboot)]
    pub install: InstallParams,
    /// Motor pole count, for the eRPM → RPM conversion on DShot
    /// telemetry. A motor nameplate fact (it is stamped on the bell), not
    /// a controller tunable — one value for the set, since mixing motor
    /// types on one airframe would already invalidate the shared G1.
    ///
    /// Reboot: the eRPM scale factor is folded into a constant at task
    /// start and into the `RpmTracker`'s conversion.
    #[param(key = "motor_poles", min = 2.0, max = 60.0, reboot)]
    pub motor_pole_count: u8,
}

/// State estimation: the ESKF core filter plus both source guards.
///
/// All three sub-groups are always present in the schema even though a
/// given build only runs one guard (`est_pos_gps` xor `est_pos_mocap`) —
/// groups are never `cfg`-gated out, so a blob written by one build
/// reads back on any other.
#[derive(Clone, Debug, Params)]
pub struct EskfParams {
    #[param(nested)]
    pub filter: EskfFilterParams,
    #[param(nested)]
    pub gps_guard: EskfGpsGuardParams,
    #[param(nested)]
    pub mocap_guard: EskfMocapGuardParams,
    #[param(nested)]
    pub faults: EskfFaultParams,
}

impl Default for EskfParams {
    fn default() -> Self {
        Self {
            filter: EskfFilterParams::default(),
            gps_guard: EskfGpsGuardParams::default(),
            mocap_guard: EskfMocapGuardParams::default(),
            faults: EskfFaultParams::default(),
        }
    }
}

/// Per-motor RPM estimator (FOPDT Kalman filter) tuning.
///
/// These were compile-time constants in `RpmEstimatorConfigBuilder`, which
/// meant the one part of the RPM pipeline that most needs bench tuning was
/// the one part that needed a reflash — while `m*_tau` and `m*_omega_max`,
/// which are physical constants, were fully loadable. Motor dynamics still
/// come from the motor params; this group is the *filter's* own tuning.
///
/// Not reboot-flagged: the disarmed hot-reload rebuilds the estimator
/// config in place, preserving the state estimate, so these can be swept
/// on the bench between arms.
#[derive(Clone, Debug, Params)]
pub struct RpmEstimatorParams {
    /// Measurement-noise variance on ω at/below 2000 rad/s [(rad/s)²].
    ///
    /// This ESC firmware transmits a *single unaveraged* 60° electrical
    /// step rather than the 6-step average upstream AM32 sends, so the
    /// per-sample noise is dominated by commutation asymmetry (magnet
    /// placement, per-phase comparator offset) rather than by timer
    /// quantization — and it scales with ω. The estimator therefore
    /// applies `R(y) = this · max(1, (y/2000)²)`: set the variance
    /// measured *at ω ≈ 2000* and the scaling handles the rest. SAKURAH743
    /// flight logs measure σ ≈ 8.7 % of ω (≈ 177 rad/s at 2000 → 3.1e4).
    /// Too small and the NIS gate rejects good samples — sustained
    /// rejection makes the filter coast open-loop on its model (the
    /// pre-2026-09 divergence); too large and outliers get fused.
    #[param(key = "rpm_est_omega_var", unit = "(rad/s)^2", min = 0.01, max = 1e6)]
    pub omega_noise_cov: f32,
    /// Intensity of the throttle-error process driving ω [s].
    ///
    /// A spectral density, not a per-step variance: it is multiplied by
    /// `dt·(c_m/τ)²`, so it means the same thing at 8 kHz and 1 kHz. Raise
    /// it to make the filter trust its motor model less and telemetry more.
    #[param(key = "rpm_est_thr_psd", unit = "s", min = 1e-9, max = 1.0)]
    pub throttle_noise_cov: f32,
    /// Random-walk intensity on the estimated full-throttle speed `c_m`
    /// [(rad/s)²/s]. `c_m` is a filter state — it is the throttle→ω gain,
    /// which tracks pack voltage — so this sets how fast it may follow
    /// battery sag.
    ///
    /// Simulated against a 6S pack sagging 25.2 → 21.0 V over a 3-minute
    /// flight (a ~17 % drift in `c_m`), the residual error on `c_m` is:
    /// 0.1 → 48 rad/s of standing lag, 1 → 15, 10 → 4.7, **100 → 2.5**,
    /// 1000 → 4.0. The optimum is broad and holds across 1 %, 3 % and 5 %
    /// measurement-noise assumptions, and raising it does not loosen the
    /// NIS gate (converged P₀₀ moved 93.6 → 92.9), so the two knobs are
    /// independent.
    #[param(key = "rpm_est_cm_psd", min = 1e-6, max = 1e4)]
    pub c_m_noise_cov: f32,
    /// Normalized-innovation-squared gate (χ², 1 DOF). Samples with
    /// `innov² > gate·S` are rejected and the filter coasts. 3.84 ≈ 95th
    /// percentile, 9.0 ≈ 3σ. Count `nis_reject` in `DshotMotorHealth`
    /// after changing it — a gate that rejects a large fraction of frames
    /// is mis-tuned, not protective.
    #[param(key = "rpm_est_nis_gate", min = 1.0, max = 100.0)]
    pub nis_gate: f32,
    /// Transport delay from commanding a throttle to it affecting ω [s].
    /// The estimator keeps a throttle history specifically to model it.
    /// 0 disables the lookup and uses the newest command.
    #[param(key = "rpm_est_tau_d", unit = "s", min = 0.0, max = 0.02)]
    pub tau_d: f32,
    /// Reference speed [rad/s] at which `rpm_est_omega_var` is stated.
    ///
    /// The measurement variance scales as
    /// `R(y) = omega_var · max(1, (y/this)²)`, so this is what makes
    /// that number mean anything. **Change the two together**: moving
    /// this alone rescales the NIS gate at every speed.
    #[param(key = "rpm_est_omega_ref", unit = "rad/s", min = 100.0, max = 20000.0)]
    pub omega_noise_ref: f32,
    /// Initial and re-seed variance on the ω state [(rad/s)²].
    ///
    /// Deliberately wide: it says the filter knows nothing about motor
    /// speed at construction, after a reconfigure, and after the
    /// rejection escape below, so the first good measurement dominates.
    #[param(key = "rpm_est_init_omega_var", unit = "(rad/s)^2", min = 1.0, max = 1e6)]
    pub init_omega_var: f32,
    /// Consecutive NIS rejections before ω is re-seeded from the
    /// measurement.
    ///
    /// The escape from a rejection spiral: once the estimate has drifted
    /// far enough that every sample gates out, the filter would coast
    /// open-loop on its model forever without it. Rate-coupled like
    /// `indi_rpm_stale_gaps` — the default 25 is 25–50 ms at 500 Hz–1 kHz.
    #[param(as_u16, key = "rpm_est_escape_rejects", min = 1.0, max = 1000.0)]
    pub escape_consecutive_rejects: usize,
    /// Hard plausibility bound on a measurement, as a multiple of the
    /// estimated full-throttle speed `c_m`.
    ///
    /// Above it the sample is a decode error, not a fast motor, and is
    /// discarded before it can reach the gate or the escape counter.
    #[param(key = "rpm_est_plausible_frac", min = 1.0, max = 10.0)]
    pub plausible_omega_frac: f32,
}

impl Default for RpmEstimatorParams {
    fn default() -> Self {
        Self {
            // σ = 20 rad/s ≡ ~1 % commutation asymmetry at ω ≈ 2000 rad/s.
            // Deliberately sized at high thrust rather than hover: R is a
            // constant while the true noise scales with ω, and the costly
            // direction is a too-tight gate during aggressive maneuvers.
            // Provisional until measured — see the doc comment.
            omega_noise_cov: 400.0,
            // 1.25e-6 s ≡ the historical per-step variance of 0.01 at 8 kHz,
            // and simulated-optimal at a ~25 % `m*_tau` error. Lower it
            // toward 1.25e-8 once the motor time constants are identified.
            throttle_noise_cov: 1.25e-6,
            c_m_noise_cov: 100.0,
            nis_gate: 9.0,
            tau_d: 0.0,
            omega_noise_ref: 2000.0,
            init_omega_var: 1000.0,
            escape_consecutive_rejects: 25,
            plausible_omega_frac: 1.5,
        }
    }
}

/// INDI inner loop: effectiveness + controller tuning. The always-hot group.
#[derive(Clone, Debug, Params)]
pub struct IndiParams {
    #[param(nested)]
    pub effectiveness: IndiEffectivenessParams,
    #[param(nested, reboot)]
    pub controller: IndiControllerParams,
    #[param(nested)]
    pub rpm_estimator: RpmEstimatorParams,
}

/// Trajectory generation and tracking: planner, reference sampler, and the
/// persisted offline-mission selection.
#[derive(Clone, Debug, Params)]
pub struct TrajectoryParams {
    #[param(nested)]
    pub planner: PlannerParams,
    #[param(nested)]
    pub sampler: SamplerParams,
    /// Index into `cybflight::control::offline_mission::PROFILES` selecting
    /// which prebaked offline trajectory to fly. Mirrored into
    /// `offline_mission::ACTIVE_PROFILE` at boot; the shell verb
    /// `mission set <env> <variant> <speed>` writes both.
    pub mission_profile: u8,
}

/// Mahony complementary-filter tuning.
///
/// The Mahony filter is the independent attitude reference. Its task is
/// spawned unconditionally: it feeds the blackbox `/attitude` topic on
/// every build, and on a build with no ESKF it is the sole
/// `VEHICLE_ATTITUDE` publisher. Its whole tuning surface used to be
/// constructor literals with no way to reach them.
///
/// It is *designed* to also be the cross-check the arming gate compares
/// the ESKF against (`arm_eskf_mahony_tol_deg`), but on an ESKF build
/// that gate is currently dead: `health.rs` reports a `mahony_ready`
/// sentinel and no attitude, so both `MahonyNotReady` and
/// `EskfMahonyTiltDisagreement` are unreachable. Tuning here therefore
/// moves logging today, and the arming gate only once that is wired.
///
/// Reboot-flagged: the filter is built once at task start.
#[derive(Clone, Debug, Params)]
pub struct MahonyParams {
    /// Proportional gain on the accel/mag correction, all axes.
    ///
    /// How hard the filter pulls the attitude estimate toward the
    /// gravity (and field) reference. Higher tracks a drifting gyro
    /// faster but lets linear acceleration tilt the estimate.
    #[param(key = "mahony_kp", min = 0.0, max = 20.0)]
    pub kp: f32,
    /// Integral gain estimating gyro bias, all axes. Raise it for a
    /// sensor with visible bias drift; too high and it absorbs a
    /// sustained real rotation as bias.
    #[param(key = "mahony_ki", min = 0.0, max = 10.0)]
    pub ki: f32,
    /// Accelerometer norm floor as a fraction of 1 g. Below it the
    /// sample carries no usable gravity direction and the correction is
    /// skipped rather than applied to free-fall noise.
    #[param(key = "mahony_min_accel_g", unit = "g", min = 0.01, max = 1.0)]
    pub min_accel_g: f32,
    /// Magnetometer norm floor [µT]. Below it the filter degrades from
    /// MARG to IMU-only instead of taking a heading from noise.
    #[param(key = "mahony_min_mag_ut", unit = "uT", min = 0.1, max = 100.0)]
    pub min_mag_ut: f32,
}

impl Default for MahonyParams {
    fn default() -> Self {
        // The previous `Mahony::new()` literals: kp = 1, ki = 0.3, and
        // the two norm floors, which were written as 10 % of a nominal
        // 1 g and of a nominal 50 uT Earth field.
        Self {
            kp: 1.0,
            ki: 0.3,
            min_accel_g: 0.1,
            min_mag_ut: 5.0,
        }
    }
}

/// Device/UX toggles — system settings, not vehicle physics.
#[derive(Clone, Debug, Params)]
pub struct SystemSettings {
    /// External arm-LED enable. When true, the LED task drives the WS2812
    /// strip (dimmed disarmed, full-bright armed); when false the strip is
    /// held off. Toggled by `led on`/`led off` (auto-saves).
    #[param(key = "arm_led")]
    pub arm_led_enabled: bool,
    /// Blackbox record-set tier: 0=None, 1=Small, 2=Mid, 3=Large,
    /// 4=Sysid (Large topics + INDI telemetry mirrors at ≥500 Hz).
    /// Mirrored into the `BLACKBOX_RECORD_SET` atomic at boot; updated by
    /// `blackbox set <tier>`. Out-of-range values fall back to
    /// `RecordSet::DEFAULT` at use site.
    #[param(key = "blackbox_tier", min = 0.0, max = 4.0)]
    pub blackbox_record_set: u8,
    /// Blackbox rate divider for `/imu1_raw`: one sample in N is
    /// published, the rest are never produced. 1 = full rate.
    ///
    /// Applies to `/imu1_raw` **only**, and takes effect at the
    /// publisher rather than in the recorder. That placement is the
    /// whole point: thinning behind the PubSub costs what it saves,
    /// because a discarded sample has already spent a channel slot and
    /// a drain-budget iteration, so the records the recorder emits per
    /// pass fall in step with the bytes. `/imu1_raw` can be divided at
    /// the source because the recorder is its only subscriber;
    /// `/imu1` and `/odometry` feed the inner loop, the ESKF and the
    /// outer loop, so they are left at full rate and `blackbox_tier`
    /// is the lever for their bandwidth.
    ///
    /// Mirrored into the `BLACKBOX_RATE_DIV` atomic on param apply.
    /// The shell refuses `param set` while a session is recording, so
    /// the publisher's divider and the `rate_div` the recorder stamps
    /// into the file's metadata cannot disagree within one flight.
    #[param(key = "blackbox_rate_div", min = 1.0, max = 64.0)]
    pub blackbox_rate_div: u8,
    /// Bitmask of muted blackbox topics, keyed by MCAP channel id
    /// (bit N = channel id N; docs/blackbox.md has the id table). A
    /// muted topic is dropped from the session entirely — no
    /// Schema/Channel records, no subscription. `/events` (id 4)
    /// cannot be muted; the recorder ignores its bit. 0 = nothing
    /// muted. Snapshotted at session start.
    ///
    /// The ceiling is `2^(MAX_CHANNEL_ID+1) − 1`: every channel the
    /// recorder can emit must be addressable, or the topic silently
    /// cannot be muted at all. It widened past 16 bits when `/mpc_cost`
    /// (id 16) landed, which is exactly the failure this bound is meant
    /// to prevent — raise it in step with the last channel id, and keep
    /// it under 2^24 so the `f32` the param plane carries stays exact.
    #[param(key = "blackbox_mute_mask", min = 0.0, max = 131_071.0)]
    pub blackbox_mute_mask: u32,
}

/// The firmware's full persisted configuration, grouped by subsystem.
///
/// This is the single container behind `params::get()`, the shell `param`
/// namespace, and the flash blob. Groups are never `cfg`-gated out of the
/// schema — a group may be dead in a given build (e.g. `cascade` under
/// `outer_mpc`) but never absent, so a blob written by one feature build
/// reads back on any other.
#[derive(Clone, Debug, Params)]
pub struct FirmwareConfig {
    #[param(nested)]
    pub airframe: AirframeParams,
    #[param(nested, reboot)]
    pub sensors: SensorParams,
    /// Reboot-flagged: the ESKF and its guard are constructed once at
    /// estimation-task start, so a live edit only takes effect on the
    /// next boot.
    #[param(nested, reboot)]
    pub eskf: EskfParams,
    /// Reboot-flagged for the same reason as `eskf`: the Mahony filter
    /// is constructed once at task start.
    #[param(nested, reboot)]
    pub mahony: MahonyParams,
    #[param(nested, reboot)]
    pub battery: BatteryParams,
    /// Transmitter / stick configuration. Reboot-flagged: the RC task
    /// reads its calibration once at start.
    #[param(nested, reboot)]
    pub rc: RcParams,
    /// Where you fly: gravity and the stick-integrator envelope.
    #[param(nested, reboot)]
    pub site: SiteParams,
    /// Arming preconditions and RC-loss failsafe timing.
    #[param(nested, reboot)]
    pub safety: SafetyParams,
    #[param(nested)]
    pub indi: IndiParams,
    #[param(nested, reboot)]
    pub rpm_notch: RpmNotchParams,
    #[param(nested, reboot)]
    pub cascade: CascadeParams,
    #[param(nested)]
    pub mpc: MpcParams,
    #[param(nested)]
    pub trajectory: TrajectoryParams,
    #[param(nested)]
    pub system: SystemSettings,
}

impl FirmwareConfig {
    /// Construction scaffold for the baked-parameter overlay: every
    /// tunable group at its `Default`, the airframe **geometry zeroed** (a
    /// zero mass is deliberately unflyable — identity values carry no
    /// default and must come from the vehicle YAML, which the firmware
    /// `build.rs` enforces at compile time).
    ///
    /// The motors' *identified dynamics* are the exception and keep their
    /// real defaults, because the two kinds of field fail differently: a
    /// wrong mass or motor position flies and looks plausible, so silence
    /// is the danger and a zero must stop the build; a zero `tau` or
    /// `omega_max` is structurally meaningless (the G2 scaler divides by
    /// tau) and would only force every vehicle to restate a number that
    /// is the same on all of them. `nonlin` keeps its documented zero
    /// sentinel ("use the thrust-model-matched fallback").
    ///
    /// Not intended as a usable configuration; exists so the baked
    /// snapshot (which covers every parameter, airframe included) has
    /// something to overlay onto.
    #[doc(hidden)]
    pub fn scaffold() -> Self {
        let zero_motor = MotorParams::STOCK_DYNAMICS;
        Self {
            airframe: AirframeParams {
                body: RigidBodyParams {
                    mass_kg: 0.0,
                    inertia_kg_m2: [0.0; 9],
                    max_rate_rad_s: [0.0; 3],
                },
                motors: [zero_motor; 4],
                install: InstallParams::default(),
                motor_pole_count: 14,
            },
            sensors: SensorParams::default(),
            eskf: EskfParams::default(),
            mahony: MahonyParams::default(),
            battery: BatteryParams::default(),
            rc: RcParams::default(),
            site: SiteParams::default(),
            safety: SafetyParams::default(),
            indi: IndiParams {
                effectiveness: IndiEffectivenessParams::default(),
                controller: IndiControllerParams::default(),
                rpm_estimator: RpmEstimatorParams::default(),
            },
            rpm_notch: RpmNotchParams::default(),
            cascade: CascadeParams {
                pos_kp: [0.0; 3],
                pos_kd: [0.0; 3],
                att_k_rate: [0.0; 3],
                att_k_torque: [0.0; 3],
                // Unlike the gains (zero = "untuned, no authority"), a zero
                // rate would be a structural brick (divide-by-zero ticker),
                // so the scaffold carries the real default. The error
                // clamps and the staleness window are structural for the
                // same reason.
                rate_hz: 100,
                pos_err_max: [1.0; 3],
                vel_err_max: [1.0; 3],
                odom_stale_s: 0.1,
            },
            mpc: MpcParams::default(),
            trajectory: TrajectoryParams {
                planner: PlannerParams::default(),
                sampler: SamplerParams::default(),
                mission_profile: 0,
            },
            system: SystemSettings {
                arm_led_enabled: true,
                blackbox_record_set: 3,
                blackbox_rate_div: 1,
                blackbox_mute_mask: 0,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Name-addressed access (shell surface)
// ---------------------------------------------------------------------------

impl FirmwareConfig {
    /// Read a parameter by full name (e.g. `"mass"`, `"m0_px"`).
    pub fn get_named(&self, name: &str) -> Option<crate::param_registry::ParamValue> {
        self.param_get(<Self as ParamGroup>::param_find(name)?)
    }

    /// Write a parameter by full name from an f32 (shell/text source).
    /// Returns `false` for unknown names.
    pub fn set_named(&mut self, name: &str, v: f32) -> bool {
        match <Self as ParamGroup>::param_find(name) {
            Some(idx) => self.param_set_f32(idx, v),
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Flash blob
// ---------------------------------------------------------------------------

impl FirmwareConfig {
    /// Serialize to a flash-ready buffer with header and CRC.
    pub fn to_bytes(&self) -> [u8; PADDED_SIZE] {
        let mut buf = [0u8; PADDED_SIZE];
        let mut off = HEADER_SIZE;
        for idx in 0..PARAM_COUNT {
            // idx < COUNT by construction — get always succeeds.
            let v = self.param_get(idx).map_or(0.0, |v| v.as_f32());
            off = put_f32(&mut buf, off, v);
        }
        debug_assert_eq!(off - HEADER_SIZE, PAYLOAD_SIZE);

        let payload = &buf[HEADER_SIZE..HEADER_SIZE + PAYLOAD_SIZE];
        let crc = crc32fast::hash(payload);
        put_u32(&mut buf, 0, MAGIC);
        put_u32(&mut buf, 4, VERSION);
        put_u32(&mut buf, 8, PAYLOAD_SIZE as u32);
        put_u32(&mut buf, 12, crc);
        buf
    }

    /// Overlay a flash image onto `self` (typically the compile-time
    /// defaults). Returns `false` — leaving `self` untouched — if magic,
    /// version, length, or CRC mismatch.
    pub fn apply_from_bytes(&mut self, buf: &[u8; PADDED_SIZE]) -> bool {
        let magic = get_u32(buf, 0);
        let version = get_u32(buf, 4);
        let length = get_u32(buf, 8);
        let stored_crc = get_u32(buf, 12);

        if magic != MAGIC || version != VERSION || length as usize != PAYLOAD_SIZE {
            return false;
        }
        let payload = &buf[HEADER_SIZE..HEADER_SIZE + PAYLOAD_SIZE];
        if crc32fast::hash(payload) != stored_crc {
            return false;
        }

        for idx in 0..PARAM_COUNT {
            self.param_set_f32(idx, get_f32(buf, HEADER_SIZE + 4 * idx));
        }
        true
    }
}

fn put_f32(buf: &mut [u8], off: usize, v: f32) -> usize {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    off + 4
}

fn put_u32(buf: &mut [u8], off: usize, v: u32) -> usize {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    off + 4
}

fn get_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn get_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Shared host-test fixture (also used by `param_store` tests).
#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;
    use crate::mixer::SpinDir;

    pub(crate) fn test_config() -> FirmwareConfig {
        FirmwareConfig {
            // non-default so round-trip catches drops
            mahony: MahonyParams {
                kp: 1.5,
                ki: 0.25,
                min_accel_g: 0.15,
                min_mag_ut: 6.0,
            },
            airframe: AirframeParams {
                body: RigidBodyParams {
                    mass_kg: 0.55,
                    inertia_kg_m2: [0.0025, 0.0, 0.0, 0.0, 0.0021, 0.0, 0.0, 0.0, 0.0043],
                    max_rate_rad_s: [10.0, 10.0, 6.0],
                },
                motors: [
                    MotorParams {
                        position_m: [-0.075, -0.1],
                        spin_dir: SpinDir::Cw,
                        max_thrust_n: 8.5,
                        torque_coeff_m: 0.022,
                        ..MotorParams::STOCK_DYNAMICS
                    },
                    MotorParams {
                        position_m: [0.075, -0.1],
                        spin_dir: SpinDir::Ccw,
                        max_thrust_n: 8.5,
                        torque_coeff_m: 0.022,
                        ..MotorParams::STOCK_DYNAMICS
                    },
                    MotorParams {
                        position_m: [-0.075, 0.1],
                        spin_dir: SpinDir::Ccw,
                        max_thrust_n: 8.5,
                        torque_coeff_m: 0.022,
                        ..MotorParams::STOCK_DYNAMICS
                    },
                    MotorParams {
                        position_m: [0.075, 0.1],
                        spin_dir: SpinDir::Cw,
                        max_thrust_n: 8.5,
                        torque_coeff_m: 0.022,
                        ..MotorParams::STOCK_DYNAMICS
                    },
                ],
                install: InstallParams::default(),
                motor_pole_count: 14,
            },
            sensors: SensorParams::default(),
            // Every field non-default so the round-trip test catches a
            // dropped or mis-keyed ESKF parameter.
            eskf: EskfParams {
                filter: EskfFilterParams {
                    accel_noise_density: 0.02,
                    gyro_noise_density: 2.0e-4,
                    accel_bias_random_walk: 2.0e-3,
                    gyro_bias_random_walk: 2.0e-5,
                    baro_noise_std: 0.6,
                    mag_noise_std: 0.06,
                    gate_sigma: 12.0,
                    max_pos_jump_m: 3.0,
                    max_att_jump_rad: 0.8,
                    init_pos_var: 2.0,
                    init_vel_var: 1.5,
                    init_accel_bias_var: 0.02,
                    init_gyro_bias_var: 0.015,
                    init_att_var_rp: 0.12,
                    inflation_cap: 120.0,
                    mag_norm_gate: 0.35,
                    max_predict_dt_s: 0.06,
                },
                gps_guard: EskfGpsGuardParams {
                    max_consecutive_jumps: 4,
                    max_consecutive_rejects: 12,
                    stale_s: 2.5,
                    rtk_fix_debounce_s: 1.5,
                    rtk_loss_debounce_s: 1.25,
                    gyro_bias_cov_trace_xy_thresh: 0.0025,
                    init_yaw_cov: 9.0,
                    min_sv: 7,
                    h_acc_max_m: 40.0,
                    pos_sigma_floor_fix_m: 0.06,
                    pos_sigma_floor_float_m: 0.35,
                    pos_sigma_floor_none_m: 2.5,
                    vel_sigma_floor_m_s: 0.12,
                    reinit_min_carr_soln: 1,
                    heading_sigma_floor_rad: 0.03,
                },
                mocap_guard: EskfMocapGuardParams {
                    max_consecutive_jumps: 4,
                    max_consecutive_rejects: 9,
                    stale_s: 0.15,
                    gyro_bias_cov_trace_thresh: 0.0035,
                    pos_std: 0.012,
                    att_std: 0.035,
                    reanchor_radius_m: 0.12,
                    reanchor_frames: 6,
                },
                faults: EskfFaultParams {
                    pos_timeout_s: 2.5,
                    cov_blowup_m2: 30.0,
                    nan_reset_hold_s: 3.5,
                    cascade_hold_s: 4.0,
                },
            },
            battery: BatteryParams {
                nominal_v: 15.0, // non-default so round-trip catches drops
                min_plausible_v: 12.0,
                max_plausible_v: 30.0,
                cell_detect_v: 4.2,
                no_battery_v: 2.5,
                lpf_hz: 5.0,
                settle_ticks: 12,
                max_cells: 6,
            },
            rc: RcParams {
                min_us: 1000,
                mid_us: 1495,
                max_us: 2000,
                arm_channel: 6,
                arm_threshold_us: 1600,
                throttle_mincheck_us: 1040,
                mission_channel: 7,
                mission_high_us: 1750,
                mission_low_us: 1250,
                throttle_land_us: 1120,
                launch_us: 1650,
                launch_confirm_frames: 6,
                xy_deadband: 0.06,
                throttle_deadband: 0.11,
                rate_deadband: 0.04,
                xy_rate_m_s: 1.2,
                z_rate_m_s: 0.6,
                land_rate_m_s: 0.45,
                land_lead_m: 1.15,
                max_rate_rp_rad_s: 7.5,
                max_rate_yaw_rad_s: 3.5,
            },
            site: SiteParams {
                gravity_m_s2: 9.807,
                fence_enable: true,
                fence_x_m: 3.0,
                fence_y_m: 4.0,
                fence_z_max_m: 2.5,
                fence_z_min_m: 0.1,
            },
            safety: SafetyParams {
                arm_max_tilt_deg: 25.0,
                arm_min_link_quality: 60,
                arm_eskf_mahony_tol_deg: 12.0,
                arm_accel_tol_m_s2: 1.2,
                arm_gyro_limit_rad_s: 0.4,
                arm_switch_hold_s: 0.15,
                arm_link_stats_max_age_s: 0.6,
                fs_rxloss_trigger_s: 0.2,
                fs_guard_period_s: 1.75,
                fs_recovery_period_s: 0.6,
                fs_ctrl_timeout_s: 0.45,
            },
            indi: IndiParams {
                effectiveness: IndiEffectivenessParams::default(),
                controller: IndiControllerParams::default(),
                rpm_estimator: RpmEstimatorParams {
                    // non-default so round-trip catches drops
                    omega_noise_cov: 250.0,
                    throttle_noise_cov: 2.5e-6,
                    c_m_noise_cov: 0.25,
                    nis_gate: 6.0,
                    tau_d: 0.002,
                    omega_noise_ref: 2500.0,
                    init_omega_var: 900.0,
                    escape_consecutive_rejects: 30,
                    plausible_omega_frac: 1.75,
                },
            },
            rpm_notch: RpmNotchParams {
                enable: true, // non-default so round-trip catches drops
                q: 4.0,
                min_hz: 120.0,
                fade_hz: 40.0,
                freq_lpf_hz: 140.0,
            },
            cascade: CascadeParams {
                pos_kp: [4.0, 4.0, 5.0],
                pos_kd: [4.0, 4.0, 4.0],
                att_k_rate: [3.0, 3.0, 1.0],
                att_k_torque: [1.0, 1.0, 0.2],
                rate_hz: 100,
                pos_err_max: [1.0; 3],
                vel_err_max: [1.0; 3],
                odom_stale_s: 0.1,
            },
            mpc: MpcParams::default(),
            trajectory: TrajectoryParams {
                planner: PlannerParams::default(),
                sampler: SamplerParams::default(),
                mission_profile: 3,
            },
            system: SystemSettings {
                arm_led_enabled: false,
                blackbox_record_set: 2, // Mid — non-default so round-trip catches drops
                blackbox_rate_div: 8,   // non-default so round-trip catches drops
                blackbox_mute_mask: 0x1_2002, // non-default (incl. a >16-bit id) so round-trip catches drops
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::test_config;
    use super::*;
    use crate::mixer::SpinDir;
    use crate::param_registry::{ParamName, ParamValue};

    /// Every parameter has a unique name, and every name resolves back to
    /// its own index. This is the cross-group collision check the derive
    /// macro cannot do per-struct.
    #[test]
    fn names_unique_and_roundtrip() {
        let mut seen = std::collections::HashSet::new();
        for idx in 0..PARAM_COUNT {
            let name = ParamName::of::<FirmwareConfig>(idx).expect("name");
            let name = name.as_str().to_string();
            assert!(seen.insert(name.clone()), "duplicate param name: {name}");
            assert_eq!(
                <FirmwareConfig as ParamGroup>::param_find(&name),
                Some(idx),
                "find({name})"
            );
        }
    }

    /// Well-known key names resolve. Mostly a legacy check (these names
    /// survived the registry migration); the four `m*_tau` / `m*_omega_max`
    /// / `m*_nonlin` / `m*_g2_*` entries are the post-rename spellings of
    /// the actuator facts that moved from `indi.effectiveness` into
    /// `airframe.motors` — see [`retired_names_absent`].
    #[test]
    fn legacy_names_present() {
        for name in [
            "mass",
            "ixx",
            "izz",
            "m0_px",
            "m3_torque",
            "m2_spin",
            "pos_kp_x",
            "pos_kd_z",
            "g1_fx_m0",
            "m3_g2_ry",
            "m2_omega_max",
            "m1_tau",
            "m0_nonlin",
            "indi_rate_r",
            "indi_sync_hz",
            "wls_wv_fz",
            "wls_wu_m2",
            "motor_poles",
            "plan_max_vel",
            "plan_w_body_rate",
            "plan_bfgs_max_iter",
            "mpc_w_pos_x",
            "mpc_pos_cost_mode",
            "sampler_max_lag_s",
            "sampler_max_search_steps",
        ] {
            assert!(
                <FirmwareConfig as ParamGroup>::param_find(name).is_some(),
                "legacy name missing: {name}"
            );
        }
        // New in v31: previously persisted-but-unkeyed fields.
        for name in [
            "att_k_r",
            "max_rate_r",
            "mission_profile",
            "arm_led",
            "blackbox_tier",
            // v48–v49
            "blackbox_rate_div",
            "blackbox_mute_mask",
            "indi_ctrl_div",
        ] {
            assert!(
                <FirmwareConfig as ParamGroup>::param_find(name).is_some(),
                "new name missing: {name}"
            );
        }
        // New in v33+: loop rates and promoted hardware consts.
        for name in [
            "mpc_rate_hz",
            "cascade_rate_hz",
            "batt_nominal_v",
            "batt_min_v",
            "batt_max_v",
            "rpm_notch_en",
            "rpm_notch_q",
            "rpm_notch_lpf_hz",
            "sampler_kind",
            "gps_fuse_vel",
            // v51
            "mpc_drag_x",
            "mpc_bodydrag_z",
        ] {
            assert!(
                <FirmwareConfig as ParamGroup>::param_find(name).is_some(),
                "new name missing: {name}"
            );
        }
    }

    /// `site.gravity_m_s2` must reach every model built from params.
    ///
    /// Each of these constructors has the whole `FirmwareConfig` in hand
    /// and used to write `9.81` anyway, so the param validated, persisted
    /// and displayed while the MPC, the full model and the planner all
    /// flew standard gravity. A non-default value is used here precisely
    /// because the old hardcode is the default: asserting against 9.81
    /// would have passed against the bug.
    #[test]
    fn site_gravity_reaches_every_model() {
        let mut cfg = test_config();
        cfg.site.gravity_m_s2 = 9.78; // Singapore-ish, inside [9.6, 10.0]

        let simple = crate::mpc::QuadModel::from_vehicle_params(&cfg);
        assert_eq!(simple.grav, 9.78, "QuadModel ignored site.gravity_m_s2");

        let full = crate::mpc::FullQuadModel::from_vehicle_params(&cfg);
        assert_eq!(full.grav, 9.78, "FullQuadModel ignored site.gravity_m_s2");

        let plan = crate::trajectory_planning::quad_planning_config::QuadPlanningConfig::
            from_vehicle_params(&cfg);
        assert_eq!(plan.grav, 9.78, "QuadPlanningConfig ignored site.gravity_m_s2");
        // The 10%-of-hover thrust floor is derived from g, not a literal.
        let expect_floor = cfg.airframe.body.mass_kg * 9.78 * 0.1;
        assert!(
            (plan.min_collective_thrust_n - expect_floor).abs() < 1e-6,
            "min_collective_thrust_n kept a 9.81 literal: {} vs {expect_floor}",
            plan.min_collective_thrust_n,
        );
    }

    /// Renamed keys must NOT still resolve. A stale vehicle YAML is
    /// supposed to fail the bake loudly ("unknown tuning key"), and a
    /// stale flash override is supposed to be dropped — both rely on the
    /// old spelling being genuinely gone. Reintroducing one as an alias
    /// would resurrect the split-brain these renames removed: two keys
    /// feeding one field, only one of which the shell displays.
    #[test]
    fn retired_names_absent() {
        for name in [
            // Actuator facts that moved into `airframe.motors`.
            "indi_tau_m0",
            "indi_omega_m0",
            "indi_nonlin_m0",
            "g2_ry_m0",
            "g2_rr_m3",
            // Renamed to name the install fact, then promoted out of the
            // registry entirely — it is a `build:` knob now.
            "gps_fuse_heading",
            "gps_dual_antenna",
        ] {
            assert!(
                <FirmwareConfig as ParamGroup>::param_find(name).is_none(),
                "retired name still resolves: {name}"
            );
        }
    }

    /// Risk guard for feature→param conversions: these defaults define
    /// the DEFAULT FLIGHT BUILD. Getting one wrong silently changes
    /// behavior while every compile check stays green (the old Justfile
    /// built SAMPLER=position; Contouring cost mode requires it).
    #[test]
    fn behavior_preserving_defaults() {
        let d = FirmwareConfig::scaffold();
        assert_eq!(
            d.trajectory.sampler.kind,
            crate::trajectory_planning::sampler::SamplerKind::Position
        );
        assert_eq!(d.mpc.pos_cost_mode, PosCostMode::Contouring);
        // Old build defaults: GPS_FUSE_VEL=on, GPS_FUSE_HEADING=off.
        assert!(d.sensors.gps_fuse_vel);
    }

    /// [`FS_CTRL_TIMEOUT_MIN_S`] must equal the registered minimum of
    /// `fs_ctrl_timeout_s`.
    ///
    /// The firmware asserts the failsafe-ordering invariant against the
    /// constant at compile time, so if the schema minimum were lowered
    /// without moving the constant the assertion would be guarding a
    /// number nobody enforces any more.
    #[test]
    fn fs_ctrl_timeout_min_matches_schema() {
        let idx = <FirmwareConfig as ParamGroup>::param_find("fs_ctrl_timeout_s")
            .expect("fs_ctrl_timeout_s must exist");
        let meta = <FirmwareConfig as ParamGroup>::param_meta(idx);
        assert_eq!(meta.min, FS_CTRL_TIMEOUT_MIN_S);
    }

    /// The `rc` endpoint defaults must reproduce
    /// `StickEndpoints::DEFAULT` exactly.
    ///
    /// The flown stick calibration is built from these three keys. The
    /// constant is the fallback used when the triple is unusable and the
    /// starting point every radio is trimmed from, so a retune on either
    /// side that did not move the other would put the fallback travel
    /// and the schema travel silently at odds.
    #[test]
    fn rc_endpoint_defaults_match_constant() {
        use crate::rc::rc_mapping::StickEndpoints;
        let p = RcParams::default();
        let d = StickEndpoints::DEFAULT;
        assert_eq!(p.min_us as i16, d.min_us);
        assert_eq!(p.mid_us as i16, d.mid_us);
        assert_eq!(p.max_us as i16, d.max_us);
        assert!(d.is_usable(), "the default travel must be strictly ordered");
    }

    /// An unusable endpoint triple must degrade to the standard travel
    /// rather than invert or flatten an axis.
    ///
    /// Each of the three keys validates against its own range
    /// independently, so a set can pass every write check and still be
    /// unordered.
    #[test]
    fn unusable_rc_endpoints_degrade_to_default() {
        use crate::rc::rc_mapping::{ChannelCalibration, StickEndpoints};
        // `mid` below `min`: each value is inside its own schema range.
        let bad = StickEndpoints {
            min_us: 1400,
            mid_us: 900,
            max_us: 2012,
        };
        assert!(!bad.is_usable());
        assert_eq!(bad.or_default(), StickEndpoints::DEFAULT);

        // A centred stick built from it still reads 0 at mid-stick and
        // saturates the right way round.
        let cal = ChannelCalibration::centered_with(0, bad);
        assert_eq!(cal.normalize(StickEndpoints::DEFAULT.mid_us), 0.0);
        assert!(cal.normalize(StickEndpoints::DEFAULT.max_us) > 0.9);
        assert!(cal.normalize(StickEndpoints::DEFAULT.min_us) < -0.9);

        // The throttle branch is selected by `center == min`; a
        // degenerate triple must not lose that.
        let thr = ChannelCalibration::throttle_with(2, bad);
        assert_eq!(thr.center, thr.min);
        assert_eq!(thr.normalize(StickEndpoints::DEFAULT.min_us), 0.0);
        assert!(thr.normalize(StickEndpoints::DEFAULT.max_us) > 0.9);
    }

    /// The `eskf.filter` schema defaults must reproduce
    /// `EskfConfig::default()` exactly. Without this, a retune on either
    /// side silently diverges and a vehicle that pins nothing flies a
    /// different filter than the struct's documented defaults claim.
    #[test]
    fn eskf_filter_defaults_match_config() {
        let p = EskfFilterParams::default().to_eskf_config();
        let c = EskfConfig::default();
        assert_eq!(p.accel_noise_density, c.accel_noise_density);
        assert_eq!(p.gyro_noise_density, c.gyro_noise_density);
        assert_eq!(p.accel_bias_random_walk, c.accel_bias_random_walk);
        assert_eq!(p.gyro_bias_random_walk, c.gyro_bias_random_walk);
        assert_eq!(p.baro_noise_std, c.baro_noise_std);
        assert_eq!(p.mag_noise_std, c.mag_noise_std);
        assert_eq!(p.gate_sigma, c.gate_sigma);
        assert_eq!(p.max_pos_jump_m, c.max_pos_jump_m);
        assert_eq!(p.max_att_jump_rad, c.max_att_jump_rad);
        assert_eq!(p.init_pos_var, c.init_pos_var);
        assert_eq!(p.init_vel_var, c.init_vel_var);
        assert_eq!(p.init_accel_bias_var, c.init_accel_bias_var);
        assert_eq!(p.init_gyro_bias_var, c.init_gyro_bias_var);
        assert_eq!(p.init_att_var_rp, c.init_att_var_rp);
    }

    /// Same contract for the GPS guard. `fuse_velocity` is supplied by
    /// the caller (it lives in `SensorParams`), so pass the default.
    #[test]
    fn eskf_gps_guard_defaults_match_config() {
        let p = EskfGpsGuardParams::default().to_gps_guard_config(true);
        let c = GpsGuardConfig::default();
        assert_eq!(p.max_consecutive_jumps, c.max_consecutive_jumps);
        assert_eq!(p.max_consecutive_rejects, c.max_consecutive_rejects);
        assert_eq!(p.gps_stale_ms, c.gps_stale_ms);
        assert_eq!(p.rtk_fix_debounce_ms, c.rtk_fix_debounce_ms);
        assert_eq!(p.rtk_loss_debounce_ms, c.rtk_loss_debounce_ms);
        assert_eq!(
            p.gyro_bias_cov_trace_xy_thresh,
            c.gyro_bias_cov_trace_xy_thresh
        );
        assert_eq!(p.init_yaw_cov, c.init_yaw_cov);
        assert_eq!(p.gps_min_sv, c.gps_min_sv);
        assert_eq!(p.gps_h_acc_max_mm, c.gps_h_acc_max_mm);
        assert_eq!(p.pos_sigma_floor_fix_m, c.pos_sigma_floor_fix_m);
        assert_eq!(p.pos_sigma_floor_float_m, c.pos_sigma_floor_float_m);
        assert_eq!(p.pos_sigma_floor_none_m, c.pos_sigma_floor_none_m);
        assert_eq!(p.vel_sigma_floor_m_s, c.vel_sigma_floor_m_s);
        assert_eq!(p.reinit_min_carr_soln, c.reinit_min_carr_soln);
        assert_eq!(p.fuse_velocity, true);
    }

    /// Same contract for the mocap guard.
    #[test]
    fn eskf_mocap_guard_defaults_match_config() {
        let p = EskfMocapGuardParams::default().to_mocap_guard_config();
        let c = MocapGuardConfig::default();
        assert_eq!(p.max_consecutive_jumps, c.max_consecutive_jumps);
        assert_eq!(p.max_consecutive_rejects, c.max_consecutive_rejects);
        assert_eq!(p.mocap_stale_ms, c.mocap_stale_ms);
        assert_eq!(p.gyro_bias_cov_trace_thresh, c.gyro_bias_cov_trace_thresh);
        assert_eq!(p.mocap_pos_std, c.mocap_pos_std);
        assert_eq!(p.mocap_att_std, c.mocap_att_std);
        assert_eq!(p.reanchor_radius_m, c.reanchor_radius_m);
    }

    /// The Tier-1 groups replaced firmware constants; their defaults must
    /// reproduce those exact values or a vehicle that pins nothing changes
    /// behaviour silently. Values below are the pre-migration literals.
    #[test]
    fn tier1_defaults_match_replaced_constants() {
        let rc = RcParams::default();
        // rc_mapping.rs ChannelCalibration endpoints.
        assert_eq!((rc.min_us, rc.mid_us, rc.max_us), (988, 1500, 2012));
        // sensors/rc.rs arming + rc_interpreter.rs stick policy.
        assert_eq!(rc.arm_channel, 5);
        assert_eq!(rc.arm_threshold_us, 1500);
        assert_eq!(rc.throttle_mincheck_us, 1050);
        assert_eq!(rc.mission_channel, 4);
        assert_eq!((rc.mission_high_us, rc.mission_low_us), (1700, 1300));
        assert_eq!(rc.throttle_land_us, 1100);
        assert_eq!((rc.launch_us, rc.launch_confirm_frames), (1600, 5));
        assert_eq!(rc.xy_deadband, 0.05);
        assert_eq!(rc.throttle_deadband, 0.12);
        assert_eq!(rc.rate_deadband, 0.05);
        assert_eq!(rc.xy_rate_m_s, 1.0);
        assert_eq!(rc.z_rate_m_s, 0.5);
        assert_eq!(rc.land_rate_m_s, 0.4);
        // New in v55 rather than a replaced constant: the landing floor
        // became relative, so there was no prior literal to match.
        assert_eq!(rc.land_lead_m, 1.0);
        assert_eq!(rc.max_rate_rp_rad_s, 8.0);
        assert_eq!(rc.max_rate_yaw_rad_s, 4.0);

        let site = SiteParams::default();
        assert_eq!(site.gravity_m_s2, 9.81);
        // Envelope defaults carry the old mocap-only values, but disabled:
        // GPS builds had no envelope, and indoor vehicles pin `fence_enable`.
        assert!(!site.fence_enable);
        assert_eq!(site.fence_x_m, 2.5);
        assert_eq!(site.fence_y_m, 3.5);
        assert_eq!(site.fence_z_max_m, 2.0);
        assert_eq!(site.fence_z_min_m, 0.0);

        let s = SafetyParams::default();
        assert_eq!(s.arm_max_tilt_deg, 30.0);
        assert_eq!(s.arm_min_link_quality, 50);
        assert_eq!(s.arm_eskf_mahony_tol_deg, 10.0);
        assert_eq!(s.arm_accel_tol_m_s2, 1.5);
        assert_eq!(s.arm_gyro_limit_rad_s, 0.5);
        assert_eq!(s.arm_switch_hold_s, 0.1);
        assert_eq!(s.arm_link_stats_max_age_s, 0.5);
        assert_eq!(s.fs_rxloss_trigger_s, 0.15);
        assert_eq!(s.fs_guard_period_s, 1.5);
        assert_eq!(s.fs_recovery_period_s, 0.5);
        assert_eq!(s.fs_ctrl_timeout_s, 0.5);

        let f = EskfFaultParams::default();
        assert_eq!(f.pos_timeout_s, 2.0);
        assert_eq!(f.cov_blowup_m2, 25.0);
        assert_eq!(f.nan_reset_hold_s, 3.0);
        assert_eq!(f.cascade_hold_s, 3.0);

        let i = IndiControllerParams::default();
        assert_eq!(i.idle_normalized, 0.055);
        assert_eq!(i.ground_gyro_dps, 100.0);
        assert_eq!(i.ground_accel_g, 0.8);
        assert_eq!(i.ground_thrust_sp_m_s2, 3.0);

        let b = BatteryParams::default();
        assert_eq!(b.cell_detect_v, 4.30);
        assert_eq!(b.no_battery_v, 3.00);
    }

    /// The failsafe windows must stay ordered, or the state machine can
    /// disarm before it ever enters the guard period.
    #[test]
    fn failsafe_windows_are_ordered() {
        let s = SafetyParams::default();
        assert!(
            s.fs_guard_period_s > s.fs_rxloss_trigger_s,
            "guard period must outlast the rx-loss trigger"
        );
        let rc = RcParams::default();
        assert!(
            rc.mission_high_us > rc.mission_low_us,
            "mission trigger Schmitt band must be non-inverted"
        );
        assert!(rc.min_us < rc.mid_us && rc.mid_us < rc.max_us);
    }

    /// The seconds→milliseconds and metres→millimetres conversions must
    /// be exact at the values a user is likely to type, not just at the
    /// defaults (f32 → integer truncation is the failure mode).
    #[test]
    fn eskf_unit_conversions_are_exact() {
        let g = EskfGpsGuardParams {
            stale_s: 0.25,
            rtk_fix_debounce_s: 3.5,
            rtk_loss_debounce_s: 0.05,
            h_acc_max_m: 12.5,
            ..EskfGpsGuardParams::default()
        }
        .to_gps_guard_config(false);
        assert_eq!(g.gps_stale_ms, 250);
        assert_eq!(g.rtk_fix_debounce_ms, 3_500);
        assert_eq!(g.rtk_loss_debounce_ms, 50);
        assert_eq!(g.gps_h_acc_max_mm, 12_500);

        let m = EskfMocapGuardParams {
            stale_s: 0.008,
            ..EskfMocapGuardParams::default()
        }
        .to_mocap_guard_config();
        assert_eq!(m.mocap_stale_ms, 8);
    }

    /// Every parameter's value survives a registry-level copy (the same
    /// get/set surface the YAML bake overlay and the KV replay use). This
    /// replaces the deleted flash-blob round-trip as the drop-a-field
    /// tripwire: a field missing from the derive's get/set arms shows up
    /// as a mismatch here because `test_config` pins non-default values.
    #[test]
    fn round_trip() {
        let mut cfg = test_config();
        cfg.airframe.motor_pole_count = 12;
        cfg.indi.effectiveness.g1_force[2][1] = -1.25;
        cfg.mpc.pos_cost_mode = PosCostMode::Quadratic;
        cfg.trajectory.planner.bfgs_trust.max_iterations = 750;

        let bytes = cfg.to_bytes();
        let mut restored = FirmwareConfig {
            // Deliberately different base — overlay must overwrite everything.
            system: SystemSettings {
                arm_led_enabled: true,
                blackbox_record_set: 0,
                blackbox_rate_div: 1,
                blackbox_mute_mask: 0,
            },
            ..test_config()
        };
        assert!(restored.apply_from_bytes(&bytes));

        for idx in 0..PARAM_COUNT {
            assert_eq!(
                restored.param_get(idx),
                cfg.param_get(idx),
                "param {} mismatch",
                ParamName::of::<FirmwareConfig>(idx).unwrap().as_str()
            );
        }
        assert_eq!(restored.airframe.motors[1].spin_dir, SpinDir::Ccw);
        assert_eq!(restored.mpc.pos_cost_mode, PosCostMode::Quadratic);
        assert_eq!(restored.trajectory.planner.bfgs_trust.max_iterations, 750);
        assert_eq!(restored.system.blackbox_record_set, 2);
        assert!(!restored.system.arm_led_enabled);
    }

    #[test]
    fn rejects_bad_header() {
        let cfg = test_config();
        let good = cfg.to_bytes();

        let mut base = test_config();
        base.airframe.body.mass_kg = 9.9;

        // Corrupt magic
        let mut bad = good;
        bad[0] ^= 0xFF;
        assert!(!base.apply_from_bytes(&bad));
        // Corrupt version
        let mut bad = good;
        bad[4] = bad[4].wrapping_add(1);
        assert!(!base.apply_from_bytes(&bad));
        // Corrupt payload (CRC)
        let mut bad = good;
        bad[HEADER_SIZE + 8] ^= 0x01;
        assert!(!base.apply_from_bytes(&bad));
        // Base untouched by all three rejections.
        assert_eq!(base.airframe.body.mass_kg, 9.9);
        // The pristine image applies.
        assert!(base.apply_from_bytes(&good));
        assert_eq!(base.airframe.body.mass_kg, 0.55);
    }

    #[test]
    fn named_access() {
        let mut cfg = test_config();
        assert_eq!(cfg.get_named("mass"), Some(ParamValue::F32(0.55)));
        assert_eq!(cfg.get_named("m0_spin"), Some(ParamValue::U8(1)));
        assert_eq!(cfg.get_named("blackbox_tier"), Some(ParamValue::U8(2)));
        assert!(cfg.set_named("indi_rate_p", 90.0));
        assert_eq!(cfg.indi.controller.rate_gains[1], 90.0);
        assert!(cfg.set_named("m1_spin", 1.0));
        assert_eq!(cfg.airframe.motors[1].spin_dir, SpinDir::Cw);
        assert!(!cfg.set_named("learn_fx_hz", 1.0)); // removed in v30
        assert!(!cfg.set_named("no_such", 1.0));
    }

    /// SamplerParams::default() must mirror PositionSamplerParams::defaults().
    #[test]
    fn sampler_defaults_match_position_sampler_defaults() {
        let s = SamplerParams::default();
        let p = PositionSamplerParams::default();
        let proj = s.to_position_sampler_params();
        assert_eq!(proj.axis_weights_sqrt, p.axis_weights_sqrt);
        assert_eq!(proj.search_dt, p.search_dt);
        assert_eq!(proj.max_search_steps, p.max_search_steps);
        assert_eq!(proj.radius_of_acceptance, p.radius_of_acceptance);
        assert_eq!(proj.max_lag_s, p.max_lag_s);
        assert_eq!(proj.max_lead_s, p.max_lead_s);
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;
    use crate::param_registry::{ParamGroup, ParamName};

    /// Every parameter resolves to a group path, and the paths arrive in
    /// contiguous runs — the property the shell's grouped listing relies on
    /// to print each header exactly once while streaming.
    ///
    /// No alloc here (no_std crate), so "has this group appeared before?" is
    /// answered by rescanning the prefix rather than by a set.
    #[test]
    fn group_paths_are_contiguous_runs() {
        let path_at = |i: usize| {
            ParamName::path_of::<FirmwareConfig>(i).expect("every index has a path")
        };
        let mut runs = 0usize;
        for idx in 0..PARAM_COUNT {
            let cur = path_at(idx);
            if idx > 0 && path_at(idx - 1).as_str() == cur.as_str() {
                continue; // still inside the current run
            }
            runs += 1;
            // A new run started here: this path must not appear anywhere
            // earlier, or its header would be printed twice.
            for j in 0..idx.saturating_sub(1) {
                assert!(
                    path_at(j).as_str() != cur.as_str(),
                    "group {:?} reappears at index {idx} after ending at {j}",
                    cur.as_str()
                );
            }
        }
        assert!(runs > 5, "expected several groups, got {runs}");
    }

    /// The path is the *structural* location, which name prefixes cannot
    /// recover: `eskf_mocap_pos_std` lives in `eskf.mocap_guard`.
    #[test]
    fn nested_paths_are_dotted_and_not_prefix_derivable() {
        let idx = FirmwareConfig::param_find("eskf_mocap_pos_std").expect("key exists");
        let path = ParamName::path_of::<FirmwareConfig>(idx).unwrap();
        assert_eq!(path.as_str(), "eskf.mocap_guard");

        let idx = FirmwareConfig::param_find("mass").expect("key exists");
        let path = ParamName::path_of::<FirmwareConfig>(idx).unwrap();
        assert_eq!(path.as_str(), "airframe.body");
    }

    #[test]
    fn path_filter_is_separator_insensitive_and_boundary_safe() {
        assert!(ParamName::path_matches("eskf.mocap_guard", "eskf"));
        assert!(ParamName::path_matches("eskf.mocap_guard", "eskf-mocap_guard"));
        assert!(ParamName::path_matches("eskf.mocap_guard", "eskf.mocap-guard"));
        assert!(ParamName::path_matches("eskf.mocap_guard", "ESKF"));
        // Must not match a partial segment.
        assert!(!ParamName::path_matches("eskf.mocap_guard", "es"));
        assert!(!ParamName::path_matches("mpc", "mp"));
        assert!(!ParamName::path_matches("eskf.filter", "eskf.mocap_guard"));
    }
}
