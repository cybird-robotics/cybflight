//! MPC outer-loop task: 50 Hz position/attitude controller using
//! `SimpleSqpSolver` over `QuadModel`. Publishes body-rate + collective-thrust
//! commands to `super::RATE_COMMAND` for the INDI inner loop to consume.
//!
//! Lives in its own embassy task to avoid blocking the 8 kHz INDI loop.
//! Stack-allocated locals are tiny (~1 KB); the ~32 KB SQP workspace lives
//! in BSS via `static_cell::StaticCell`.
//!
//! Gated on `cfg(feature = "outer_mpc")`.

#![cfg(feature = "outer_mpc")]

use cybflight_core::mpc::model_utils;
use cybflight_core::mpc::quad_model::{N as MPC_N, NU as MPC_NU, NX as MPC_NX, PosCostMode};
use cybflight_core::mpc::{QuadModel, SimpleQuadProblem, SimpleSqpSolver};
use cybflight_core::params::VehicleParams;
use cybflight_core::rotation::quaternion_from_zb_and_yaw;
use cybflight_core::trajectory_planning::flatness::reference_quaternion;
use cybflight_core::trajectory_planning::minco_snap::{FlatnessFault, flatness_to_thrust_omega};
#[cfg(feature = "position_sampler")]
use cybflight_core::trajectory_planning::sampler::PositionSampler;
#[cfg(not(feature = "position_sampler"))]
use cybflight_core::trajectory_planning::sampler::TimeSampler;
use cybflight_core::trajectory_planning::sampler::{Sampler, SamplerInputs, SamplerNode};
use embassy_time::{Duration, Instant, Ticker};
use nalgebra::{SVector, UnitQuaternion, Vector3};
use static_cell::StaticCell;

type MpcStateVec = SVector<f32, MPC_NX>;
type MpcInputVec = SVector<f32, MPC_NU>;

use crate::msgs;
use crate::sensors::VEHICLE_ODOMETRY;
use crate::vehicle::QUADROTOR_BODY;

/// Static-allocated SQP workspace (~32 KB in BSS, init-once at task startup).
static MPC_SOLVER: StaticCell<SimpleSqpSolver> = StaticCell::new();

/// Maximum age of an odometry sample (against its own timestamp) we will use
/// as the MPC initial state. ESKF divergence often produces valid-looking
/// (finite) but stale odometry; without this gate the MPC would happily plan
/// from an ancient pose.
///
/// Tightened from 100 ms to 50 ms to support the position-sampler path: at
/// the planner's 4 m/s cap, 100 ms of stale `state_pos` translates to up to
/// 0.4 m of position error fed into PositionSampler's closest-point search,
/// which on a tight curve or near a self-intersection can lock the search
/// onto the wrong τ. 50 ms = 2.5 outer-loop ticks worth of slack and caps
/// the worst-case input error at ~0.2 m. TimeSampler doesn't read
/// `state_pos` and is unaffected by this tighter gate.
const ODOM_STALE_TIMEOUT: Duration = Duration::from_millis(50);

/// Wall-clock budget for one SQP solve. The solver is synchronous (no
/// `with_timeout` possible) so this is a *post-hoc* check: if a solve exceeds
/// the budget we discard its output and refuse to publish, on the theory that
/// (a) the command is now stale relative to the 50 Hz tick, and (b) a solve
/// that ran long is more likely to have diverged. Persistent overruns will
/// trip the inner loop's `MPC_CMD_STALE_TIMEOUT` and the failsafe watchdog.
// const MPC_SOLVE_BUDGET: Duration = Duration::from_millis(8);

const POS_PUB_DECIMATION: u32 = 1;
const ATT_PUB_DECIMATION: u32 = 1;
const OCP_PUB_DECIMATION: u32 = 1;
const MISSION_PUB_DECIMATION: u32 = 5;

/// Reference-quaternion construction for the per-node attitude target
/// in `x_refs`. The two paths encode **different yaw conventions** —
/// pick by what the source of `yaw_setpoint_rad` semantically means:
///
/// * `false` — `flatness::reference_quaternion(acc, yaw, g)`. Builds
///   `x_b = (y_c × z_b).normalize()` where `y_c = (-sin yaw, cos yaw, 0)`
///   is the yaw-aligned world ŷ. **Yaw input = world-frame azimuth of
///   the body x-axis projection.** The drone's compass heading on the
///   ground tracks the input yaw exactly, regardless of tilt — this is
///   what an operator/RC stick or "point the camera north" command
///   means by "yaw". Singular when `y_c` is parallel to `z_b` (≈ 90°
///   tilt in a specific azimuthal direction).
///
/// * `true` — `rotation::quaternion_from_zb_and_yaw(z_b, yaw, true)`.
///   Closed-form tilt-then-yaw. **Yaw input = the intrinsic Euler
///   angle of the tilt-then-yaw decomposition.** The body x-axis
///   projection on the world xy-plane rotates with tilt, so the
///   compass heading is not directly the input yaw. The natural choice
///   when yaw is a derived flat output (e.g. from a planner that
///   already accounts for tilt). Singular only at `z_b.z = -1` (drone
///   fully inverted), where the library substitutes the canonical
///   yaw-consistent 180° flip.
///
/// Default `false` for the firmware: yaw comes from the RC stick
/// integrator (`rc_interpreter`) and is the operator-facing compass
/// heading. Flip to `true` for trajectories whose yaw schedule was
/// designed under the tilt-then-yaw convention.
const USE_TILT_REFERENCE_QUATERNION: bool = true;

/// Enable / disable the trajectory-derived `u_refs` feedforward.
///
/// * `true` (default): each Executing-tick non-past-end node has its
///   `u_refs[k] = [mass·‖α‖, ω_x, ω_y, ω_z]` populated from
///   [`flatness_to_thrust_omega`] applied to the trajectory's
///   (acc, jerk) sample. The MPC's input cost biases toward the
///   open-loop differential-flatness solution, so the SQP only has to
///   handle model error and disturbance — measurably tighter tracking
///   on aggressive (high-α, high-‖j‖) trajectories.
/// * `false`: every horizon step is biased to `hover_u` regardless of
///   trajectory state. Equivalent to the pre-feedforward behaviour and
///   useful as an A/B baseline if the feedforward ever needs to be
///   debugged in flight without a firmware reflash + flash-param dance.
///
/// Note that the flag only controls the *bias term* in the SQP's input
/// cost. The trajectory-derived **state** references (`x_refs[k]` —
/// position, velocity, attitude) remain active in both modes; turning
/// this off does not turn the trajectory into a hover.
const USE_FLATNESS_U_REF_FEEDFORWARD: bool = true;

/// One-shot flag for the "reference at inverted pole" diagnostic warn
/// inside the per-node fan-out. Stays `true` after the first hit so the
/// log isn't flooded — the planner-side problem (or genuine acrobatic
/// intent) is the same on every subsequent node, and one warning per
/// firmware boot is enough to catch it. Only meaningful when
/// `USE_TILT_REFERENCE_QUATERNION = true`; the cross-product path has
/// a different singularity profile.
static INVERTED_REF_WARNED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// One-shot flag for the flatness-feedforward fault diagnostic. Same
/// rationale as `INVERTED_REF_WARNED`: the underlying condition (free-
/// fall α, infeasible trajectory sample) repeats on every subsequent
/// node, so one boot-level warning is enough to point a debugger at it.
static FLATNESS_U_REF_FAULT_WARNED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Reject odometry with any non-finite component.
fn odom_is_valid(odom: &msgs::VehicleOdometry) -> bool {
    let fin = |v: &Vector3<f32>| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    let q = odom.pose.orientation.as_vector();
    fin(&odom.pose.position)
        && q.x.is_finite()
        && q.y.is_finite()
        && q.z.is_finite()
        && q.w.is_finite()
        && fin(&odom.twist.linear)
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
///    `PositionSampler` does so by construction; the `TimeSampler` build
///    is paired with hover-style references where the tangent isn't
///    well-defined for our missions. Compile-time guard: when the
///    `position_sampler` feature is off, this helper unconditionally
///    clamps `pos_cost_mode` to `Quadratic`.
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
fn build_outer_quad_model(vp: &VehicleParams) -> QuadModel {
    #[cfg_attr(feature = "position_sampler", allow(unused_mut))]
    let mut model = QuadModel::from_vehicle_params(vp);
    #[cfg(not(feature = "position_sampler"))]
    {
        if model.pos_cost_mode != PosCostMode::Quadratic {
            defmt::warn!(
                "MPC: position_sampler feature disabled — \
                 forcing PosCostMode::Quadratic (Contouring requires the position sampler)"
            );
            model.pos_cost_mode = PosCostMode::Quadratic;
            model.w_pos[1] = model.w_pos[0];
            model.w_pos[2] = model.w_pos[0];
        }
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

#[embassy_executor::task]
pub async fn control_loop_task() {
    // ── Construct MPC ──────────────────────────────────────────────────
    let params = crate::params::get();
    let mpc_solver: &mut SimpleSqpSolver = MPC_SOLVER.init(SimpleSqpSolver::new());
    let mut mpc_problem = SimpleQuadProblem::with_rk4(build_outer_quad_model(&params), MPC_N);

    // ── Reference + warm-start trajectories ────────────────────────────
    let mut hover_thrust = QUADROTOR_BODY.mass_kg * 9.81;
    let identity_x = {
        let mut x = MpcStateVec::zeros();
        x[6] = 1.0; // qw = 1 (identity quaternion in xyzw layout)
        x
    };
    let mut x_refs: [MpcStateVec; MPC_N + 1] = [identity_x; MPC_N + 1];
    let hover_u = MpcInputVec::from_row_slice(&[hover_thrust, 0.0, 0.0, 0.0]);
    let mut u_refs: [MpcInputVec; MPC_N] = [hover_u; MPC_N];
    let mut u_warm: [MpcInputVec; MPC_N] = u_refs;

    // ── Publishers / Subscribers ──────────────────────────────────────
    let pos_ctrl_pub = super::POSITION_CONTROL_SETPOINT.immediate_publisher();
    let att_ctrl_pub = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();
    let ocp_pub = super::OCP_SOLVER_OUTPUT.immediate_publisher();
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
    defmt::info!("MPC outer loop task started (50 Hz)");

    // ── Param hot-reload bookkeeping (mirror of indi_task pattern) ────
    let mut local_param_ver =
        crate::params::PARAM_VERSION.load(core::sync::atomic::Ordering::Acquire);

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
    // Compile-time selected via the `position_sampler` feature. The
    // sampler is pure — it never touches mission state, the active
    // setpoint cell, or the trajectory slot. The outer loop owns the
    // Idle ↔ Executing transition and calls `sampler.reset()` on entry
    // so a stateful variant starts each mission fresh.
    //
    // PositionSampler params come from VehicleParams.sampler so flash
    // updates land via the existing PARAM_VERSION hot-reload below.
    #[cfg(not(feature = "position_sampler"))]
    let mut sampler = Sampler::Time(TimeSampler::new());
    #[cfg(feature = "position_sampler")]
    let mut sampler = Sampler::Position(PositionSampler::new(
        params.sampler.to_position_sampler_params(),
    ));
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

    // ── 50 Hz tick loop ───────────────────────────────────────────────
    let mut ticker = Ticker::every(Duration::from_millis(20));
    let mut tick: u32 = 0;
    loop {
        ticker.next().await;
        tick = tick.wrapping_add(1);

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
        // Under the current stick integrator and MINCO planner this is
        // always ψ=0 (identity quaternion), but threading the actual
        // `yaw_rad` through keeps the plumbing honest and makes future
        // yaw-commanding writers (e.g. a yaw-capable planner) drop in
        // without touching this file.
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
                let np = crate::params::get();
                mpc_problem = SimpleQuadProblem::with_rk4(build_outer_quad_model(&np), MPC_N);
                hover_thrust = np.body.mass_kg * 9.81;
                let hover_u = MpcInputVec::from_row_slice(&[hover_thrust, 0.0, 0.0, 0.0]);
                u_refs = [hover_u; MPC_N];
                u_warm = u_refs;
                // max_vel_m_s = np.planner.max_vel_m_s;

                // Rebuild the PositionSampler from the new flash params.
                // The TimeSampler arm is stateless; nothing to reload.
                // Disarmed-only reload guarantees we don't swap a sampler
                // mid-mission. The new sampler starts with no
                // `prev_query_tau` — equivalent to a `reset()` — which is
                // the right semantics: any tunable change invalidates the
                // last tick's converged search base.
                #[cfg(feature = "position_sampler")]
                {
                    sampler = Sampler::Position(PositionSampler::new(
                        np.sampler.to_position_sampler_params(),
                    ));
                }

                defmt::info!("MPC outer loop: params reloaded (ver {})", cur);
            }
        }

        // 3. Drain latest valid odometry (skip the tick if none arrived).
        //    Validity = finite components AND timestamp within
        //    `ODOM_STALE_TIMEOUT` of now (and not future-dated). The timestamp
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
            if now_for_odom.duration_since(o.timestamp) > ODOM_STALE_TIMEOUT {
                continue;
            }
            latest = Some(o);
        }
        let Some(odom) = latest else {
            defmt::warn!("MPC outer loop: no fresh odometry, skipping tick");
            continue;
        };

        // 4. Build MPC initial state from odometry.
        // QuadModel state = [px, py, pz, qx, qy, qz, qw, vx, vy, vz].
        let q = odom.pose.orientation;
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
                        yaw_rad: 0.0,
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
        let mut mission_done_final: Option<Vector3<f32>> = None;
        // Captured for MISSION_STATUS telemetry publish below.
        let mut tau_and_duration: Option<(f32, f32)> = None;
        let mut solve_diag: Option<msgs::SolveDiagnostics> = None;
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
                let now = Instant::now();
                let tau0_s = if now >= traj.t_start {
                    now.duration_since(traj.t_start).as_micros() as f32 * 1e-6
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
                let result = sampler.sample(&inputs, &mut sample_buf);

                // Per-node fan-out into x_refs. Quaternion construction
                // stays here because it depends on `yaw_setpoint_rad`
                // (RC-owned policy, not a trajectory property) and the
                // past-end identity-tilt rule (the pre-loop yaw-only fill
                // already wrote the right quaternion for those nodes, so
                // we leave it untouched).
                //
                // u_refs feedforward is populated alongside x_refs in the
                // same node loop. Each non-past-end node gets its own
                // `[T, ω_x, ω_y, ω_z]` from the pole-safe flatness map
                // applied to (acc, jerk) plus quasi-static yaw
                // (`yaw_rate = 0`); past-end nodes bias to hover
                // (zero body rate, mass·g thrust). On a `FlatnessFault`
                // (free-fall α — should be unreachable on a valid MINCO
                // schedule) we fall back to hover_u for that node and
                // emit a single boot-level diagnostic.
                let grav = mpc_problem.model.grav;
                let mass = mpc_problem.model.mass;
                let mpc_n_inputs = MPC_N; // u_refs is length MPC_N (no terminal input)
                for (k, n) in sample_buf.iter().enumerate() {
                    x_refs[k][0] = n.pos[0];
                    x_refs[k][1] = n.pos[1];
                    x_refs[k][2] = n.pos[2];
                    x_refs[k][7] = n.vel[0];
                    x_refs[k][8] = n.vel[1];
                    x_refs[k][9] = n.vel[2];

                    // Single-call flatness: when `USE_TILT_REFERENCE_QUATERNION`
                    // is true, the attitude reference and the body-rate +
                    // thrust feedforward share their entire computation
                    // (z_b construction, projection-based dz_b, the pole-
                    // safe `quaternion_from_zb_and_yaw` substitution). We
                    // call `flatness_to_thrust_omega` once per node and
                    // unpack both outputs, instead of building z_b twice
                    // and invoking the quaternion construction redundantly.
                    //
                    // The cross-product attitude path is structurally
                    // different (its yaw convention rotates body-x heading
                    // around the world ẑ rather than around z_b) and shares
                    // no intermediates with the tilt-yaw flatness map, so
                    // it stays on its own branch that calls
                    // `reference_quaternion` directly.
                    // Per-node `u_refs` population. There are three
                    // sub-cases here, and the `tilt_attitude` cache for
                    // the `q_ref` path below depends on which one fires:
                    //
                    //  (a) Feedforward enabled, non-terminal, non-past-
                    //      end node: call `flatness_to_thrust_omega` and
                    //      use both outputs (u_refs + cached attitude).
                    //  (b) Feedforward disabled OR past-end node:
                    //      `u_refs[k]` stays at the top-of-tick `hover_u`
                    //      default (we deliberately do *not* re-write
                    //      it). When the feedforward is disabled we also
                    //      need the attitude reference, so we still call
                    //      `flatness_to_thrust_omega` for non-past-end
                    //      nodes purely to populate `tilt_attitude` —
                    //      saves an extra `quaternion_from_zb_and_yaw`
                    //      call further down. Past-end nodes have no
                    //      tilt reference (the pre-loop yaw-only fill
                    //      is the right answer), so we skip the call.
                    //  (c) Terminal node (k == MPC_N): no `u_refs[k]`
                    //      slot exists. The `q_ref` path below falls
                    //      back to a direct `quaternion_from_zb_and_yaw`
                    //      call.
                    let mut tilt_attitude: Option<UnitQuaternion<f32>> = None;
                    let needs_flatness_call = !n.past_end
                        && (USE_FLATNESS_U_REF_FEEDFORWARD || USE_TILT_REFERENCE_QUATERNION);
                    if needs_flatness_call {
                        match flatness_to_thrust_omega(
                            n.acc,
                            n.jerk,
                            yaw_setpoint_rad,
                            0.0,
                            grav,
                        ) {
                            Ok((tpm, attitude, omega)) => {
                                tilt_attitude = Some(attitude);
                                if USE_FLATNESS_U_REF_FEEDFORWARD && k < mpc_n_inputs {
                                    u_refs[k] = MpcInputVec::from_row_slice(&[
                                        mass * tpm,
                                        omega.x,
                                        omega.y,
                                        omega.z,
                                    ]);
                                }
                            }
                            Err(fault) => {
                                if !FLATNESS_U_REF_FAULT_WARNED
                                    .swap(true, core::sync::atomic::Ordering::Relaxed)
                                {
                                    let kind: &'static str = match fault {
                                        FlatnessFault::NearFreeFall => "NearFreeFall",
                                        FlatnessFault::InvertedTilt => "InvertedTilt",
                                    };
                                    defmt::warn!(
                                        "outer_loop: flatness fault ({=str}, node={=usize}) — falling back to hover_u and tilt-yaw fallback; trajectory may be infeasible at this sample",
                                        kind,
                                        k,
                                    );
                                }
                                // tilt_attitude stays None → q_ref path
                                // falls back to direct construction.
                                // u_refs[k] stays at top-of-tick hover_u.
                            }
                        }
                    }
                    if !n.past_end {
                        let q_ref = if USE_TILT_REFERENCE_QUATERNION {
                            // Reuse the attitude already produced by
                            // `flatness_to_thrust_omega`. For the terminal
                            // horizon node (k == MPC_N) we never entered
                            // the u_refs branch above, so `tilt_attitude`
                            // is still `None` — fall back to a direct
                            // call. The pre-loop inverted-pole diagnostic
                            // is preserved on the *fallback* call only;
                            // the in-loop tilt_attitude value already came
                            // from the same closed form, so re-checking
                            // its z_b would be redundant noise.
                            if let Some(q) = tilt_attitude {
                                q
                            } else {
                                let acc_cmd = Vector3::new(
                                    n.acc[0],
                                    n.acc[1],
                                    n.acc[2] + grav,
                                );
                                let inv_norm = 1.0 / acc_cmd.norm().max(1e-8);
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
                                quaternion_from_zb_and_yaw(&z_b, yaw_setpoint_rad, true)
                            }
                        } else {
                            // Cross-product construction. Yaw input is
                            // the world-frame compass heading of the
                            // body x-axis projection — the operator-
                            // facing "yaw" the RC stick integrator
                            // produces. Singular at 90° tilts aligned
                            // with the yaw axis; on aggressive racing
                            // trajectories prefer the tilt path above.
                            reference_quaternion(n.acc, yaw_setpoint_rad, grav)
                        };
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
                    mission_done_final = Some(sample_buf[MPC_N].pos);
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
                if p[0].is_finite() && p[1].is_finite() && p[2].is_finite() {
                    let now = Instant::now();
                    super::ACTIVE_POSITION_SETPOINT.lock(|cell| {
                        cell.set(Some(super::ActiveSetpoint {
                            timestamp: now,
                            position: Vector3::new(p[0], p[1], p[2]),
                            yaw_rad: 0.0,
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
        }

        // 6. Solve one SQP iteration (max_iters = 1, matches host benchmark).
        let solve_start = Instant::now();
        let result = mpc_solver.solve(&mpc_problem, &mpc_x0, &x_refs, &u_refs, &u_warm, 1, 1e-3);
        let solve_time_us = Instant::now().duration_since(solve_start).as_micros();
        u_warm = *mpc_solver.u_bar();
        let mut u0 = mpc_solver.u_bar()[0];

        // 7. Non-finite guard — skip publishing on NaN/Inf. The inner loop's
        //    last_mpc_cmd_time staleness check will eventually trip the
        //    failsafe if this persists.
        if !u0.iter().all(|v| v.is_finite()) {
            defmt::warn!("MPC outer loop: non-finite output, skipping tick");
            // Reset warm-start so a transient NaN does not poison the next
            // iteration via u_warm.
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
        let setpoint = msgs::AttitudeControlSetpoint {
            timestamp: publish_time,
            collective_thrust_n: u0[0],
            attitude_quaternion: ref_att,
            body_rate_rad_s: Vector3::new(u0[1], u0[2], u0[3]),
            torque_n_m: Vector3::zeros(),
        };
        super::RATE_COMMAND.signal(setpoint.clone());
        ctrl_sp_pub.publish_immediate(setpoint);

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
            att_ctrl_pub.publish_immediate(msgs::AttitudeControlSetpoint {
                timestamp: publish_time,
                collective_thrust_n: u0[0],
                attitude_quaternion: ref_att,
                body_rate_rad_s: Vector3::new(u0[1], u0[2], u0[3]),
                torque_n_m: Vector3::zeros(),
            });
        }
        if tick % OCP_PUB_DECIMATION == 0 {
            ocp_pub.publish_immediate(msgs::OcpSolverOutput {
                timestamp: publish_time,
                command: u0,
                iterations: result.iters as i32,
                converged: result.converged,
                solve_time_us,
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
            tracking_err_pub.publish_immediate(super::TrackingError {
                timestamp: publish_time,
                pos_err,
                vel_err,
                attitude_err: ea,
                body_rate_err: Vector3::zeros(),
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
