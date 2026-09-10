//! Host-side controller stacks.
//!
//! Three stacks exist, all implementing the same `Controller` trait. The
//! runner drives them at their declared `tick_rate_hz()`.
//!
//! - `MpcIndiController` (default, firmware-match) — `SimpleSqpSolver` over the
//!   10-state `QuadModel` running at 100 Hz, producing (collective thrust,
//!   body-rate reference). An `IndiController` from `cybflight-core` runs at
//!   8 kHz, tracking the rate reference and emitting per-motor commands.
//!   Its normalized commands go to the plant **unmodified** — the same bytes
//!   that would reach an ESC.
//!
//! - `MpcDirectController` — `FullSqpSolver` over the 13-state `FullQuadModel`
//!   at 100 Hz, emitting per-motor thrusts directly. Useful as an
//!   upper-bound reference: "what would MPC + a perfect inner loop do?"
//!
//! - `CascadeController` — PD position + geometric attitude + rate-P + mixer,
//!   at 100 Hz. Legacy baseline; PD position control is no longer the
//!   firmware's active path. Kept for diagnostic diffs.
//!
//! `CtbrRateLoop` is not a stack but a component: the inner half of the
//! first two, packaged for outer stages that command collective thrust and
//! body rates without driving the full `Controller` interface — which is
//! what `cybflight_core::acmpc` and the neural policies do, since they
//! consume a gate course rather than a setpoint trajectory.
//!
//! ## Output convention
//!
//! Every stack returns a per-motor **normalized command** `d ∈ [0, 1]`, which
//! is what the plant's actuator model consumes. `MpcIndiController` returns
//! INDI's own output untouched, so the whole thrust-linearization path
//! (`ThrustModel`, the voltage argument, the `[0.025, 1.0]` nonlinearity
//! clamp) is live rather than bypassed.
//!
//! The two thrust-producing baselines need an inverse to reach `d`. They get
//! the plant's *exact* inverse via [`ThrustCommandMap`] rather than INDI's
//! approximate one, deliberately: they exist to answer "how well does this
//! control law track?", and folding an ESC-linearization error into that
//! number would make a controller regression indistinguishable from a
//! linearization regression.

use crate::baselines::{
    AttitudeControlSetpoint, AttitudeControlState,
    geometric_controller::GeometricAttitudeController,
};
use cybflight_core::attitude_control::geometric_controller::{
    GeometricTrackingController, GeometricTrackingParams, GeometricTrackingReference,
    GeometricTrackingState,
};
use air_filters::iir::biquad::{
    BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType, DirectForm2,
};
use air_filters::Filter;
use cybflight_core::acmpc::CtbrCommand;
use cybflight_core::indi::{
    controller::{IndiConfig, IndiController, MotorState, NU as INDI_NU, NV},
    effectiveness::IndiMotorParams,
    linearization::ThrustModel,
    rpm_tracker::RpmInput,
};

type Biquad = BiquadFilter<f32, DirectForm2<f32>>;
use cybflight_core::mixer::{LinearAllocator, MotorEffectiveness};
use cybflight_core::mpc::cost_adapt::{
    situation_obs, CostNominal, CostPolicy, SituationInputs, COST_MAP_VERSION, NZ as COST_NZ,
    OBS_DIM as COST_OBS_DIM,
};
use cybflight_core::nn::mlp::LayerShape;
use cybflight_core::mpc::{
    FullQuadModel, FullQuadProblem, FullSqpSolver, N as FULL_N, NU, NX, QuadModel,
    SimpleQuadProblem, SimpleSqpSolver, TinyMpc, TinySettings,
    quad_model::{N as SIMPLE_N, NU as SIMPLE_NU, NX as SIMPLE_NX, PosCostMode},
    tinympc::{HOVER_NU, HOVER_NX, hover_linear_model},
};
use cybflight_core::params::FirmwareConfig;
use cybflight_core::position_control::{
    self, PositionControlSetpoint, PositionControlState, pd_ff_control::PositionController,
};
use cybflight_core::rotation::quaternion_from_zb_and_yaw;
use cybflight_core::trajectory_planning::flatness::{
    flatness_to_thrust_omega, flatness_to_thrust_omega_tilt_yaw, reference_quaternion,
};
use cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial;
use cybflight_core::trajectory_planning::sampler::{
    PositionSampler, PositionSamplerParams, SamplerInputs, SamplerNode,
};
use nalgebra::{Matrix3, Quaternion, SVector, UnitQuaternion, Vector3, Vector4, stack, vector};

/// Reference-quaternion construction switch for the MPC `q_ref` chain
/// inside the sim controllers. Mirrors the firmware's
/// `outer_loop::USE_TILT_REFERENCE_QUATERNION` so regression metrics
/// reflect the path that actually flies.
///
/// * `false` — `flatness::reference_quaternion(acc, yaw, g)`
///   (cross-product). Yaw input = world-frame compass heading of the
///   body x-axis projection. Operator-facing yaw semantics. Singular
///   at 90° tilts aligned with the yaw axis.
/// * `true` — `rotation::quaternion_from_zb_and_yaw(z_b, yaw, true)`
///   (tilt-then-yaw closed form). Yaw input = intrinsic Euler angle of
///   the tilt-yaw decomposition. Natural for trajectories whose yaw
///   schedule was designed under this convention. Singular only at
///   fully inverted, with a yaw-consistent fallback.
const USE_TILT_REFERENCE_QUATERNION: bool = true;

/// One-shot warning for "reference attitude at the inverted pole",
/// mirroring the firmware's `INVERTED_REF_WARNED`. Only fires on the
/// tilt path.
static INVERTED_REF_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[inline]
fn q_ref_from_setpoint(acc: Vector3<f32>, yaw: f32, grav: f32) -> UnitQuaternion<f32> {
    q_ref_with_map(acc, yaw, grav, USE_TILT_REFERENCE_QUATERNION)
}

/// Same as [`q_ref_from_setpoint`] with the map chosen per call
/// (`tilt = true` → tilt-then-yaw closed form, the firmware default for
/// `flatness_map: tilt_yaw` missions).
#[inline]
fn q_ref_with_map(acc: Vector3<f32>, yaw: f32, grav: f32, tilt: bool) -> UnitQuaternion<f32> {
    if tilt {
        let acc_cmd = Vector3::new(acc[0], acc[1], acc[2] + grav);
        let inv_norm = 1.0 / acc_cmd.norm().max(1e-8);
        let z_b = acc_cmd * inv_norm;
        if z_b.z < -1.0 + 1e-3
            && !INVERTED_REF_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "[sim controller] reference attitude at the inverted pole \
                 (z_b.z={:.4}, yaw={:.3}) — using fallback 180° flip",
                z_b.z, yaw,
            );
        }
        quaternion_from_zb_and_yaw(&z_b, yaw, true)
    } else {
        reference_quaternion(acc, yaw, grav)
    }
}

use crate::plant::PlantParams;
use crate::sensors::{ImuMeasurement, RotorTelemetry};
use crate::trajectory::Setpoint;

/// Exact thrust → normalized-command inverse for the plant's actuator map.
///
/// Composes `ω = √(T/c_T)` with the inverse of the steady-state throttle
/// curve. Used only by the thrust-producing diagnostic baselines; the INDI
/// stack does its own linearization, which is the point of testing it.
#[derive(Clone, Debug)]
pub struct ThrustCommandMap {
    plant: PlantParams,
}

impl ThrustCommandMap {
    pub fn new(plant: PlantParams) -> Self {
        Self { plant }
    }

    /// Per-motor thrust [N] → per-motor normalized command `d ∈ [0, 1]`.
    pub fn command(&self, thrust_n: &SVector<f32, NU>) -> SVector<f32, NU> {
        SVector::<f32, NU>::from_fn(|i, _| {
            self.plant
                .command_for_omega(i, self.plant.omega_for_thrust(i, thrust_n[i]))
        })
    }
}

/// Inner loop for an outer stage that commands **CTBR** — collective
/// thrust plus a body-rate reference — on cybflight's own plant.
///
/// [`cybflight_core::acmpc`] emits exactly that, as does
/// [`MpcIndiController`]'s outer MPC; this is the cheap, model-based
/// alternative to the INDI stack for closing that loop: a proportional
/// rate law into the linear thrust/torque allocator, then the plant's
/// exact actuator inverse.
///
/// Unlike INDI it needs no IMU, no ESC telemetry and no filter state, so
/// it can be driven at the outer loop's own rate. It is correspondingly
/// less robust: the allocation inverts the *nominal* effectiveness model,
/// with no incremental correction for what the model gets wrong.
pub struct CtbrRateLoop {
    allocator: LinearAllocator<NU>,
    cmd_map: ThrustCommandMap,
    rate_kp: Vector3<f32>,
    inertia: Matrix3<f32>,
    mass_kg: f32,
    per_motor_max_n: f32,
    max_collective_n: f32,
    idle_n: f32,
}

impl CtbrRateLoop {
    /// `rate_kp` is the proportional gain per axis [1/s]; its usable
    /// magnitude is bounded by the rotor time constant.
    pub fn new(vp: &FirmwareConfig, sim: &vehicle_yaml::SimYaml, rate_kp: Vector3<f32>) -> Self {
        let per_motor_max_n = vp
            .airframe
            .motors
            .iter()
            .map(|m| m.max_thrust_n)
            .fold(0.0f32, f32::max);
        let max_collective_n = vp.airframe.motors.iter().map(|m| m.max_thrust_n).sum::<f32>();
        Self {
            allocator: LinearAllocator::new(MotorEffectiveness::from_motors(&vp.airframe.motors)),
            cmd_map: ThrustCommandMap::new(PlantParams::from_config(vp, sim)),
            rate_kp,
            inertia: vp.airframe.body.inertia_matrix(),
            mass_kg: vp.airframe.body.mass_kg,
            per_motor_max_n,
            max_collective_n,
            idle_n: 0.005 * max_collective_n,
        }
    }

    /// CTBR command (ENU/FLU) + measured body rate → per-motor normalized
    /// commands `d ∈ [0, 1]`.
    ///
    /// The demanded torque is `I·α_des`, dropping the `ω × Iω` term the
    /// plant carries. That omission is deliberate and matches what the
    /// training environment's own allocation did: the gyroscopic coupling
    /// is a disturbance the proportional loop rejects, not a term the
    /// policy was trained to have cancelled for it.
    pub fn command(
        &self,
        cmd: &CtbrCommand,
        body_rate_rad_s: Vector3<f32>,
    ) -> SVector<f32, NU> {
        let alpha = self
            .rate_kp
            .component_mul(&(cmd.body_rate_rad_s - body_rate_rad_s));
        let torque = self.inertia * alpha;
        let thrust_n = (self.mass_kg * cmd.specific_thrust_m_s2)
            .clamp(self.idle_n, self.max_collective_n);
        let fractions = self
            .allocator
            .allocate(Vector4::new(thrust_n, torque.x, torque.y, torque.z));
        self.cmd_map.command(&(fractions * self.per_motor_max_n))
    }
}

/// Controller stack abstraction. The runner ticks `step()` at `tick_rate_hz`
/// and supplies a fresh horizon of `horizon_samples` setpoints sampled at
/// stride `horizon_stride_s` starting from the current sim time.
pub trait Controller {
    fn name(&self) -> &'static str;

    /// Fast-loop tick rate for this stack. Runner drives `step()` at this rate.
    fn tick_rate_hz(&self) -> f32;

    /// Setpoint samples this controller wants per tick. Default 1.
    fn horizon_samples(&self) -> usize {
        1
    }

    /// Stride [s] between consecutive horizon samples. Default 0.
    fn horizon_stride_s(&self) -> f32 {
        0.0
    }

    /// One tick. Returns per-motor **normalized commands** `d ∈ [0, 1]` —
    /// the ESC-bound values, not forces.
    ///
    /// `x` is the 13-state the controller sees: plant ground truth, or ESKF
    /// output when the scenario has a GPS model attached. `imu` is the
    /// scenario-provided IMU measurement; ground-truth controllers ignore
    /// it, INDI consumes it. `rotor` is ESC telemetry — INDI feeds it to
    /// its `RpmTracker` to enable the G2 (rotor-reaction) columns; stacks
    /// without an inner loop ignore it.
    fn step(
        &mut self,
        x: &SVector<f32, NX>,
        imu: &ImuMeasurement,
        rotor: &RotorTelemetry,
        horizon: &[Setpoint],
    ) -> SVector<f32, NU>;

    /// `(solve_count, total_wall_time)` of the outer-loop optimiser so far,
    /// for stacks that have one. Host wall-clock, single-threaded — a
    /// relative compute-cost measure, not an MCU estimate.
    fn solve_stats(&self) -> Option<(u64, std::time::Duration)> {
        None
    }
}

// Sim's own cadence — deliberately NOT wired to the `mpc_rate_hz` param
// (firmware defaults to 50 Hz): the regression snapshot froze this stack
// at 100 Hz before that param existed, same freeze policy as the rest of
// `sim_baseline.yaml`.
const MPC_SOLVE_RATE_HZ: f32 = 100.0;
const INDI_LOOP_HZ: f32 = 8000.0;
const SIMPLE_MPC_DT: f32 = 0.05;
const FULL_MPC_DT: f32 = 0.05;

// ───────────────────────────────────────────────────────────────────────────
// MPC + INDI stack (default — matches firmware topology)
// ───────────────────────────────────────────────────────────────────────────

/// 10-state MPC at 100 Hz + INDI inner loop at 8 kHz.
///
/// Outer MPC emits `u = [thrust_N, wx_sp, wy_sp, wz_sp]`. INDI consumes
/// `(rate_sp, thrust_sp / mass)` and the synthesized body-frame specific
/// force, and emits per-motor normalized throttles. Plant integrates
/// `per_motor_max_N * throttle` every tick.
pub struct MpcIndiController {
    pub solve_count: u64,
    pub solve_time: std::time::Duration,
    solver: Box<SimpleSqpSolver>,
    problem: SimpleQuadProblem,
    x_refs: [SVector<f32, SIMPLE_NX>; SIMPLE_N + 1],
    u_refs: [SVector<f32, SIMPLE_NU>; SIMPLE_N],
    u_warm: [SVector<f32, SIMPLE_NU>; SIMPLE_N],
    last_mpc_u: SVector<f32, SIMPLE_NU>,
    /// SQP iteration budget / KKT tolerance from the param plane
    /// (`mpc_max_iters` / `mpc_kkt_tol`), mirroring the firmware loop.
    max_iters: usize,
    kkt_tol: f32,
    indi: IndiController,
    /// Inner-loop (INDI) tick rate [Hz]. 8 kHz by default; the 1 kHz-IMU
    /// vehicles fly INDI at 500 Hz (`indi_ctrl_div 2`).
    indi_rate_hz: f32,
    mpc_stride: u32,
    tick_counter: u32,
    mass: f32,
    grav: f32,
    /// Mirrors the firmware's `motor_omega_filter` + finite-difference
    /// pair (`indi_task.rs`): a 15 Hz biquad on measured rotor speed,
    /// differentiated at loop rate. INDI needs both, and they must carry
    /// the same group delay as `rate_dot_fs` / `spf_fs` or the
    /// incremental `dv = sp − fs` is differenced across mismatched times.
    omega_filter: [Biquad; INDI_NU],
    omega_fs: SVector<f32, INDI_NU>,
    omega_dot_fs: SVector<f32, INDI_NU>,
    omega_hold: SVector<f32, INDI_NU>,
    omega_fs_has_prev: bool,
    erpm_to_rads: f32,
    /// Nominal pack voltage handed to INDI's thrust linearization. Only
    /// `ThrustModel::Table` reads this; the analytic models ignore it.
    /// 23.0 V is mid-pack 6S, matching the bench thrust map at
    /// `tmp/thrust_map/a2rl_0114.csv`. Sim plant has no battery sag model.
    nominal_voltage_v: f32,
    // ── Firmware reference-chain fidelity switches (all default to the
    // frozen sim behaviour so the regression snapshot is unaffected) ──
    /// `q_ref` via the tilt-then-yaw map (`quaternion_from_zb_and_yaw`),
    /// which is what the firmware uses for `flatness_map: tilt_yaw`
    /// missions. Default on (`USE_TILT_REFERENCE_QUATERNION`); `false`
    /// is the legacy cross-product map, singular at 90° tilt.
    pub use_tilt_map: bool,
    /// Firmware `USE_FLATNESS_U_REF_FEEDFORWARD`: `u_refs[k] =
    /// [m·‖α‖, ω]` from the flatness map instead of the hover input.
    /// Default on, as in the firmware.
    pub flatness_feedforward: bool,
    /// ω source for the feedforward. `true` (default, = firmware
    /// `outer_loop.rs`): the closed-form tilt-yaw map
    /// (`flatness_to_thrust_omega_tilt_yaw`, the angular velocity of the
    /// tilt `q_ref` itself, including the `(1−cos θ)·φ̇` body-z term).
    /// `false`: the min-norm ω from `flatness_to_thrust_omega`, whose
    /// body-z is zero by construction — kept as the A/B control
    /// (`tests/omega_ref_ab.rs`). Only read when `flatness_feedforward`
    /// is on.
    pub closed_form_omega_ref: bool,
    /// Firmware `sampler_kind: Position`: when attached, the horizon is
    /// re-sampled every solve by a `PositionSampler` (closest point on the
    /// trajectory within the `[τ₀ − max_lag, τ₀ + max_lead]` trust window)
    /// instead of the runner's time-indexed horizon. `τ₀` is the elapsed
    /// mission time from the controller's own tick count (the sim mission
    /// starts at t = 0 when no estimator hold is in the loop).
    position_sampler: Option<SimPositionSampler>,
    // ── Situation-conditioned cost adaptation (docs/learned_mpc_cost.md) ──
    /// Hand-tuned weights captured at construction; `set_cost_z` modulates
    /// the live model relative to these.
    cost_nominal: CostNominal,
    /// Horizon nodes of the last `fill_reference`, kept for the
    /// observation's maneuver preview.
    last_nodes: [SamplerNode; SIMPLE_N + 1],
    /// Assemble [`Self::situation`] at every solve. Off by default (the
    /// snapshot path never pays for it).
    pub observe_situation: bool,
    /// Apply the policy's output one solve LATE: the firmware's
    /// latency-first ordering runs the policy after publishing the
    /// command, so the weights used at solve k were computed from solve
    /// k−1's situation. Off = the historical immediate application.
    pub cost_policy_delayed: bool,
    /// The pending `z` computed at the previous solve, applied at the
    /// next one when `cost_policy_delayed` is set.
    pending_cost_z: [f32; COST_NZ],
    /// Observation assembled at the most recent solve (before that solve
    /// ran), valid when `observe_situation` is set or a policy is loaded.
    pub situation: [f32; COST_OBS_DIM],
    /// The `z` in effect for the most recent solve.
    pub cost_z: [f32; COST_NZ],
    /// Embedded policy: `situation → z`, applied before every solve.
    cost_policy: Option<OwnedCostPolicy>,
    last_diverged: bool,
    /// Per-solve `(z, situation)` log, filled only when `log_cost` is set.
    pub cost_log: Vec<([f32; COST_NZ], [f32; COST_OBS_DIM])>,
    pub log_cost: bool,
}

/// A cost policy that owns its weights (the firmware borrows a baked
/// slice; the sim loads a checkpoint file).
pub struct OwnedCostPolicy {
    pub weights: Vec<f32>,
    pub shapes: Vec<LayerShape>,
}

impl OwnedCostPolicy {
    /// Load `tools/train_cost_policy.py`'s export: little-endian
    /// `u32 magic` (`CFCP`), `u32 format`, `u32 map_version`,
    /// `u32 n_layers`, then `n_layers × (u32 inputs, u32 outputs)`, then
    /// the `f32` weights, per layer `W(out, in)` row-major followed by
    /// `b(out)` — the `nn::Mlp` layout.
    ///
    /// The map revision is checked here for the same reason the firmware
    /// bake checks it: nothing in the shapes distinguishes a checkpoint
    /// trained under a different weight map, and the sim is where a
    /// checkpoint's numbers get judged before it is ever flown.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        const MAGIC: usize = 0x5043_4643; // b"CFCP" little-endian
        const FORMAT: usize = 1;
        let bytes = std::fs::read(path)?;
        let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, "policy file truncated");
        let invalid = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
        let u32_at = |i: usize| -> std::io::Result<usize> {
            let b = bytes.get(i..i + 4).ok_or_else(bad)?;
            Ok(u32::from_le_bytes(b.try_into().unwrap()) as usize)
        };
        if u32_at(0)? != MAGIC {
            return Err(invalid(
                "policy file: not a cost-policy export, or predates the versioned header                  — re-export it with tools/train_cost_policy.py"
                    .into(),
            ));
        }
        if u32_at(4)? != FORMAT {
            return Err(invalid(format!(
                "policy file: export format {}, this build reads {FORMAT}",
                u32_at(4)?
            )));
        }
        let map_version = u32_at(8)? as u32;
        if map_version != COST_MAP_VERSION {
            return Err(invalid(format!(
                "policy file: trained against cost-map revision {map_version}, this build                  computes weights with revision {COST_MAP_VERSION} — the same z means                  something different under the two maps"
            )));
        }
        let n = u32_at(12)?;
        if n == 0 || n > 16 {
            return Err(bad());
        }
        let mut shapes = Vec::with_capacity(n);
        let mut off = 16;
        for _ in 0..n {
            shapes.push(LayerShape::new(u32_at(off)? as u16, u32_at(off + 4)? as u16));
            off += 8;
        }
        let weights: Vec<f32> = bytes[off..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let expect: usize = shapes.iter().map(|s| s.weight_count()).sum();
        if weights.len() != expect {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("policy file: {} weights, shapes need {expect}", weights.len()),
            ));
        }
        Ok(Self { weights, shapes })
    }
}

/// Controller-side position sampler state (see
/// [`MpcIndiController::attach_position_sampler`]).
struct SimPositionSampler {
    sampler: PositionSampler,
    traj: PiecewisePolynomial,
    total_duration_s: f32,
    nodes: [SamplerNode; SIMPLE_N + 1],
}

/// Cutoff for the ω / ω̇ filter feeding `MotorState::External`.
///
/// Mirrors the firmware, which derives it from the controller's effective
/// sync cutoff (`IndiController::effective_sync_filter_hz`) so ω carries the
/// same group delay as every other signal INDI compares against. Both used
/// to pin 15 Hz here against a 12 Hz sync filter — a 3.8 ms delay mismatch
/// on the ω̇ term, invisible to tuning.
fn make_omega_biquad(loop_rate_hz: f32, cutoff_hz: f32) -> Biquad {
    let cfg = BiquadFilterConfigBuilder::direct_form_2()
        .sample_frequency_hz(loop_rate_hz)
        .filter_type(BiquadFilterType::LowPass)
        .cutoff_frequency_hz(cutoff_hz)
        .build()
        .expect("sim indi: biquad filter config invalid");
    BiquadFilter::new(cfg)
}

/// Build the sim-side `IndiConfig` from vehicle params — the same values
/// the firmware `indi_task` derives. Shared by `MpcIndiController` and
/// `MpcFullIndiController` so the two stacks run byte-identical inner
/// loops. `indi_enabled = false` selects the degraded static-inversion
/// inner loop (`build: indi: no` analogue).
fn build_indi_config(vp: &FirmwareConfig, indi_enabled: bool) -> IndiConfig {
    let ic = &vp.indi.controller;
    // Motor dynamics come from the vehicle definition instead of
    // sim-local literals. They are the same registry keys the firmware
    // INDI task reads (`m*_tau`, `m*_omega_max`, `m*_g2_ry`,
    // `m*_nonlin`), so a retune moves the plant and the controller
    // together unless a scenario deliberately splits them.
    let indi_motors: [IndiMotorParams; INDI_NU] = core::array::from_fn(|i| {
        let m = &vp.airframe.motors[i];
        let tau = m.time_const_s;
        let omega = m.max_omega_rad_s;
        IndiMotorParams {
            time_const_s: if tau.is_finite() && tau > 1e-4 { tau } else { 0.02 },
            max_rpm: if omega.is_finite() && omega > 1.0 {
                omega * 60.0 / core::f32::consts::TAU
            } else {
                40000.0
            },
            g2_yaw: if m.g2[2].is_finite() { m.g2[2] } else { 0.0 },
        }
    });
    // `0.0` is the registry's "unset" sentinel; fall back to the
    // near-linear curve the sim used before the plant had a curve at all.
    let nonlinearity = SVector::<f32, INDI_NU>::from_fn(|i, _| {
        let k = vp.airframe.motors[i].nonlinearity;
        if k.is_finite() && k > 0.0 { k } else { 0.025 }
    });
    IndiConfig {
        indi_enabled,
        // Frozen sim baseline, matching the firmware defaults.
        ground_gyro_rad_s: 100.0_f32 * core::f32::consts::PI / 180.0,
        ground_accel_m_s2: 0.8 * 9.81,
        ground_thrust_sp_m_s2: 3.0,
        rate_gains: Vector3::new(ic.rate_gains[0], ic.rate_gains[1], ic.rate_gains[2]),
        sync_filter_hz: ic.sync_filter_hz,
        rate_dot_sg_window_size: 13,
        rate_dot_sg_order: 2,
        motors: vp.airframe.motors,
        body: vp.airframe.body,
        indi_motors,
        thrust_model: ThrustModel::Quadratic,
        nonlinearity,
        act_limit: SVector::from_element(1.0),
        wls_wv: SVector::<f32, NV>::from_row_slice(&ic.wls_wv),
        wls_wu: SVector::<f32, INDI_NU>::from_row_slice(&ic.wls_wu),
        wls_cond_bound: 3.2768e8,
        wls_theta: 1e-4,
        wls_imax: 1,
        nan_limit: 20,
        // Frozen with the rest of this baseline: the sim must not drift
        // when the schema default is retuned.
        nan_rampdown: 0.95,
        rpm_invalid_limit: 50,
        rpm_all_invalid_limit: 50,
        rpm_recovery_count: 10,
        motor_pole_count: vp.airframe.motor_pole_count,
    }
}

impl MpcIndiController {
    pub fn from_params(vp: &FirmwareConfig) -> Self {
        Self::from_params_with_mode(vp, vp.mpc.pos_cost_mode)
    }

    /// Same as [`Self::from_params`] with an explicit INDI tick rate
    /// (e.g. 500 Hz for the 1 kHz-IMU vehicles).
    pub fn from_params_at_indi_rate(vp: &FirmwareConfig, indi_rate_hz: f32) -> Self {
        Self::with_options(
            vp,
            vp.mpc.pos_cost_mode,
            MPC_SOLVE_RATE_HZ,
            indi_rate_hz,
            (vp.mpc.horizon_n as usize).clamp(1, SIMPLE_N),
        )
    }

    /// Build the controller with an explicit [`PosCostMode`] override that
    /// supersedes `vp.mpc.pos_cost_mode`. Useful for sim sweeps that
    /// compare Quadratic vs. Contouring on the same `FirmwareConfig`
    /// without having to mutate the param struct between runs.
    pub fn from_params_with_mode(vp: &FirmwareConfig, pos_cost_mode: PosCostMode) -> Self {
        Self::with_options(
            vp,
            pos_cost_mode,
            MPC_SOLVE_RATE_HZ,
            INDI_LOOP_HZ,
            (vp.mpc.horizon_n as usize).clamp(1, SIMPLE_N),
        )
    }

    /// `mpc_rate_hz` is the outer solve cadence (the firmware's
    /// `mpc_rate_hz`); the horizon spacing stays `SIMPLE_MPC_DT`.
    /// `horizon_n` is the SQP horizon in stages (`≤ SIMPLE_N`, the
    /// solver's workspace capacity).
    pub fn with_options(
        vp: &FirmwareConfig,
        pos_cost_mode: PosCostMode,
        mpc_rate_hz: f32,
        indi_rate_hz: f32,
        horizon_n: usize,
    ) -> Self {
        assert!(
            (1..=SIMPLE_N).contains(&horizon_n),
            "SQP horizon {horizon_n} exceeds workspace capacity {SIMPLE_N}"
        );
        // Outer MPC (QuadModel, same as firmware outer_loop.rs)
        let mut model = QuadModel::from_vehicle_params(vp);
        model.dt = SIMPLE_MPC_DT;
        // Sim-local override: restore the pre-7b7bf81 uniform `w_input`
        // mapping. Upstream `QuadModel::from_vehicle_params` was changed
        // from `w_input = SVector::from_element(thrust_weight)` to
        // `w_input = [thrust_weight, rate_weight[0..3]]` — semantically
        // correct (the four MPC inputs are heterogeneous: collective
        // thrust + 3 body rates), but the change broke the sim's
        // committed regression snapshot. We re-apply the uniform shape
        // *only* in this controller; FullQuadModel (used by
        // `MpcDirectController`) keeps the canonical `w_rate` mapping.
        model.w_input = SVector::from_element(vp.mpc.thrust_weight);
        model.pos_cost_mode = pos_cost_mode;
        let grav = model.grav;
        let problem = SimpleQuadProblem::with_rk4(model, horizon_n);
        let mass = vp.airframe.body.mass_kg;

        let hover_thrust_n = mass * grav;
        let u_ref = vector![hover_thrust_n, 0.0, 0.0, 0.0];

        let mut x_ref = SVector::<f32, SIMPLE_NX>::zeros();
        x_ref[6] = 1.0;

        // Inner INDI (same config as firmware indi_task.rs). The sim's
        // firmware-match stack always exercises the incremental law;
        // `build: indi: no` is exercised via `MpcFullIndiController`'s
        // static-inversion variant instead.
        let indi_cfg = build_indi_config(vp, true);
        let indi = IndiController::new(&indi_cfg, indi_rate_hz);
        // Take the cutoff from the controller, not from `ic.sync_filter_hz`,
        // so the sim's ω filter follows the same Nyquist clamp the firmware's
        // does — one source for the shared group delay.
        let indi_sync_hz = indi.effective_sync_filter_hz();

        let mpc_stride = (indi_rate_hz / mpc_rate_hz).round().max(1.0) as u32;
        let pole_pairs = (vp.airframe.motor_pole_count.max(2) as f32) / 2.0;
        let erpm_to_rads = core::f32::consts::TAU * 100.0 / (pole_pairs * 60.0);
        let cost_nominal = CostNominal::from_model(&problem.model);

        Self {
            solve_count: 0,
            solve_time: std::time::Duration::ZERO,
            solver: Box::new(SimpleSqpSolver::new()),
            problem,
            x_refs: [x_ref; SIMPLE_N + 1],
            u_refs: [u_ref; SIMPLE_N],
            u_warm: [u_ref; SIMPLE_N],
            last_mpc_u: u_ref,
            max_iters: vp.mpc.max_iters as usize,
            kkt_tol: vp.mpc.kkt_tol,
            indi,
            indi_rate_hz,
            mpc_stride,
            tick_counter: 0,
            mass,
            grav,
            omega_filter: core::array::from_fn(|_| {
                make_omega_biquad(indi_rate_hz, indi_sync_hz)
            }),
            omega_fs: SVector::zeros(),
            omega_dot_fs: SVector::zeros(),
            omega_hold: SVector::zeros(),
            omega_fs_has_prev: false,
            erpm_to_rads,
            nominal_voltage_v: 23.0,
            use_tilt_map: USE_TILT_REFERENCE_QUATERNION,
            flatness_feedforward: true,
            closed_form_omega_ref: true,
            position_sampler: None,
            cost_nominal,
            last_nodes: [SamplerNode::default(); SIMPLE_N + 1],
            observe_situation: false,
            situation: [0.0; COST_OBS_DIM],
            cost_z: [0.0; COST_NZ],
            cost_policy_delayed: false,
            pending_cost_z: [0.0; COST_NZ],
            cost_policy: None,
            last_diverged: false,
            cost_log: Vec::new(),
            log_cost: false,
        }
    }

    /// The hand-tuned weights the adaptive cost modulates.
    pub fn cost_nominal(&self) -> CostNominal {
        self.cost_nominal
    }

    /// Fix the input weights `[thrust, ωx, ωy, ωz]` (the canonical
    /// `mpc_w_thrust` / `mpc_w_rate_*` mapping; the constructor's uniform
    /// override exists only for the frozen snapshot). Not part of the
    /// learned set.
    pub fn set_input_weights(&mut self, w: [f32; SIMPLE_NU]) {
        self.problem.model.w_input = SVector::<f32, SIMPLE_NU>::from_row_slice(&w);
        // Only the rate part of the nominal follows: recapturing the whole
        // nominal from the live model would bake a prior `set_cost_z`
        // modulation into it and compound on the next `apply`.
        self.cost_nominal.w_rate = [w[1], w[2], w[3]];
    }

    /// Set the cost residual used by every solve from now on
    /// (`w = w_nom·4^z`, see `cost_adapt::CostNominal::apply`).
    pub fn set_cost_z(&mut self, z: &[f32; COST_NZ]) {
        self.cost_z = *z;
        self.cost_nominal.apply(z, &mut self.problem.model);
    }

    /// Embed a trained cost policy: the controller then assembles the
    /// situation observation and applies the policy's `z` before every
    /// solve, which is the flight configuration.
    pub fn set_cost_policy(&mut self, policy: OwnedCostPolicy) {
        // Validate once so a bad checkpoint fails at load, not mid-flight.
        CostPolicy::new(&policy.weights, &policy.shapes).expect("cost policy shape mismatch");
        self.observe_situation = true;
        self.cost_policy = Some(policy);
    }

    /// The live model's weights (for logging / probes).
    pub fn model(&self) -> &QuadModel {
        &self.problem.model
    }

    /// Put the identified rotor-drag term into the prediction model
    /// (`coeff` = `sim: aero_drag`, `c_t` = `max_thrust_n / ω_max²`).
    pub fn set_rotor_drag(&mut self, coeff: [f32; 3], c_t: f32) {
        // Model physics only; the cost nominal is untouched.
        self.problem.model.set_rotor_drag(coeff, c_t);
    }

    /// Put the identified quadratic body drag (`½ρC_dA` per axis) into the
    /// prediction model.
    pub fn set_body_drag(&mut self, coeff: [f32; 3]) {
        self.problem.model.set_body_drag(coeff);
    }

    /// Drive the horizon from a firmware-style `PositionSampler` over
    /// `traj` (the same trajectory the runner samples by time). Parameters
    /// normally come from `vp.trajectory.sampler.to_position_sampler_params()`.
    pub fn attach_position_sampler(&mut self, traj: PiecewisePolynomial, params: PositionSamplerParams) {
        let total_duration_s = traj.total_duration();
        self.position_sampler = Some(SimPositionSampler {
            sampler: PositionSampler::new(params),
            traj,
            total_duration_s,
            nodes: [SamplerNode::default(); SIMPLE_N + 1],
        });
    }

    /// One horizon node in the form the reference builder consumes.
    #[inline]
    fn node_from_setpoint(sp: &Setpoint) -> SamplerNode {
        SamplerNode {
            pos: sp.position,
            vel: sp.velocity,
            acc: sp.acceleration,
            jerk: sp.jerk,
            past_end: sp.terminal,
        }
    }

    /// `x_ref` / `u_ref` for one node, mirroring the firmware's per-node
    /// fan-out (`outer_loop.rs`): past-end nodes get identity attitude and
    /// the hover input; otherwise `q_ref` from the selected map and, with
    /// `flatness_feedforward`, `u_ref` from the pole-safe flatness map
    /// (falling back to `[m·‖α‖, 0]` on a fault).
    fn node_reference(&self, n: &SamplerNode) -> (SVector<f32, SIMPLE_NX>, SVector<f32, SIMPLE_NU>) {
        let grav = self.grav;
        let hover = vector![self.mass * grav, 0.0, 0.0, 0.0];
        if n.past_end {
            let x = stack![n.pos; UnitQuaternion::<f32>::identity().coords; Vector3::<f32>::zeros()];
            return (x, hover);
        }
        let q_ref = q_ref_with_map(n.acc, 0.0, grav, self.use_tilt_map);
        let x = stack![n.pos; q_ref.coords; n.vel];
        let u = if self.flatness_feedforward {
            // Closed-form ω needs no snap (snap only enters ω̇), so a zero
            // snap gives the exact ω; see `omega_ref_ab.rs`.
            let flat = if self.closed_form_omega_ref {
                flatness_to_thrust_omega_tilt_yaw(n.acc, n.jerk, 0.0, 0.0, grav)
            } else {
                flatness_to_thrust_omega(n.acc, n.jerk, 0.0, 0.0, grav)
            };
            let u = match flat {
                Ok((alpha_norm, _att, omega)) => vector![self.mass * alpha_norm, omega.x, omega.y, omega.z],
                Err(_) => {
                    let alpha = Vector3::new(n.acc.x, n.acc.y, n.acc.z + grav).norm();
                    vector![self.mass * alpha, 0.0, 0.0, 0.0]
                }
            };
            let b = self.problem.model.u_bounds;
            SVector::<f32, SIMPLE_NU>::from_fn(|i, _| u[i].clamp(b[i][0], b[i][1]))
        } else {
            hover
        };
        (x, u)
    }

    /// Build `x_refs[k]` / `u_refs[k]` from one node.
    fn write_node_reference(&mut self, k: usize, n: &SamplerNode) {
        let (x, u) = self.node_reference(n);
        self.x_refs[k] = x;
        if k < SIMPLE_N {
            self.u_refs[k] = u;
        }
    }

    /// The situation observation the *next* solve will see, computed
    /// without side effects (the position sampler is `Copy`, so its
    /// closest-point state is advanced on a scratch copy). Used by the
    /// training environment to hand the policy the observation its
    /// action will apply to.
    pub fn peek_situation(&self, x_full: &SVector<f32, NX>, horizon: &[Setpoint]) -> [f32; COST_OBS_DIM] {
        let n = self.problem.n;
        let mut nodes = [SamplerNode::default(); SIMPLE_N + 1];
        if let Some(ps) = self.position_sampler.as_ref() {
            let mut sampler = ps.sampler;
            let inputs = SamplerInputs {
                traj: &ps.traj,
                total_duration_s: ps.total_duration_s,
                tau0_s: self.tick_counter as f32 / self.indi_rate_hz,
                state_pos: Vector3::new(x_full[0], x_full[1], x_full[2]),
                horizon_dt: SIMPLE_MPC_DT,
            };
            let _ = sampler.sample(&inputs, &mut nodes[..=n]);
        } else {
            for k in 0..=n {
                nodes[k] = match horizon.get(k.min(horizon.len().saturating_sub(1))) {
                    Some(sp) => Self::node_from_setpoint(sp),
                    None => SamplerNode::default(),
                };
            }
        }
        let mut x_refs = [SVector::<f32, SIMPLE_NX>::zeros(); SIMPLE_N + 1];
        for k in 0..=n {
            x_refs[k] = self.node_reference(&nodes[k]).0;
        }
        for k in n + 1..=SIMPLE_N {
            x_refs[k] = x_refs[n];
            nodes[k] = nodes[n];
        }
        let x0: SVector<f32, SIMPLE_NX> = x_full.fixed_rows::<SIMPLE_NX>(0).into_owned();
        let inp = SituationInputs {
            x0: &x0,
            body_rate: Vector3::new(x_full[10], x_full[11], x_full[12]),
            x_refs: &x_refs,
            nodes: &nodes,
            last_u: &self.last_mpc_u,
            u_bounds: self.problem.model.u_bounds,
            mass: self.mass,
            grav: self.grav,
        };
        let mut obs = [0.0f32; COST_OBS_DIM];
        situation_obs(&inp, &mut obs);
        obs
    }

    /// Feed one ESC telemetry frame through the same two paths the
    /// firmware uses: the `RpmTracker` (which owns G2 validity, dropout
    /// counting and recovery hysteresis) and the 15 Hz biquad +
    /// finite-difference pair that produces `omega_fs` / `omega_dot_fs`.
    ///
    /// On a dropped frame the last valid *input* is held rather than the
    /// filter output — feeding the output back would close a marginally
    /// stable loop that drifts under sustained telemetry loss.
    fn update_rotor(&mut self, rotor: &RotorTelemetry) -> [bool; INDI_NU] {
        let inputs: [RpmInput; INDI_NU] = core::array::from_fn(|i| {
            if !rotor.valid[i] || !rotor.omega_rad_s[i].is_finite() {
                RpmInput::Invalid
            } else if rotor.omega_rad_s[i] <= 0.0 {
                RpmInput::Stopped
            } else {
                RpmInput::Erpm((rotor.omega_rad_s[i] / self.erpm_to_rads).round() as u32)
            }
        });
        let (g2_valid, _rpm_failsafe) = self.indi.update_rpm(&inputs);

        for i in 0..INDI_NU {
            let x = if rotor.valid[i] && rotor.omega_rad_s[i].is_finite() {
                self.omega_hold[i] = rotor.omega_rad_s[i];
                rotor.omega_rad_s[i]
            } else {
                self.omega_hold[i]
            };
            let new_fs = self.omega_filter[i].apply(x);
            self.omega_dot_fs[i] = if self.omega_fs_has_prev {
                (new_fs - self.omega_fs[i]) * self.indi_rate_hz
            } else {
                0.0
            };
            self.omega_fs[i] = new_fs;
        }
        self.omega_fs_has_prev = true;
        g2_valid
    }

    fn fill_reference(&mut self, horizon: &[Setpoint], state_pos: Vector3<f32>) {
        let n = self.problem.n;
        if let Some(ps) = self.position_sampler.as_mut() {
            let tau0_s = self.tick_counter as f32 / self.indi_rate_hz;
            let inputs = SamplerInputs {
                traj: &ps.traj,
                total_duration_s: ps.total_duration_s,
                tau0_s,
                state_pos,
                horizon_dt: SIMPLE_MPC_DT,
            };
            let _ = ps.sampler.sample(&inputs, &mut ps.nodes[..=n]);
            let nodes = ps.nodes;
            self.last_nodes = nodes;
            for k in 0..=n {
                self.write_node_reference(k, &nodes[k]);
            }
            return;
        }
        debug_assert!(horizon.len() == n + 1);
        for k in 0..=n {
            let node = Self::node_from_setpoint(&horizon[k]);
            self.last_nodes[k] = node;
            self.write_node_reference(k, &node);
        }
    }

    /// Assemble the situation observation from the references just
    /// built, then (with a policy loaded) modulate the cost for this
    /// solve. Runs between `fill_reference` and the solve.
    fn adapt_cost(&mut self, x_full: &SVector<f32, NX>, x0: &SVector<f32, SIMPLE_NX>) {
        if !self.observe_situation {
            return;
        }
        let n = self.problem.n;
        let inp = SituationInputs {
            x0,
            body_rate: Vector3::new(x_full[10], x_full[11], x_full[12]),
            x_refs: &self.x_refs[..=n],
            nodes: &self.last_nodes[..=n],
            last_u: &self.last_mpc_u,
            u_bounds: self.problem.model.u_bounds,
            mass: self.mass,
            grav: self.grav,
        };
        let mut obs = [0.0f32; COST_OBS_DIM];
        situation_obs(&inp, &mut obs);
        self.situation = obs;
        if let Some(p) = self.cost_policy.as_ref() {
            let policy = CostPolicy::new(&p.weights, &p.shapes).expect("validated at load");
            let mut z = [0.0f32; COST_NZ];
            policy.act(&obs, &mut z);
            if self.cost_policy_delayed {
                // Firmware ordering: this solve runs on the z computed
                // from the PREVIOUS solve's situation; the fresh z waits
                // one solve. `cost_log` records the applied z with the
                // current observation (one-solve offset by design).
                let applied = self.pending_cost_z;
                self.pending_cost_z = z;
                self.set_cost_z(&applied);
            } else {
                self.set_cost_z(&z);
            }
        }
        if self.log_cost {
            self.cost_log.push((self.cost_z, obs));
        }
    }

    fn solve_mpc(&mut self, x_full: &SVector<f32, NX>, horizon: &[Setpoint]) {
        // Extract the 10-state slice (position, quaternion, velocity) from
        // the plant's 13-state. The remaining 3 (body rates) are not part of
        // the simple MPC's state.
        let x0: SVector<f32, SIMPLE_NX> = x_full.fixed_rows::<SIMPLE_NX>(0).into_owned();
        self.fill_reference(horizon, Vector3::new(x_full[0], x_full[1], x_full[2]));
        self.adapt_cost(x_full, &x0);

        let t0 = std::time::Instant::now();
        let bounds = self.problem.model.u_bounds;
        let res = self.solver.solve(
            &self.problem,
            &x0,
            &self.x_refs,
            &self.u_refs,
            &self.u_warm,
            self.max_iters,
            self.kkt_tol,
        );
        self.solve_time += t0.elapsed();
        self.solve_count += 1;
        self.last_diverged = res.diverged;
        let u_bar = self.solver.u_bar();
        let u0 = u_bar[0];
        self.last_mpc_u =
            SVector::<f32, SIMPLE_NU>::from_fn(|i, _| u0[i].clamp(bounds[i][0], bounds[i][1]));
        self.u_warm = *u_bar;
    }

    /// Body-rate / thrust command of the last solve (`[N, rad/s×3]`).
    pub fn last_command(&self) -> SVector<f32, SIMPLE_NU> {
        self.last_mpc_u
    }

    /// Whether the last solve reported `diverged`.
    pub fn last_diverged(&self) -> bool {
        self.last_diverged
    }
}

impl Controller for MpcIndiController {
    fn solve_stats(&self) -> Option<(u64, std::time::Duration)> {
        Some((self.solve_count, self.solve_time))
    }
    fn name(&self) -> &'static str {
        "mpc_indi"
    }

    fn tick_rate_hz(&self) -> f32 {
        self.indi_rate_hz
    }

    fn horizon_samples(&self) -> usize {
        self.problem.n + 1
    }

    fn horizon_stride_s(&self) -> f32 {
        SIMPLE_MPC_DT
    }

    fn step(
        &mut self,
        x: &SVector<f32, NX>,
        imu: &ImuMeasurement,
        rotor: &RotorTelemetry,
        horizon: &[Setpoint],
    ) -> SVector<f32, NU> {
        // Outer MPC solve every `mpc_stride` ticks.
        if self.tick_counter % self.mpc_stride == 0 {
            self.solve_mpc(x, horizon);
        }
        self.tick_counter = self.tick_counter.wrapping_add(1);
        self.step_indi(imu, rotor)
    }
}

impl MpcIndiController {
    /// Inner-loop tick: track `last_mpc_u` (collective thrust + body-rate
    /// setpoint) with INDI. Shared with [`TinyMpcIndiController`], which
    /// swaps only the outer solver.
    fn step_indi(&mut self, imu: &ImuMeasurement, rotor: &RotorTelemetry) -> SVector<f32, NU> {
        let thrust_sp_n = self.last_mpc_u[0];
        let rate_sp = Vector3::new(self.last_mpc_u[1], self.last_mpc_u[2], self.last_mpc_u[3]);
        // Collective thrust setpoint expressed as specific force on body-z.
        let spf_sp_z = thrust_sp_n / self.mass;

        let g2_valid = self.update_rotor(rotor);
        let any_valid = g2_valid.iter().any(|v| *v);
        let motor_state = if any_valid && self.omega_fs_has_prev {
            MotorState::External {
                omega_fs: &self.omega_fs,
                omega_dot_fs: &self.omega_dot_fs,
            }
        } else {
            MotorState::Internal
        };

        let (out, _step_state) = self.indi.step(
            &imu.gyro,
            &imu.accel,
            &rate_sp,
            spf_sp_z,
            true,
            &g2_valid,
            motor_state,
            self.nominal_voltage_v,
        );

        // Straight through to the plant — these are the ESC-bound values.
        out.motor_commands
    }
}

// ───────────────────────────────────────────────────────────────────────────
// TinyMPC + INDI (linear-MPC counterpart of MpcIndiController)
// ───────────────────────────────────────────────────────────────────────────

/// TinyMPC horizon and cadence: N knot points spaced at the solve period,
/// solved every tick — the paper's figure-eight experiment (Nguyen et al.
/// 2024 §V-C: "TinyMPC ran at 500 Hz with a horizon length of N = 15") and
/// the reference implementation's convention (`params_<f>hz.h` bake
/// `dt = 1/f` into `(A, B)`).
const TINY_N: usize = 15;
const TINY_SOLVE_RATE_HZ: f32 = 500.0;
const TINY_DT: f32 = 1.0 / TINY_SOLVE_RATE_HZ;

/// TinyMPC (ADMM, hover-linearised 9-state model) at `TINY_SOLVE_RATE_HZ`
/// / `TINY_N` + the same INDI inner loop as [`MpcIndiController`].
///
/// Outer solver emits `u = [thrust_N, wx_sp, wy_sp, wz_sp]` exactly like the
/// SQP stack; the differences are the optimiser (fixed `(A, B)` about
/// hover, cached LQR, ADMM box constraints) versus the nonlinear SQP, and
/// the horizon (`TINY_N` × 2 ms versus 20 × 50 ms). Cost weights are the
/// vehicle's `mpc.*` weights mapped onto the `[p, θ, v]` error state and
/// `[δc, ω]` inputs; `rho` and the ADMM settings are TinyMPC's own.
pub struct TinyMpcIndiController {
    pub solve_count: u64,
    pub solve_time: std::time::Duration,
    inner: MpcIndiController,
    tiny: Box<TinyMpc<HOVER_NX, HOVER_NU, TINY_N>>,
    /// Inner-loop ticks per TinyMPC solve.
    stride: u32,
    hover_thrust_n: f32,
    /// Feed the reference collective thrust `m·‖a_ref + g‖` and the
    /// body rates implied by consecutive `q_ref` samples as `u_ref`
    /// (TinyMPC's `Uref`). `false` (default) leaves `u_ref = 0` — hover
    /// relative — as in the reference repository's quadrotor examples.
    pub reference_feedforward: bool,
    pub last_iters: usize,
    pub last_converged: bool,
}

impl TinyMpcIndiController {
    /// ADMM penalty ρ. TinyMPC's quadrotor examples use 5 with position
    /// weights of O(100); the vehicle weights are of the same order.
    pub const DEFAULT_RHO: f32 = 5.0;

    pub fn from_params(vp: &FirmwareConfig) -> Self {
        Self::with_settings(vp, Self::DEFAULT_RHO, TinySettings::default(), INDI_LOOP_HZ)
    }

    pub fn with_settings(
        vp: &FirmwareConfig,
        rho: f32,
        settings: TinySettings,
        indi_rate_hz: f32,
    ) -> Self {
        let inner = MpcIndiController::from_params_at_indi_rate(vp, indi_rate_hz);
        let mass = inner.mass;
        let grav = inner.grav;
        let (a, b) = hover_linear_model(mass, grav, TINY_DT);
        let m = &vp.mpc;
        let q = SVector::<f32, HOVER_NX>::from_row_slice(&[
            m.pos_weight[0],
            m.pos_weight[1],
            m.pos_weight[2],
            m.att_weight[0],
            m.att_weight[1],
            m.att_weight[2],
            m.vel_weight[0],
            m.vel_weight[1],
            m.vel_weight[2],
        ]);
        // Same uniform input weight the sim SQP stack applies (see
        // `MpcIndiController::from_params_with_mode`).
        let r = SVector::<f32, HOVER_NU>::from_element(m.thrust_weight);
        let mut tiny = Box::new(TinyMpc::new(a, b, q, r, rho, settings));
        let bounds = inner.problem.model.u_bounds;
        let hover_thrust_n = mass * grav;
        // Inputs are hover-relative: shift the thrust bound by m·g.
        tiny.set_input_bounds(
            SVector::from_row_slice(&[
                bounds[0][0] - hover_thrust_n,
                bounds[1][0],
                bounds[2][0],
                bounds[3][0],
            ]),
            SVector::from_row_slice(&[
                bounds[0][1] - hover_thrust_n,
                bounds[1][1],
                bounds[2][1],
                bounds[3][1],
            ]),
        );
        Self {
            solve_count: 0,
            solve_time: std::time::Duration::ZERO,
            inner,
            tiny,
            stride: (indi_rate_hz / TINY_SOLVE_RATE_HZ).round().max(1.0) as u32,
            hover_thrust_n,
            reference_feedforward: false,
            last_iters: 0,
            last_converged: false,
        }
    }

    /// Error-state `[p, θ, v]` from a 13-state plant vector.
    fn error_state(x_full: &SVector<f32, NX>) -> SVector<f32, HOVER_NX> {
        let q = UnitQuaternion::from_quaternion(Quaternion::new(
            x_full[6], x_full[3], x_full[4], x_full[5],
        ));
        let theta = q.scaled_axis();
        stack![x_full.fixed_rows::<3>(0); theta; x_full.fixed_rows::<3>(7)]
    }

    fn solve_mpc(&mut self, x_full: &SVector<f32, NX>, horizon: &[Setpoint]) {
        debug_assert!(horizon.len() == TINY_N);
        let grav = self.inner.grav;
        let mut q_prev = UnitQuaternion::identity();
        for k in 0..TINY_N {
            let sp = &horizon[k];
            let q_ref = q_ref_from_setpoint(sp.acceleration, sp.yaw, grav);
            self.tiny.x_ref[k] = stack![sp.position; q_ref.scaled_axis(); sp.velocity];
            if self.reference_feedforward {
                let acc = Vector3::new(sp.acceleration.x, sp.acceleration.y, sp.acceleration.z + grav);
                self.tiny.u_ref[k][0] = self.inner.mass * acc.norm() - self.hover_thrust_n;
                if k > 0 {
                    // Body rate that carries q_ref[k-1] to q_ref[k] in one stride.
                    let w = (q_prev.inverse() * q_ref).scaled_axis() / TINY_DT;
                    self.tiny.u_ref[k - 1].fixed_rows_mut::<3>(1).copy_from(&w);
                }
            }
            q_prev = q_ref;
        }
        let x0 = Self::error_state(x_full);
        let t0 = std::time::Instant::now();
        let res = self.tiny.solve(&x0);
        self.solve_time += t0.elapsed();
        self.solve_count += 1;
        self.last_iters = res.iters;
        self.last_converged = res.converged;
        let mut u0 = self.tiny.u0();
        if !u0.iter().all(|v| v.is_finite()) {
            self.tiny.reset();
            u0 = SVector::zeros();
        }
        u0[0] += self.hover_thrust_n;
        self.inner.last_mpc_u = u0;
    }
}

impl Controller for TinyMpcIndiController {
    fn solve_stats(&self) -> Option<(u64, std::time::Duration)> {
        Some((self.solve_count, self.solve_time))
    }
    fn name(&self) -> &'static str {
        if self.reference_feedforward {
            "tinympc_ff_indi"
        } else {
            "tinympc_indi"
        }
    }

    fn tick_rate_hz(&self) -> f32 {
        self.inner.indi_rate_hz
    }

    fn horizon_samples(&self) -> usize {
        TINY_N
    }

    fn horizon_stride_s(&self) -> f32 {
        TINY_DT
    }

    fn step(
        &mut self,
        x: &SVector<f32, NX>,
        imu: &ImuMeasurement,
        rotor: &RotorTelemetry,
        horizon: &[Setpoint],
    ) -> SVector<f32, NU> {
        if self.inner.tick_counter % self.stride == 0 {
            self.solve_mpc(x, horizon);
        }
        self.inner.tick_counter = self.inner.tick_counter.wrapping_add(1);
        self.inner.step_indi(imu, rotor)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Geometric tracking controller + INDI (RPG position controller port)
// ───────────────────────────────────────────────────────────────────────────

/// Geometric tracking controller solve rate. The reference implementation
/// runs its position controller at the state-estimate rate; 500 Hz keeps
/// it on the same footing as TinyMPC.
const GEO_RATE_HZ: f32 = 500.0;

/// Geometric tracking controller (`cybflight_core::attitude_control::
/// geometric_controller::GeometricTrackingController`, the RPG position
/// controller port) at 500 Hz + the same INDI inner loop as
/// [`MpcIndiController`]. Emits `[m·c, ω_ff + ω_fb]` — the same command
/// vector as the MPC stacks — so the comparison isolates the outer law.
pub struct GeometricIndiController {
    inner: MpcIndiController,
    geo: GeometricTrackingController,
    stride: u32,
    mass: f32,
    /// Command bounds (the SQP model's `u_bounds`): thrust ceiling and rate
    /// limits applied to the output exactly as for the MPC stacks.
    u_bounds: [[f32; 2]; SIMPLE_NU],
}

impl GeometricIndiController {
    pub fn from_params(vp: &FirmwareConfig) -> Self {
        Self::with_params(vp, Self::tuned_params(vp), INDI_LOOP_HZ)
    }

    /// Gains tuned in `sweep_geometric` / `sweep_geometric_fine`
    /// (tests/figure8_tinympc_compare.rs) for the leader vehicle with the
    /// 1 kHz INDI inner loop: geometric RMS over figure8 slow/mid/timeopt
    /// and splits mid/fast, monotone in kp up to 32 (ω_n = 5.7 rad/s,
    /// ζ ≈ 1.06 with kd 12 — 3.5× below the krp = 20 rad/s attitude loop,
    /// where the sweep stops rather than let perfect-state sim reward
    /// unrealistic stiffness). The reference `default.yaml` (10/4/15/6,
    /// krp 12, tight saturations) completes the same missions at 2–3× the
    /// error; the tight error saturations (0.6 m / 1 m/s) cost most on the
    /// fast profiles, so they are opened to 2 m / 4 m/s (1 m / 3 m/s in z).
    pub fn tuned_params(vp: &FirmwareConfig) -> GeometricTrackingParams {
        GeometricTrackingParams {
            kpxy: 32.0,
            kdxy: 12.0,
            kpz: 48.0,
            kdz: 18.0,
            krp: 20.0,
            kyaw: 5.0,
            pxy_error_max: 2.0,
            vxy_error_max: 4.0,
            pz_error_max: 1.0,
            vz_error_max: 3.0,
            min_normalized_thrust: 1.0,
            gravity: vp.site.gravity_m_s2,
        }
    }

    pub fn with_params(vp: &FirmwareConfig, params: GeometricTrackingParams, indi_rate_hz: f32) -> Self {
        let inner = MpcIndiController::from_params_at_indi_rate(vp, indi_rate_hz);
        let u_bounds = inner.problem.model.u_bounds;
        Self {
            mass: inner.mass,
            inner,
            geo: GeometricTrackingController::new(params),
            stride: (indi_rate_hz / GEO_RATE_HZ).round().max(1.0) as u32,
            u_bounds,
        }
    }

    fn solve(&mut self, x_full: &SVector<f32, NX>, horizon: &[Setpoint]) {
        let sp = &horizon[0];
        let state = GeometricTrackingState {
            position: Vector3::new(x_full[0], x_full[1], x_full[2]),
            velocity: Vector3::new(x_full[7], x_full[8], x_full[9]),
            orientation: UnitQuaternion::from_quaternion(Quaternion::new(
                x_full[6], x_full[3], x_full[4], x_full[5],
            )),
            bodyrates: Vector3::new(x_full[10], x_full[11], x_full[12]),
        };
        let reference = GeometricTrackingReference {
            position: sp.position,
            velocity: sp.velocity,
            acceleration: sp.acceleration,
            jerk: sp.jerk,
            snap: sp.snap,
            heading: sp.yaw,
            heading_rate: 0.0,
            heading_acceleration: 0.0,
        };
        let cmd = self.geo.run(&state, &reference);
        let u = vector![
            self.mass * cmd.collective_thrust_per_mass,
            cmd.bodyrates.x,
            cmd.bodyrates.y,
            cmd.bodyrates.z
        ];
        let b = self.u_bounds;
        self.inner.last_mpc_u = SVector::<f32, SIMPLE_NU>::from_fn(|i, _| u[i].clamp(b[i][0], b[i][1]));
    }
}

impl Controller for GeometricIndiController {
    fn name(&self) -> &'static str {
        "geometric_indi"
    }

    fn tick_rate_hz(&self) -> f32 {
        self.inner.indi_rate_hz
    }

    fn horizon_samples(&self) -> usize {
        1
    }

    fn horizon_stride_s(&self) -> f32 {
        1.0 / GEO_RATE_HZ
    }

    fn step(
        &mut self,
        x: &SVector<f32, NX>,
        imu: &ImuMeasurement,
        rotor: &RotorTelemetry,
        horizon: &[Setpoint],
    ) -> SVector<f32, NU> {
        if self.inner.tick_counter % self.stride == 0 {
            self.solve(x, horizon);
        }
        self.inner.tick_counter = self.inner.tick_counter.wrapping_add(1);
        self.inner.step_indi(imu, rotor)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// MPC direct-to-motor (diagnostic baseline)
// ───────────────────────────────────────────────────────────────────────────

pub struct MpcDirectController {
    solver: Box<FullSqpSolver>,
    problem: FullQuadProblem,
    x_refs: [SVector<f32, NX>; FULL_N + 1],
    u_refs: [SVector<f32, NU>; FULL_N],
    u_warm: [SVector<f32, NU>; FULL_N],
    last_u: SVector<f32, NU>,
    max_iters: usize,
    kkt_tol: f32,
    mpc_stride: u32,
    tick_counter: u32,
    grav: f32,
    cmd_map: ThrustCommandMap,
}

impl MpcDirectController {
    pub fn from_params(vp: &FirmwareConfig, sim: &vehicle_yaml::SimYaml) -> Self {
        let mut model = FullQuadModel::from_vehicle_params(vp);
        model.dt = FULL_MPC_DT;
        let grav = model.grav;
        let problem = FullQuadProblem::with_rk4(model, FULL_N);

        let hover_per_motor = vp.airframe.body.mass_kg * grav / NU as f32;
        let u_ref = SVector::<f32, NU>::from_element(hover_per_motor);
        let mut x_ref = SVector::<f32, NX>::zeros();
        x_ref[6] = 1.0;

        let mpc_stride = (MPC_SOLVE_RATE_HZ / MPC_SOLVE_RATE_HZ) as u32; // = 1

        Self {
            solver: Box::new(FullSqpSolver::new()),
            problem,
            x_refs: [x_ref; FULL_N + 1],
            u_refs: [u_ref; FULL_N],
            u_warm: [u_ref; FULL_N],
            last_u: u_ref,
            max_iters: vp.mpc.max_iters as usize,
            kkt_tol: vp.mpc.kkt_tol,
            mpc_stride,
            tick_counter: 0,
            grav,
            cmd_map: ThrustCommandMap::new(PlantParams::from_config(vp, sim)),
        }
    }

    fn fill_reference(&mut self, horizon: &[Setpoint]) {
        for k in 0..=FULL_N {
            let sp = &horizon[k];
            let q_ref = q_ref_from_setpoint(sp.acceleration, sp.yaw, self.grav);
            self.x_refs[k] = stack![sp.position; q_ref.coords; sp.velocity; Vector3::zeros()];
        }
    }
}

impl Controller for MpcDirectController {
    fn name(&self) -> &'static str {
        "mpc_direct"
    }

    fn tick_rate_hz(&self) -> f32 {
        MPC_SOLVE_RATE_HZ
    }

    fn horizon_samples(&self) -> usize {
        FULL_N + 1
    }

    fn horizon_stride_s(&self) -> f32 {
        FULL_MPC_DT
    }

    fn step(
        &mut self,
        x: &SVector<f32, NX>,
        _imu: &ImuMeasurement,
        _rotor: &RotorTelemetry,
        horizon: &[Setpoint],
    ) -> SVector<f32, NU> {
        if self.tick_counter % self.mpc_stride == 0 {
            self.fill_reference(horizon);
            let _ = self.solver.solve(
                &self.problem,
                x,
                &self.x_refs,
                &self.u_refs,
                &self.u_warm,
                self.max_iters,
                self.kkt_tol,
            );
            let u_bar = self.solver.u_bar();
            let u0 = u_bar[0];
            let bounds = self.problem.model.u_bounds;
            self.last_u =
                SVector::<f32, NU>::from_fn(|i, _| u0[i].clamp(bounds[i][0], bounds[i][1]));
            self.u_warm = *u_bar;
        }
        self.tick_counter = self.tick_counter.wrapping_add(1);
        self.cmd_map.command(&self.last_u)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Full-model MPC + INDI inner loop (`outer_loop: mpc_full` prototype)
// ───────────────────────────────────────────────────────────────────────────
//
// The Sun et al. (T-RO 2022, arXiv:2109.01365v6 Fig. 3) hybrid: the
// 13-state `FullQuadModel` NMPC solves for per-motor thrusts at
// `mpc_rate_hz`; its first control is reduced to `(T_d, α_d)` via
// `FullQuadModel::inner_setpoint` (eq. 32) and *held* between solves,
// while `IndiController::step_alpha` runs the torque loop + disturbance
// rejection at the full INDI rate. No rate gains anywhere in this path.
//
// `indi_active = false` selects the paper's "NMPC w/o INDI" ablation:
// the same `(T_d, α_d)` command chain, but the inner loop degrades to
// model-based static inversion through G1 (no incremental correction) —
// the firmware `build: indi: no` analogue.

pub struct MpcFullIndiController {
    solver: Box<FullSqpSolver>,
    problem: FullQuadProblem,
    x_refs: [SVector<f32, NX>; FULL_N + 1],
    u_refs: [SVector<f32, NU>; FULL_N],
    u_warm: [SVector<f32, NU>; FULL_N],
    /// Held inner-loop setpoint: desired collective thrust [N] and RAW
    /// model body torque τ(u0) [N·m], refreshed each MPC solve. The α
    /// pseudo-control is re-derived from τ_d every INDI tick with the
    /// fresh gyro (α = I⁻¹·(τ_d − ω×Iω)) — mirrors the firmware
    /// `indi_task` decode and the paper's inner-loop placement of
    /// eq. (32).
    thrust_d_n: f32,
    torque_d_n_m: Vector3<f32>,
    max_iters: usize,
    kkt_tol: f32,
    indi: IndiController,
    mpc_stride: u32,
    tick_counter: u32,
    mass: f32,
    grav: f32,
    // Rotor-telemetry plumbing — mirrors `MpcIndiController::update_rotor`.
    omega_filter: [Biquad; INDI_NU],
    omega_fs: SVector<f32, INDI_NU>,
    omega_dot_fs: SVector<f32, INDI_NU>,
    omega_hold: SVector<f32, INDI_NU>,
    omega_fs_has_prev: bool,
    erpm_to_rads: f32,
    nominal_voltage_v: f32,
    name: &'static str,
}

impl MpcFullIndiController {
    /// Flight-representative configuration: INDI active, MPC at
    /// `MPC_SOLVE_RATE_HZ`.
    pub fn from_params(vp: &FirmwareConfig) -> Self {
        Self::with_options(vp, MPC_SOLVE_RATE_HZ, true)
    }

    /// `mpc_rate_hz` sets the outer solve rate (stride off the INDI
    /// loop); `indi_active = false` degrades the inner loop to static
    /// inversion (the paper's no-INDI ablation).
    pub fn with_options(vp: &FirmwareConfig, mpc_rate_hz: f32, indi_active: bool) -> Self {
        let mut model = FullQuadModel::from_vehicle_params(vp);
        model.dt = FULL_MPC_DT;
        let grav = model.grav;
        let mass = vp.airframe.body.mass_kg;
        let problem = FullQuadProblem::with_rk4(model, FULL_N);

        let hover_per_motor = mass * grav / NU as f32;
        let u_ref = SVector::<f32, NU>::from_element(hover_per_motor);
        let mut x_ref = SVector::<f32, NX>::zeros();
        x_ref[6] = 1.0;

        let indi_cfg = build_indi_config(vp, indi_active);
        let indi = IndiController::new(&indi_cfg, INDI_LOOP_HZ);
        let indi_sync_hz = indi.effective_sync_filter_hz();

        let mpc_stride = (INDI_LOOP_HZ / mpc_rate_hz).round().max(1.0) as u32;
        let pole_pairs = (vp.airframe.motor_pole_count.max(2) as f32) / 2.0;
        let erpm_to_rads = core::f32::consts::TAU * 100.0 / (pole_pairs * 60.0);

        Self {
            solver: Box::new(FullSqpSolver::new()),
            problem,
            x_refs: [x_ref; FULL_N + 1],
            u_refs: [u_ref; FULL_N],
            u_warm: [u_ref; FULL_N],
            thrust_d_n: mass * grav,
            torque_d_n_m: Vector3::zeros(),
            max_iters: vp.mpc.max_iters as usize,
            kkt_tol: vp.mpc.kkt_tol,
            indi,
            mpc_stride,
            tick_counter: 0,
            mass,
            grav,
            omega_filter: core::array::from_fn(|_| {
                make_omega_biquad(INDI_LOOP_HZ, indi_sync_hz)
            }),
            omega_fs: SVector::zeros(),
            omega_dot_fs: SVector::zeros(),
            omega_hold: SVector::zeros(),
            omega_fs_has_prev: false,
            erpm_to_rads,
            nominal_voltage_v: 23.0,
            name: if indi_active {
                "mpc_full_indi"
            } else {
                "mpc_full_noindi"
            },
        }
    }

    /// Same telemetry path as `MpcIndiController::update_rotor` (duplicated
    /// rather than shared: that controller's fields are frozen behavior
    /// under the regression snapshot, and a joint refactor buys little for
    /// ~25 lines).
    fn update_rotor(&mut self, rotor: &RotorTelemetry) -> [bool; INDI_NU] {
        let inputs: [RpmInput; INDI_NU] = core::array::from_fn(|i| {
            if !rotor.valid[i] || !rotor.omega_rad_s[i].is_finite() {
                RpmInput::Invalid
            } else if rotor.omega_rad_s[i] <= 0.0 {
                RpmInput::Stopped
            } else {
                RpmInput::Erpm((rotor.omega_rad_s[i] / self.erpm_to_rads).round() as u32)
            }
        });
        let (g2_valid, _rpm_failsafe) = self.indi.update_rpm(&inputs);

        for i in 0..INDI_NU {
            let x = if rotor.valid[i] && rotor.omega_rad_s[i].is_finite() {
                self.omega_hold[i] = rotor.omega_rad_s[i];
                rotor.omega_rad_s[i]
            } else {
                self.omega_hold[i]
            };
            let new_fs = self.omega_filter[i].apply(x);
            self.omega_dot_fs[i] = if self.omega_fs_has_prev {
                (new_fs - self.omega_fs[i]) * INDI_LOOP_HZ
            } else {
                0.0
            };
            self.omega_fs[i] = new_fs;
        }
        self.omega_fs_has_prev = true;
        g2_valid
    }

    fn fill_reference(&mut self, horizon: &[Setpoint]) {
        debug_assert!(horizon.len() == FULL_N + 1);
        for k in 0..=FULL_N {
            let sp = &horizon[k];
            let q_ref = q_ref_from_setpoint(sp.acceleration, sp.yaw, self.grav);
            self.x_refs[k] = stack![sp.position; q_ref.coords; sp.velocity; Vector3::zeros()];
        }
    }

    fn solve_mpc(&mut self, x: &SVector<f32, NX>, horizon: &[Setpoint]) {
        self.fill_reference(horizon);
        let _ = self.solver.solve(
            &self.problem,
            x,
            &self.x_refs,
            &self.u_refs,
            &self.u_warm,
            self.max_iters,
            self.kkt_tol,
        );
        let u_bar = self.solver.u_bar();
        let bounds = self.problem.model.u_bounds;
        let u0 =
            SVector::<f32, NU>::from_fn(|i, _| u_bar[0][i].clamp(bounds[i][0], bounds[i][1]));

        // Eq. (32): reduce the first control to (T_d, τ_d) — the RAW
        // allocation torque, held between solves. Guard: a non-finite
        // solution keeps the previous held setpoint (solver-health
        // fallback — the firmware variant escalates to failsafe on
        // repeats).
        let (t_d, tau_x, tau_y, tau_z) = self.problem.model.alloc(&u0);
        let tau = Vector3::new(tau_x, tau_y, tau_z);
        if t_d.is_finite() && tau.iter().all(|v| v.is_finite()) {
            self.thrust_d_n = t_d;
            self.torque_d_n_m = tau;
            self.u_warm = *u_bar;
        } else {
            // Reset the warm start too — a NaN trajectory would poison
            // every subsequent solve.
            self.u_warm = self.u_refs;
        }
    }
}

impl Controller for MpcFullIndiController {
    fn name(&self) -> &'static str {
        self.name
    }

    fn tick_rate_hz(&self) -> f32 {
        INDI_LOOP_HZ
    }

    fn horizon_samples(&self) -> usize {
        FULL_N + 1
    }

    fn horizon_stride_s(&self) -> f32 {
        FULL_MPC_DT
    }

    fn step(
        &mut self,
        x: &SVector<f32, NX>,
        imu: &ImuMeasurement,
        rotor: &RotorTelemetry,
        horizon: &[Setpoint],
    ) -> SVector<f32, NU> {
        if self.tick_counter % self.mpc_stride == 0 {
            self.solve_mpc(x, horizon);
        }
        self.tick_counter = self.tick_counter.wrapping_add(1);

        let spf_sp_z = self.thrust_d_n / self.mass;

        // α pseudo-control from the held torque with the FRESH gyro:
        // α = I⁻¹·(τ_d − ω×Iω), full-tensor Euler form (mirrors the
        // firmware indi_task decode).
        let alpha_sp = {
            let g = &imu.gyro;
            let model = &self.problem.model;
            model.inertia_inv * (self.torque_d_n_m - g.cross(&(model.inertia * g)))
        };

        let g2_valid = self.update_rotor(rotor);
        let any_valid = g2_valid.iter().any(|v| *v);
        let motor_state = if any_valid && self.omega_fs_has_prev {
            MotorState::External {
                omega_fs: &self.omega_fs,
                omega_dot_fs: &self.omega_dot_fs,
            }
        } else {
            MotorState::Internal
        };

        let (out, _step_state) = self.indi.step_alpha(
            &imu.gyro,
            &imu.accel,
            &alpha_sp,
            spf_sp_z,
            true,
            &g2_valid,
            motor_state,
            self.nominal_voltage_v,
        );

        out.motor_commands
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Cascade controller (legacy PD+FF baseline)
// ───────────────────────────────────────────────────────────────────────────

/// Rate-loop P-gains for the legacy cascade baseline. Sim-only — the firmware's
/// active inner loop is INDI and does not carry rate PID gains in `ControlGains`.
const CASCADE_RATE_KP: Vector3<f32> = Vector3::new(0.1, 0.08, 0.05);

pub struct CascadeController {
    pos_ctrl: PositionController<f32>,
    att_ctrl: GeometricAttitudeController<f32>,
    allocator: LinearAllocator<4>,
    rate_kp: Vector3<f32>,
    rate_clamp: Vector3<f32>,
    per_motor_max_n: f32,
    max_collective_n: f32,
    idle_n: f32,
    cmd_map: ThrustCommandMap,
}

impl CascadeController {
    pub fn from_params(vp: &FirmwareConfig, sim: &vehicle_yaml::SimYaml) -> Self {
        let g = &vp.cascade;
        let per_motor_max = vp
            .airframe
            .motors
            .iter()
            .map(|m| m.max_thrust_n)
            .fold(0.0f32, f32::max);
        let max_collective = vp.airframe.motors.iter().map(|m| m.max_thrust_n).sum::<f32>();

        let pos_ctrl = PositionController::new(
            Vector3::new(g.pos_kp[0], g.pos_kp[1], g.pos_kp[2]),
            Vector3::new(g.pos_kd[0], g.pos_kd[1], g.pos_kd[2]),
            position_control::VehicleParams {
                mass: vp.airframe.body.mass_kg,
                gravity: 9.81,
            },
        );
        let att_ctrl =
            GeometricAttitudeController::new(g.att_k_rate.into(), Vector3::new(1.0, 1.0, 0.2))
                .with_inertia(vp.airframe.body.inertia_matrix());

        let effectiveness = MotorEffectiveness::from_motors(&vp.airframe.motors);
        let allocator = LinearAllocator::new(effectiveness);

        Self {
            pos_ctrl,
            att_ctrl,
            allocator,
            rate_kp: CASCADE_RATE_KP,
            rate_clamp: Vector3::new(0.8, 0.6, 0.15),
            per_motor_max_n: per_motor_max,
            max_collective_n: max_collective,
            idle_n: 0.005 * max_collective,
            cmd_map: ThrustCommandMap::new(PlantParams::from_config(vp, sim)),
        }
    }
}

impl Controller for CascadeController {
    fn name(&self) -> &'static str {
        "cascade"
    }

    fn tick_rate_hz(&self) -> f32 {
        MPC_SOLVE_RATE_HZ
    }

    fn step(
        &mut self,
        x: &SVector<f32, NX>,
        _imu: &ImuMeasurement,
        _rotor: &RotorTelemetry,
        horizon: &[Setpoint],
    ) -> SVector<f32, NU> {
        let sp = &horizon[0];
        let pos = Vector3::new(x[0], x[1], x[2]);
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(x[6], x[3], x[4], x[5]));
        let vel = Vector3::new(x[7], x[8], x[9]);
        let omega = Vector3::new(x[10], x[11], x[12]);

        let pos_out = self.pos_ctrl.compute(
            &PositionControlState {
                position: pos,
                velocity: vel,
                attitude: quat,
            },
            &PositionControlSetpoint {
                position: sp.position,
                velocity: sp.velocity,
                acceleration_ff: sp.acceleration,
                yaw: sp.yaw,
            },
        );

        let att_out = self.att_ctrl.compute(
            &AttitudeControlState {
                attitude_quaternion: quat,
                body_rate_rad_s: omega,
            },
            &AttitudeControlSetpoint {
                attitude_quaternion: Some(pos_out.desired_attitude_quaternion),
                body_rate_rad_s: Vector3::zeros(),
                angular_accel_rad_s2: Vector3::zeros(),
            },
        );

        let rate_err = att_out.body_rate_rad_s - omega;
        let torque = Vector3::new(
            (self.rate_kp.x * rate_err.x).clamp(-self.rate_clamp.x, self.rate_clamp.x),
            (self.rate_kp.y * rate_err.y).clamp(-self.rate_clamp.y, self.rate_clamp.y),
            (self.rate_kp.z * rate_err.z).clamp(-self.rate_clamp.z, self.rate_clamp.z),
        );

        let thrust = pos_out
            .collective_thrust_n
            .max(self.idle_n)
            .min(self.max_collective_n);
        let throttles = self
            .allocator
            .allocate(Vector4::new(thrust, torque.x, torque.y, torque.z));

        // `allocate` returns per-motor force fractions; scale to newtons,
        // then invert the plant's actuator map to reach an ESC command.
        self.cmd_map.command(&(throttles * self.per_motor_max_n))
    }
}
