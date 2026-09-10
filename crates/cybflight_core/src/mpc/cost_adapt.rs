//! Situation-conditioned cost adaptation for the tracking NMPC
//! (docs/learned_mpc_cost.md, PLAN B).
//!
//! A small policy network maps the vehicle's current *situation* —
//! attitude, velocity, tracking error and the maneuver the reference is
//! about to demand — to a **residual** on the [`QuadModel`] state cost
//! weights relative to the vehicle's nominal tune: one constant stage set
//! and one terminal set, `w = w_nom·UP^z` / `w_nom·DOWN^(−z)` per channel
//! (velocity and roll/pitch attitude symmetric at ×4; contour, lag and
//! yaw carry asymmetric fences — see the map constants), plus the three body-rate input weights on the
//! same kind of scale, `w_rate = w_rate,nom·RATE_LOG_BASE^z`. The thrust input weight stays at
//! its nominal value and is not learned. The solver, horizon and references are untouched;
//! the network only decides how the SQP trades its state objectives
//! against each other over the next horizon.
//!
//! Everything here is `no_std` and shared by the sim (training,
//! evaluation) and the firmware outer loop, so the observation the policy
//! was trained on is the observation it sees in flight.
//!
//! # Observation
//!
//! All vectors are expressed in the reference **path frame**
//! `{t̂ = v_ref/‖v_ref‖, n̂, b̂}` built from node 0 of the horizon
//! (world frame when the reference is at rest), which makes the
//! observation yaw-invariant. Nothing in it identifies the trajectory:
//! no absolute position, no yaw, no mission time, no solver history.
//!
//! | block | entries | index |
//! |---|---|---|
//! | position error (path frame, /1 m; ±3 m = the training crash bound) | 3 | 0..3 |
//! | velocity error (path frame, /`V_ERR_SCALE` = 15 m/s) | 3 | 3..6 |
//! | vehicle body z in path frame | 3 | 6..9 |
//! | reference body z (node 0) in path frame | 3 | 9..12 |
//! | attitude error rotation vector (/1 rad) | 3 | 12..15 |
//! | body rates (/10 rad/s, the rate limit) | 3 | 15..18 |
//! | velocity (path frame, /`V_SCALE` = 60 m/s) | 3 | 18..21 |
//! | `‖v_ref‖` (/`V_SCALE`) | 1 | 21 |
//!
//! The units are one fixed **flight envelope** shared by every vehicle
//! and mission (indoor and outdoor): `V_SCALE` = 60 m/s, velocity error
//! scale `V_SCALE/4`, position error in metres with the 3 m clamp equal
//! to the training crash bound. They are part of the checkpoint: a policy
//! trained under one set of constants is only valid with the same set.
//! | preview at nodes {0,5,10,15,20}: `cos θ_ref`, ref body z (3), `‖ω_ref‖/ω_max`, thrust margin, rate margin | 7×5 | 22..57 |
//! | commanded thrust fraction, max commanded rate fraction | 2 | 57..59 |

use nalgebra::{Quaternion, SVector, UnitQuaternion, Vector3};
#[allow(unused_imports)]
use num_traits::Float;

use super::quad_model::{QuadModel, NU, NX};
use crate::nn::mlp::{Activation, LayerShape, Mlp, MlpError};
use crate::trajectory_planning::flatness::flatness_to_thrust_omega;
use crate::trajectory_planning::sampler::SamplerNode;

/// Revision of the weight map and observation encoding a checkpoint is
/// valid against — every constant in this module that a trained policy
/// assumed, taken together: the observation layout and its scales, the
/// per-channel fences, and the rate map.
///
/// The structural checks the bake can make (input width, output width,
/// layer chain) do not see any of that. v18 → v20 moved `CONTOUR_UP`
/// from 3 to 4 and changed nothing else, so a v18 checkpoint still
/// matches every shape assertion while meaning something different by
/// the same `z` — it would fly weight modulation with the wrong
/// semantics and no build or runtime check would notice. Exporting the
/// number into the checkpoint and asserting it at bake time is what
/// makes "each map revision invalidates all earlier checkpoints"
/// (docs/learned_mpc_cost_deploy.md) enforceable rather than a note.
///
/// **Bump this whenever any constant below changes**, and re-export
/// every checkpoint that is still in service. It is the map's revision,
/// not a checkpoint name: several policies (v20, v20b) share map 20.
pub const COST_MAP_VERSION: u32 = 20;

/// Policy output width: 8 stage + 8 terminal residuals
/// (`[contour, lag, vel×3, att×3]` each) + 3 body-rate input weights.
pub const NZ: usize = 19;
/// Observation width.
pub const OBS_DIM: usize = 59;
/// Horizon nodes sampled for the maneuver preview.
pub const PREVIEW_NODES: [usize; 5] = [0, 5, 10, 15, 20];
/// Default state-weight range: `w = w_nom · STATE_LOG_BASE^z`,
/// `z ∈ [−1, 1]`, i.e. `[w_nom/4, 4·w_nom]` — log-symmetric so "×2" and
/// "÷2" are the same distance from nominal; fenced far below the ×10
/// that broke v2b (docs/learned_mpc_cost.md). Applies to the velocity
/// and roll/pitch attitude weights; contour, lag and yaw carry their
/// own asymmetric fences below (v18 map, 2026-09-08).
pub const STATE_LOG_BASE: f32 = 4.0;
/// Per-channel asymmetric fences, piecewise log around the nominal:
/// `z ≥ 0` → `w_nom·UP^z`, `z < 0` → `w_nom·DOWN^(−z)`. The absolute
/// admissible ranges at the flight nominal (contour 500, lag 200,
/// att yaw 200) are contour [100, 2000] (`CONTOUR_UP` raised 3 → 4 for
/// v20 — v18 saturated the ×3 ceiling on 19 % of fast-mission solves),
/// lag [50, 500] and yaw [50, 200] — the yaw weight can only be
/// *softened* (UP = 1). These constants are part of the checkpoint: a
/// policy trained under one map is only valid with the same map.
pub const CONTOUR_UP: f32 = 4.0;
pub const CONTOUR_DOWN: f32 = 0.2;
pub const LAG_UP: f32 = 2.5;
pub const LAG_DOWN: f32 = 0.25;
pub const ATT_YAW_UP: f32 = 1.0;
pub const ATT_YAW_DOWN: f32 = 0.25;
/// Rate-weight map, piecewise log around the nominal: `z ≥ 0` →
/// `w_nom · RATE_LOG_BASE^z` (up to `4·w_nom`), `z < 0` →
/// `w_nom · (RATE_FLOOR_FRAC)^(−z)` (down to `RATE_FLOOR_FRAC·w_nom`).
/// `RATE_FLOOR_FRAC = 1.0` is the v10 map (rate weights can only go *up*
/// from the nominal 10 — the motor-lag sweep in docs/learned_mpc_cost.md
/// put the constant-tune cliff above 5 on the fastest missions). v9 was
/// trained with 0.8; the constant is part of the checkpoint.
pub const RATE_LOG_BASE: f32 = 4.0;
pub const RATE_FLOOR_FRAC: f32 = 1.0;
/// Widest hidden layer the policy may use.
pub const POLICY_MAX_WIDTH: usize = 128;
/// Flight-envelope speed the velocity entries are normalized by [m/s].
pub const V_SCALE: f32 = 60.0;
/// Velocity-error normalization [m/s] (`V_SCALE / 4`).
pub const V_ERR_SCALE: f32 = 15.0;
/// Body-rate normalization [rad/s] (the rate limit).
pub const RATE_SCALE: f32 = 10.0;
/// Clamp on the error / rate / velocity entries after scaling; for the
/// position error this is the 3 m training crash bound in metres.
pub const OBS_CLAMP: f32 = 3.0;

/// Reference speed below which the path frame falls back to the world
/// frame (matches `PosCostMode::VEL_EPS`).
const PATH_FRAME_VEL_EPS: f32 = 0.1;

/// The hand-tuned weights the policy modulates. Captured from the model
/// once at construction so repeated modulation never compounds.
#[derive(Clone, Copy, Debug)]
pub struct CostNominal {
    pub w_pos: [f32; 3],
    pub w_vel: [f32; 3],
    pub w_att: [f32; 3],
    pub w_pos_n: [f32; 3],
    pub w_vel_n: [f32; 3],
    pub w_att_n: [f32; 3],
    /// Nominal body-rate input weights `w_input[1..4]`.
    pub w_rate: [f32; 3],
}

impl CostNominal {
    pub fn from_model(m: &QuadModel) -> Self {
        Self {
            w_pos: m.w_pos,
            w_vel: m.w_vel,
            w_att: m.w_att,
            w_pos_n: m.w_pos_n,
            w_vel_n: m.w_vel_n,
            w_att_n: m.w_att_n,
            w_rate: [m.w_input[1], m.w_input[2], m.w_input[3]],
        }
    }

    /// Write the per-channel piecewise-log map (state weights) and
    /// `w_rate,nom·RATE_LOG_BASE^z` (rate input weights) into the model.
    /// `z` is clamped to `[−1, 1]` per entry and a non-finite entry counts
    /// as `0` (nominal). The thrust input weight is left untouched.
    ///
    /// Layout of `z`: stage `[contour, lag, vel×3, att×3]`, terminal
    /// `[contour, lag, vel×3, att×3]`, then `[rate_x, rate_y, rate_z]`. "Contour" scales `w_pos[0..2]` and
    /// "lag" scales `w_pos[2]`, which in `Contouring` mode are exactly the
    /// contour / lag weights and in `Quadratic` mode the horizontal /
    /// vertical position weights.
    pub fn apply(&self, z: &[f32; NZ], m: &mut QuadModel) {
        // Piecewise log per channel: `z ≥ 0` → `up^z`, `z < 0` →
        // `down^(−z)` (identity at 0 either way; symmetric channels pass
        // `down = 1/up`).
        let s = |i: usize, up: f32, down: f32| {
            let v = z[i];
            let v = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
            if v >= 0.0 { up.powf(v) } else { down.powf(-v) }
        };
        let g = |i: usize| s(i, STATE_LOG_BASE, 1.0 / STATE_LOG_BASE);
        let (c, l) = (s(0, CONTOUR_UP, CONTOUR_DOWN), s(1, LAG_UP, LAG_DOWN));
        m.w_pos = [self.w_pos[0] * c, self.w_pos[1] * c, self.w_pos[2] * l];
        for i in 0..3 {
            m.w_vel[i] = self.w_vel[i] * g(2 + i);
        }
        m.w_att = [
            self.w_att[0] * g(5),
            self.w_att[1] * g(6),
            self.w_att[2] * s(7, ATT_YAW_UP, ATT_YAW_DOWN),
        ];
        let (cn, ln) = (s(8, CONTOUR_UP, CONTOUR_DOWN), s(9, LAG_UP, LAG_DOWN));
        m.w_pos_n = [self.w_pos_n[0] * cn, self.w_pos_n[1] * cn, self.w_pos_n[2] * ln];
        for i in 0..3 {
            m.w_vel_n[i] = self.w_vel_n[i] * g(10 + i);
        }
        m.w_att_n = [
            self.w_att_n[0] * g(13),
            self.w_att_n[1] * g(14),
            self.w_att_n[2] * s(15, ATT_YAW_UP, ATT_YAW_DOWN),
        ];
        for i in 0..3 {
            let v = z[16 + i];
            let v = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
            m.w_input[1 + i] = if v >= 0.0 {
                self.w_rate[i] * RATE_LOG_BASE.powf(v)
            } else {
                self.w_rate[i] * RATE_FLOOR_FRAC.powf(-v)
            };
        }
    }
}

/// Everything the observation needs, borrowed from the outer loop right
/// after the horizon references are assembled and before the solve.
pub struct SituationInputs<'a> {
    /// Current 10-state `[p, q_xyzw, v]` handed to the solver.
    pub x0: &'a SVector<f32, NX>,
    /// Measured body rates [rad/s].
    pub body_rate: Vector3<f32>,
    /// Horizon references `x_refs[0..=N]` (`[p_ref, q_ref_xyzw, v_ref]`).
    pub x_refs: &'a [SVector<f32, NX>],
    /// Sampler nodes `0..=N` (acceleration and jerk feed the flatness
    /// demand preview).
    pub nodes: &'a [SamplerNode],
    /// Last commanded control `[thrust N, ω]`.
    pub last_u: &'a SVector<f32, NU>,
    /// Control box `[thrust, ωx, ωy, ωz]` as `[lo, hi]`.
    pub u_bounds: [[f32; 2]; NU],
    pub mass: f32,
    pub grav: f32,
}

#[inline]
fn quat_xyzw(x: &SVector<f32, NX>) -> UnitQuaternion<f32> {
    UnitQuaternion::from_quaternion(Quaternion::new(x[6], x[3], x[4], x[5]))
}

#[inline]
fn body_z(q: &UnitQuaternion<f32>) -> Vector3<f32> {
    q * Vector3::z()
}

/// Orthonormal path frame rows `(t̂, n̂, b̂)` from the reference velocity.
fn path_frame(v_ref: Vector3<f32>) -> [Vector3<f32>; 3] {
    let t = if v_ref.norm() > PATH_FRAME_VEL_EPS {
        v_ref / v_ref.norm()
    } else {
        Vector3::x()
    };
    let up = Vector3::z();
    let mut n = up.cross(&t);
    if n.norm() < 1e-3 {
        // Vertical tangent: any horizontal normal will do; pick one that
        // varies continuously with the tangent's tiny horizontal part.
        n = t.cross(&Vector3::x());
        if n.norm() < 1e-3 {
            n = Vector3::y();
        }
    }
    let n = n / n.norm();
    let b = t.cross(&n);
    [t, n, b]
}

#[inline]
fn in_frame(f: &[Vector3<f32>; 3], v: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(f[0].dot(&v), f[1].dot(&v), f[2].dot(&v))
}

/// Assemble the situation observation. `x_refs` and `nodes` must cover
/// every index in [`PREVIEW_NODES`] (i.e. `N ≥ 20`); shorter horizons
/// reuse their last node.
pub fn situation_obs(inp: &SituationInputs<'_>, out: &mut [f32; OBS_DIM]) {
    let x0 = inp.x0;
    let r0 = &inp.x_refs[0];
    let p = Vector3::new(x0[0], x0[1], x0[2]);
    let v = Vector3::new(x0[7], x0[8], x0[9]);
    let p_ref = Vector3::new(r0[0], r0[1], r0[2]);
    let v_ref = Vector3::new(r0[7], r0[8], r0[9]);
    let f = path_frame(v_ref);

    let q = quat_xyzw(x0);
    let q_ref0 = quat_xyzw(r0);
    let e_att = (q_ref0.inverse() * q).scaled_axis();

    let e_p = in_frame(&f, p - p_ref);
    let e_v = in_frame(&f, v - v_ref);
    let zb = in_frame(&f, body_z(&q));
    let zb_ref = in_frame(&f, body_z(&q_ref0));
    let v_pf = in_frame(&f, v);

    // Error / rate / velocity blocks are clamped to the envelope the policy
    // was trained on (≈ ±3 after scaling) so an out-of-distribution input
    // — a localization jump, a tumble — saturates gracefully instead of
    // driving the hidden layers with values the network never saw.
    let cl = |v: f32| v.clamp(-OBS_CLAMP, OBS_CLAMP);
    let mut o = [0.0f32; OBS_DIM];
    for i in 0..3 {
        o[i] = cl(e_p[i]);
        o[3 + i] = cl(e_v[i] / V_ERR_SCALE);
        o[6 + i] = zb[i];
        o[9 + i] = zb_ref[i];
        o[12 + i] = cl(e_att[i]);
        o[15 + i] = cl(inp.body_rate[i] / RATE_SCALE);
        o[18 + i] = cl(v_pf[i] / V_SCALE);
    }
    o[21] = cl(v_ref.norm() / V_SCALE);

    let t_max = inp.u_bounds[0][1].max(1e-3);
    let w_max = [
        inp.u_bounds[1][1].abs().max(1e-3),
        inp.u_bounds[2][1].abs().max(1e-3),
        inp.u_bounds[3][1].abs().max(1e-3),
    ];
    let last = inp.x_refs.len().min(inp.nodes.len()).saturating_sub(1);
    for (j, &k) in PREVIEW_NODES.iter().enumerate() {
        let k = k.min(last);
        let xr = &inp.x_refs[k];
        let n = &inp.nodes[k];
        let qr = quat_xyzw(xr);
        let zr_world = body_z(&qr);
        let zr = in_frame(&f, zr_world);
        let (thrust_n, w_ref) = if n.past_end {
            (inp.mass * inp.grav, Vector3::zeros())
        } else {
            match flatness_to_thrust_omega(n.acc, n.jerk, 0.0, 0.0, inp.grav) {
                Ok((alpha, _, omega)) => (inp.mass * alpha, omega),
                Err(_) => {
                    let alpha = Vector3::new(n.acc.x, n.acc.y, n.acc.z + inp.grav).norm();
                    (inp.mass * alpha, Vector3::zeros())
                }
            }
        };
        let rate_frac = (0..3).fold(0.0f32, |m, i| m.max(w_ref[i].abs() / w_max[i]));
        let base = 22 + 7 * j;
        o[base] = zr_world.z; // cos of reference tilt
        o[base + 1] = zr.x;
        o[base + 2] = zr.y;
        o[base + 3] = zr.z;
        o[base + 4] = rate_frac.min(3.0);
        o[base + 5] = (1.0 - thrust_n / t_max).clamp(-2.0, 1.0);
        o[base + 6] = (1.0 - rate_frac).clamp(-2.0, 1.0);
    }
    o[57] = (inp.last_u[0] / t_max).clamp(0.0, 1.5);
    o[58] = (0..3)
        .fold(0.0f32, |m, i| m.max(inp.last_u[1 + i].abs() / w_max[i]))
        .min(1.5);
    for v in o.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    *out = o;
}

/// A trained cost policy: `obs → z`. The network is an SB3 `MlpPolicy`
/// actor (tanh hidden layers, linear head); the deterministic action is
/// the head output clamped to the `[−1, 1]` action box, exactly what
/// `predict(deterministic=True)` returns after SB3's clipping.
pub struct CostPolicy<'a> {
    mlp: Mlp<'a, POLICY_MAX_WIDTH>,
}

impl<'a> CostPolicy<'a> {
    pub fn new(weights: &'a [f32], shapes: &'a [LayerShape]) -> Result<Self, MlpError> {
        let mlp = Mlp::new(weights, shapes, Activation::Tanh)?;
        if mlp.input_dim() != OBS_DIM || mlp.output_dim() != NZ {
            return Err(MlpError::BadIo);
        }
        Ok(Self { mlp })
    }

    /// Returns `false` (and leaves `z` at nominal zeros) if the network
    /// produced a non-finite output.
    pub fn act(&self, obs: &[f32; OBS_DIM], z: &mut [f32; NZ]) -> bool {
        let mut out = [0.0f32; NZ];
        if self.mlp.forward(obs, &mut out).is_err() || out.iter().any(|v| !v.is_finite()) {
            *z = [0.0; NZ];
            return false;
        }
        for i in 0..NZ {
            z[i] = out[i].clamp(-1.0, 1.0);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> QuadModel {
        QuadModel {
            mass: 0.6,
            grav: 9.81,
            dt: 0.05,
            u_bounds: [[0.0, 30.0], [-10.0, 10.0], [-10.0, 10.0], [-6.0, 6.0]],
            mass_inv: 1.0 / 0.6,
            w_pos: [500.0, 500.0, 200.0],
            w_vel: [10.0, 10.0, 10.0],
            w_att: [50.0, 50.0, 200.0],
            w_pos_n: [500.0, 500.0, 200.0],
            w_vel_n: [10.0, 10.0, 10.0],
            w_att_n: [50.0, 50.0, 200.0],
            w_input: SVector::<f32, NU>::from_element(1.0),
            rho: 1e4,
            pos_cost_mode: super::super::quad_model::PosCostMode::Contouring,
            tilt_cos_max: -1.0,
            tilt_barrier_tau: 0.0,
            tilt_barrier_delta: 0.05,
            drag_coeff: [0.0; 3],
            thrust_coeff: 0.0,
            body_drag_coeff: [0.0; 3],
        }
    }

    /// `z = 0` must leave every weight untouched; `z = ±1` must land on
    /// each channel's own fence (`UP·w_nom` / `DOWN·w_nom`: contour
    /// [0.2, 3], lag [0.25, 2.5], yaw [0.25, 1], the rest ×4 symmetric;
    /// rates `4·w_nom` / `RATE_FLOOR_FRAC·w_nom`), and the thrust weight
    /// must never move.
    #[test]
    fn zero_is_identity_and_log_range_holds() {
        let mut m = model();
        let nom = CostNominal::from_model(&m);
        let w_input = m.w_input;
        nom.apply(&[0.0; NZ], &mut m);
        assert_eq!(m.w_pos, nom.w_pos);
        assert_eq!(m.w_att_n, nom.w_att_n);
        let mut z = [0.0; NZ];
        z[0] = 1.0;
        z[1] = 1.0;
        z[7] = 1.0;
        z[9] = -1.0;
        z[15] = -1.0;
        z[5] = f32::NAN;
        z[16] = 1.0;
        z[18] = -1.0;
        nom.apply(&z, &mut m);
        assert!((m.w_pos[0] / nom.w_pos[0] - CONTOUR_UP).abs() < 1e-4);
        assert!((m.w_pos[2] / nom.w_pos[2] - LAG_UP).abs() < 1e-4);
        // Yaw can only soften: z = +1 is still the nominal.
        assert!((m.w_att[2] / nom.w_att[2] - ATT_YAW_UP).abs() < 1e-5);
        assert!((m.w_att_n[2] / nom.w_att_n[2] - ATT_YAW_DOWN).abs() < 1e-5);
        assert!((m.w_pos_n[2] / nom.w_pos_n[2] - LAG_DOWN).abs() < 1e-5);
        assert_eq!(m.w_att[0], nom.w_att[0]);
        assert_eq!(m.w_input[0], w_input[0]);
        assert!((m.w_input[1] / w_input[1] - 4.0).abs() < 1e-4);
        assert_eq!(m.w_input[2], w_input[2]);
        assert!((m.w_input[3] / w_input[3] - RATE_FLOOR_FRAC).abs() < 1e-5);
    }

    /// On the reference with the reference attitude, every error entry is
    /// zero and the preview describes level flight at hover thrust.
    #[test]
    fn on_reference_observation_is_error_free() {
        let n = 21;
        let mut x_refs = [SVector::<f32, NX>::zeros(); 21];
        let mut nodes = [SamplerNode::default(); 21];
        for k in 0..n {
            x_refs[k][6] = 1.0;
            x_refs[k][7] = 3.0;
            nodes[k].vel = Vector3::new(3.0, 0.0, 0.0);
        }
        let x0 = x_refs[0];
        let last_u = SVector::<f32, NU>::new(0.6 * 9.81, 0.0, 0.0, 0.0);
        let inp = SituationInputs {
            x0: &x0,
            body_rate: Vector3::zeros(),
            x_refs: &x_refs,
            nodes: &nodes,
            last_u: &last_u,
            u_bounds: [[0.0, 30.0], [-10.0, 10.0], [-10.0, 10.0], [-6.0, 6.0]],
            mass: 0.6,
            grav: 9.81,
        };
        let mut o = [0.0; OBS_DIM];
        situation_obs(&inp, &mut o);
        for i in 0..6 {
            assert!(o[i].abs() < 1e-6, "err entry {i} = {}", o[i]);
        }
        for i in 12..18 {
            assert!(o[i].abs() < 1e-6);
        }
        // body z in path frame is +b̂ for level flight
        assert!((o[8] - 1.0).abs() < 1e-5 && (o[11] - 1.0).abs() < 1e-5);
        assert!((o[21] - 3.0 / V_SCALE).abs() < 1e-6);
        // preview: level (cos θ = 1), thrust margin = 1 − mg/T_max
        assert!((o[22] - 1.0).abs() < 1e-5);
        assert!((o[27] - (1.0 - 0.6 * 9.81 / 30.0)).abs() < 1e-4);
        assert!(o.iter().all(|v| v.is_finite()));
    }
}
