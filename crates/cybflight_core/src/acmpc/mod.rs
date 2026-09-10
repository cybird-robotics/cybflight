//! Actor-Critic Model Predictive Control (ACMPC) — a differentiable MPC
//! used as a control policy.
//!
//! Romero et al., *Actor-Critic Model Predictive Control*, IEEE T-RO 2025.
//! A network maps the gate-relative observation to the **cost** of a short
//! optimal-control problem; a box-constrained DDP solves that problem
//! against a simplified quadrotor model ([`model::DroneDx`]), and the
//! first control of the plan is the command. Training backpropagates
//! PPO's surrogate loss through the solve into the cost network; nothing
//! of that is needed to fly it, so this module is the forward path only.
//!
//! # Relationship to [`crate::mpc`]
//!
//! Both produce collective thrust plus a body-rate reference at the same
//! point in the stack, and are alternatives at that point:
//!
//! | | [`crate::mpc`] | this module |
//! |---|---|---|
//! | cost | hand-tuned weights on a tracked reference | learned, per-observation, time-varying |
//! | input | a reference trajectory | a gate course |
//! | solver | SQP | box-DDP, one iteration |
//!
//! The learned cost carries no explicit reference: the linear term `p`
//! takes over the role of `−Q·x_ref`, which is why the MPC must be handed
//! the vehicle's **absolute** state rather than a gate-relative one.
//!
//! # Observation
//!
//! `[ 20-entry gate-relative observation | p(3), q_wxyz(4), v(3) ]`. The
//! prefix is byte-identical to what [`crate::nn::RacePolicy`] consumes —
//! same [`Track`], same code — so the two methods are comparable by
//! construction. Only the cost network sees the prefix; only the solver
//! sees the suffix.
//!
//! # Frames
//!
//! In and out, this module speaks cybflight's native ENU world / FLU
//! body. The policy frame ([`PolicyFrame`]) is a property of the weights
//! and is confined to the observation adapter, exactly as in
//! [`crate::nn::race_policy`].

pub mod ddp;
pub mod model;

use nalgebra::Vector3;

use crate::nn::mlp::{Activation, LayerShape, Mlp};
use crate::nn::race_policy::{
    raw_state_in_frame, Gate, PolicyConfig, PolicyError, PolicyFrame, Track, VehicleState,
    MAX_OBS as MAX_TRACK_OBS,
};
use ddp::{ControlBox, LineSearch, QuadCost};
use model::{Control, DroneDx, State, G, NTAU, NU, NX};

/// Prediction steps in the plan. Fixed by the published ACMPC settings
/// and baked into the cost network's output width, which
/// [`AcmpcPolicy::new`] checks — a checkpoint trained at another horizon
/// is rejected rather than silently mis-decoded.
pub const HORIZON: usize = 5;

/// Cost-network output width: a diagonal `Q` and a linear `p` per step.
pub const COST_OUT: usize = 2 * HORIZON * NTAU;

/// Widest cost-network layer supported (the published net is 512).
pub const COST_NET_MAX_WIDTH: usize = 512;

/// Largest observation this policy assembles: the widest gate-relative
/// block a [`Track`] produces, plus the raw-state suffix.
pub const MAX_OBS: usize = MAX_TRACK_OBS + NX;

/// The CTBR action space the policy commands into.
///
/// These are the **normalization constants of the action space**, not the
/// airframe's capability: the map from the network's `[-1, 1]` output to
/// physical units is part of the trained policy, and substituting a
/// vehicle's real limits rescales every command it ever learned.
#[derive(Clone, Copy, Debug)]
pub struct CtbrLimits {
    /// Specific collective thrust at action `+1` [m/s²]; action `−1` is 0.
    pub max_specific_thrust: f32,
    /// Per-axis body rate at action `±1` [rad/s].
    pub max_body_rate: Vector3<f32>,
}

impl Default for CtbrLimits {
    fn default() -> Self {
        Self {
            // 4·k_w·ω_max² of the identified 5-inch quad, ≈ 9.1 g.
            max_specific_thrust: 89.64,
            max_body_rate: Vector3::new(10.0, 10.0, 4.0),
        }
    }
}

/// Static configuration of a trained ACMPC checkpoint. Every field is a
/// property of the weights; changing one invalidates them.
#[derive(Clone, Copy, Debug)]
pub struct AcmpcConfig {
    pub policy: PolicyConfig,
    pub ctbr: CtbrLimits,
    /// Prediction step of [`model::DroneDx`] [s]. May be coarser than the
    /// control period — the model only predicts.
    pub mpc_dt: f32,
    /// Sigmoid output scales of the cost head.
    pub range_q: f32,
    pub range_p: f32,
    pub line_search: LineSearch,
}

impl Default for AcmpcConfig {
    fn default() -> Self {
        Self {
            policy: PolicyConfig::default(),
            ctbr: CtbrLimits::default(),
            mpc_dt: 0.02,
            range_q: 1e5,
            range_p: 1e5,
            line_search: LineSearch { decay: 0.2, max_iter: 5 },
        }
    }
}

/// Collective thrust and body-rate reference, **ENU world / FLU body** —
/// the same interface [`crate::mpc`]'s outer loop emits, so the inner rate
/// loop does not care which one produced it.
#[derive(Clone, Copy, Debug)]
pub struct CtbrCommand {
    /// Specific collective thrust along body `+z` (FLU) [m/s²].
    pub specific_thrust_m_s2: f32,
    pub body_rate_rad_s: Vector3<f32>,
}

/// A trained ACMPC policy bound to a gate course.
pub struct AcmpcPolicy<'a> {
    pub track: Track,
    cost_net: Mlp<'a, COST_NET_MAX_WIDTH>,
    dx: DroneDx,
    bounds: ControlBox,
    cfg: AcmpcConfig,
}

impl<'a> AcmpcPolicy<'a> {
    pub fn new(
        weights: &'a [f32],
        shapes: &'a [LayerShape],
        gates: &[Gate],
        cfg: AcmpcConfig,
    ) -> Result<Self, PolicyError> {
        let cost_net = Mlp::new(weights, shapes, Activation::Gelu).map_err(PolicyError::Mlp)?;
        let track = Track::new(gates, cfg.policy)?;
        if cost_net.input_dim() != track.obs_len() {
            return Err(PolicyError::ObsDim {
                network: cost_net.input_dim(),
                config: track.obs_len(),
            });
        }
        if cost_net.output_dim() != COST_OUT {
            return Err(PolicyError::ObsDim { network: cost_net.output_dim(), config: COST_OUT });
        }
        let w = cfg.ctbr.max_body_rate;
        Ok(Self {
            track,
            cost_net,
            dx: DroneDx::new(cfg.mpc_dt),
            bounds: ControlBox {
                lower: Control::new(0.0, -w.x, -w.y, -w.z),
                upper: Control::new(cfg.ctbr.max_specific_thrust, w.x, w.y, w.z),
            },
            cfg,
        })
    }

    /// Full observation width: gate-relative prefix plus raw state.
    pub fn obs_len(&self) -> usize {
        self.track.obs_len() + NX
    }

    /// Assemble the observation. Writes [`Self::obs_len`] entries.
    pub fn observe(&self, state: &VehicleState, obs: &mut [f32]) {
        let n = self.track.obs_len();
        self.track.observe(state, &mut obs[..n]);
        obs[n..n + NX].copy_from_slice(&raw_state_in_frame(state, self.cfg.policy.frame));
    }

    /// Observation → normalized CTBR action in `[-1, 1]⁴`.
    ///
    /// Deterministic: the trained Gaussian's `log_std` is exploration
    /// noise PPO needs, not part of the controller.
    pub fn action(&self, obs: &[f32]) -> Result<[f32; NU], PolicyError> {
        let n = self.track.obs_len();
        if obs.len() < n + NX {
            return Err(PolicyError::ObsDim { network: n + NX, config: obs.len() });
        }
        let mut logits = [0.0f32; COST_OUT];
        self.cost_net.forward(&obs[..n], &mut logits).map_err(PolicyError::Mlp)?;

        // Sigmoid head → per-step diagonal Q and linear p. `+0.1` keeps Q
        // strictly positive (so it is PSD); the thrust entry of p gets the
        // published asymmetric scaling that biases the plan toward
        // positive collective.
        let sig = |v: f32| 1.0 / (1.0 + libm::expf(-v));
        let cost = core::array::from_fn::<_, HORIZON, _>(|t| {
            let (q0, p0) = (t * NTAU, (HORIZON + t) * NTAU);
            QuadCost {
                diag: nalgebra::SVector::from_fn(|i, _| {
                    sig(logits[q0 + i]) * self.cfg.range_q + 0.1
                }),
                linear: nalgebra::SVector::from_fn(|i, _| {
                    let s = sig(logits[p0 + i]);
                    if i == NX {
                        -(s * self.cfg.range_q * G + 0.1)
                    } else {
                        (s - 0.5) * self.cfg.range_p
                    }
                }),
            }
        });

        let x_init = State::from_column_slice(&obs[n..n + NX]);
        let u = ddp::solve::<HORIZON>(
            &x_init,
            &cost,
            &self.dx,
            &self.bounds,
            // Hover cold start: the published ACMPC runs one DDP
            // iteration from rest, never a warm start. Carrying the last
            // plan forward would make the policy stateful, which is
            // incoherent under the shuffled minibatches it was trained on.
            &Control::new(G, 0.0, 0.0, 0.0),
            &self.cfg.line_search,
        );

        // Normalize, then guard: a diverged solve must not command NaN.
        let w = self.cfg.ctbr.max_body_rate;
        let action = [
            2.0 * u[0] / self.cfg.ctbr.max_specific_thrust - 1.0,
            u[1] / w.x,
            u[2] / w.y,
            u[3] / w.z,
        ];
        Ok(if action.iter().all(|v| v.is_finite()) {
            action.map(|v| v.clamp(-1.0, 1.0))
        } else {
            [2.0 * G / self.cfg.ctbr.max_specific_thrust - 1.0, 0.0, 0.0, 0.0]
        })
    }

    /// Observation → solve → command, in one call. ENU world / FLU body.
    pub fn step(&self, state: &VehicleState) -> Result<CtbrCommand, PolicyError> {
        let mut obs = [0.0f32; MAX_OBS];
        let n = self.obs_len();
        self.observe(state, &mut obs[..n]);
        Ok(self.command(&self.action(&obs[..n])?))
    }

    /// Normalized action → physical command, FRD rates flipped into FLU.
    pub fn command(&self, action: &[f32; NU]) -> CtbrCommand {
        let w = self.cfg.ctbr.max_body_rate;
        CtbrCommand {
            specific_thrust_m_s2: (action[0] + 1.0) * 0.5 * self.cfg.ctbr.max_specific_thrust,
            // FRD rates → FLU: roll shares its axis, pitch and yaw negate.
            body_rate_rad_s: match self.cfg.policy.frame {
                PolicyFrame::Enu => Vector3::new(action[1] * w.x, action[2] * w.y, action[3] * w.z),
                PolicyFrame::LegacyNed => {
                    Vector3::new(action[1] * w.x, -action[2] * w.y, -action[3] * w.z)
                }
            },
        }
    }
}
