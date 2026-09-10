//! Gate racing: the [`Track`] — observation assembly plus the gate state
//! machine — and [`RacePolicy`], which wraps it with an [`Mlp`] actor and
//! the action → motor-command map.
//!
//! [`Track`] is deliberately independent of the actor: the 20-entry
//! gate-relative observation is common to every policy trained on this
//! course, and [`crate::acmpc`] consumes exactly the same vector as the
//! prefix of its own.
//!
//! # Frames
//!
//! Everything this module exposes is **ENU world / FLU body**, cybflight's
//! native convention: position and velocity in ENU, attitude as a
//! `UnitQuaternion` mapping FLU-body → ENU-world, body rates in FLU.
//!
//! The *policy* may have been trained in a different convention. That is a
//! property of the weights, declared once via [`PolicyFrame`], and it is
//! confined to [`RacePolicy::observe`] — the caller never sees it. For
//! weights trained in this project's own simulator, use
//! [`PolicyFrame::Enu`] and the conversion vanishes.
//!
//! [`PolicyFrame::LegacyNed`] exists for policies trained against the
//! upstream RL environment (NED world / FRD body, ZYX Euler attitude). The
//! adapter converts the vehicle state *and* the gate poses into that frame
//! and then applies the upstream observation formulas verbatim — a literal
//! translation rather than a re-derivation, because a re-derivation is
//! where sign errors hide.
//!
//! # Gate state machine
//!
//! A gate counts as passed when the vehicle crosses the gate plane
//! (from behind to in front, along the gate normal) **and** is inside the
//! aperture at the crossing sample. Crossing the plane outside the aperture
//! is a clip, not a pass, and does not advance the target.
//!
//! That aperture test is deliberate. The reference C deployment advances on
//! the bare half-plane crossing, which can be satisfied by flying *around* a
//! gate — so a diverging flight silently "completes" the course. Requiring
//! the aperture keeps the target index consistent with the environment the
//! policy was trained against, and makes lap accounting mean something.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
// `f32::sin_cos` lives in `std`, not `core`. This import supplies it for the
// firmware's genuine `no_std` build. It reads as unused whenever anything
// else in the build graph links `std` (the host sim, for instance), because
// the inherent impls then resolve instead — do not let that warning talk you
// into deleting it; `just check-all` fails immediately on thumbv7em.
#[allow(unused_imports)]
use num_traits::Float;

use super::mlp::{Activation, LayerShape, Mlp, MlpError};
use crate::rotation::{
    euler_angles_rpy_to_quaternion, rotation_matrix_to_euler_angles_rpy,
};

/// Motors driven by the policy.
pub const NUM_MOTORS: usize = 4;
/// Observation entries before the look-ahead gate block.
pub const OBS_BASE: usize = 16;
/// Observation entries per look-ahead gate: relative position (3) + yaw (1).
pub const OBS_PER_GATE: usize = 4;
/// Largest track this policy will hold.
pub const MAX_GATES: usize = 16;
/// Largest look-ahead the observation buffer allows.
pub const MAX_GATES_AHEAD: usize = 4;
/// Observation buffer size.
pub const MAX_OBS: usize = OBS_BASE + OBS_PER_GATE * MAX_GATES_AHEAD;

/// Convention the *weights* were trained in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyFrame {
    /// Trained in cybflight's native ENU world / FLU body convention.
    /// The observation adapter is the identity.
    Enu,
    /// Trained in the upstream RL environment's NED world / FRD body
    /// convention. The adapter converts state and gates before assembling
    /// the observation.
    LegacyNed,
}

/// A gate, in ENU. `heading_rad` is the gate normal's bearing measured from
/// +x (East) toward +y (North) — the direction the vehicle flies *through*.
#[derive(Clone, Copy, Debug)]
pub struct Gate {
    pub position_m: Vector3<f32>,
    pub heading_rad: f32,
}

impl Gate {
    pub const fn new(position_m: Vector3<f32>, heading_rad: f32) -> Self {
        Self { position_m, heading_rad }
    }
}

/// Vehicle state the policy consumes. ENU world, FLU body.
#[derive(Clone, Copy, Debug)]
pub struct VehicleState {
    pub position_m: Vector3<f32>,
    pub velocity_m_s: Vector3<f32>,
    /// FLU body → ENU world.
    pub attitude: UnitQuaternion<f32>,
    pub body_rate_rad_s: Vector3<f32>,
    pub rotor_omega_rad_s: [f32; NUM_MOTORS],
}

/// Static configuration of a trained policy. These are properties of the
/// checkpoint, not tunables — changing one silently invalidates the weights.
#[derive(Clone, Copy, Debug)]
pub struct PolicyConfig {
    pub frame: PolicyFrame,
    /// Look-ahead gates included in the observation.
    pub gates_ahead: usize,
    /// Full gate aperture [m] (square). Pass requires being within half
    /// of this on every axis.
    pub gate_size_m: f32,
    /// Rotor-speed observation normalization: `2·(ω−min)/(max−min) − 1`.
    ///
    /// These are *normalization constants from training*, not the vehicle's
    /// physical rotor limits, and must be copied from the trainer verbatim.
    /// Substituting the airframe's real ω_max rescales every rotor
    /// observation and puts the policy off-distribution.
    pub omega_norm_min: f32,
    pub omega_norm_max: f32,
    /// Upper action clip, in the policy's [-1, 1] action space, expressed as
    /// a motor fraction: `u ≤ 2·motor_limit − 1`.
    pub motor_limit: f32,
    /// Policy output index → vehicle motor index. Identity when the trainer
    /// and the airframe agree on motor ordering (both Betaflight QuadX:
    /// rear-right, front-right, rear-left, front-left).
    pub motor_map: [u8; NUM_MOTORS],
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            frame: PolicyFrame::Enu,
            gates_ahead: 1,
            gate_size_m: 1.5,
            omega_norm_min: 0.0,
            omega_norm_max: 3000.0,
            motor_limit: 1.0,
            motor_map: [0, 1, 2, 3],
        }
    }
}

/// What happened to the target gate on the last [`RacePolicy::update_gate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateEvent {
    /// No plane crossing this step.
    None,
    /// Crossed inside the aperture. Target advanced.
    Passed { gate: usize },
    /// Crossed the plane outside the aperture. Target NOT advanced.
    Clipped { gate: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyError {
    Mlp(MlpError),
    /// Track is empty or longer than [`MAX_GATES`].
    GateCount(usize),
    /// `gates_ahead` is zero or exceeds [`MAX_GATES_AHEAD`].
    GatesAhead(usize),
    /// The network's input dimension disagrees with the observation the
    /// configured track and look-ahead produce.
    ObsDim { network: usize, config: usize },
    /// `motor_map` is not a permutation of `0..NUM_MOTORS`.
    MotorMap,
}

/// ENU → NED position/velocity: `(E,N,U) → (N,E,D)`.
#[inline]
fn enu_to_ned(v: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v.y, v.x, -v.z)
}

/// ENU heading (from +East toward +North) → NED heading (from North toward
/// East).
#[inline]
fn enu_heading_to_ned(psi: f32) -> f32 {
    core::f32::consts::FRAC_PI_2 - psi
}

/// FLU body rates → FRD body rates.
#[inline]
fn flu_to_frd(v: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v.x, -v.y, -v.z)
}

/// Wrap to `[-π, π]`.
#[inline]
fn wrap_pi(mut a: f32) -> f32 {
    const TWO_PI: f32 = core::f32::consts::TAU;
    while a > core::f32::consts::PI {
        a -= TWO_PI;
    }
    while a < -core::f32::consts::PI {
        a += TWO_PI;
    }
    a
}

/// `R_ned_frd = M · R_enu_flu · N`, with `M` mapping ENU vectors to NED and
/// `N` mapping FRD vectors to FLU. Both are involutive reflections, so this
/// is exactly the same physical rotation re-expressed.
fn attitude_in_frame(q: &UnitQuaternion<f32>, frame: PolicyFrame) -> Matrix3<f32> {
    let r = q.to_rotation_matrix().into_inner();
    match frame {
        PolicyFrame::Enu => r,
        PolicyFrame::LegacyNed => {
            let m = Matrix3::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
            let n = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
            m * r * n
        }
    }
}

/// The vehicle state as `[p(3), q_wxyz(4), v(3)]` expressed in `frame` —
/// the absolute-state block a state-consuming actor needs.
///
/// The quaternion is derived from the same Euler triple [`Track::observe`]
/// writes, through the same closed form the trainer uses.
///
/// # The sign of `q`
///
/// `q` and `−q` are the same attitude, but a *linear* cost term over the
/// state is not invariant to which one it is handed. This function always
/// returns the representative built from the wrapped Euler triple, i.e.
/// the one with `cos(ψ/2) ≥ 0`.
///
/// The upstream trainer's is not always that one: its plant carries
/// attitude as an integrated ZYX Euler triple whose yaw is never wrapped
/// (it passes 17 rad over three laps of the figure-8), so its quaternion
/// inherits the winding number of that integration. No attitude
/// representation contains that number, so it cannot be reproduced from a
/// vehicle state, and canonicalizing is the only well-defined choice — see
/// `acmpc_gate_race::action_is_insensitive_to_the_quaternion_sign`, which
/// measures what the ambiguity is worth to a trained policy.
pub fn raw_state_in_frame(state: &VehicleState, frame: PolicyFrame) -> [f32; 10] {
    let p = position_in_frame(state.position_m, frame);
    let v = position_in_frame(state.velocity_m_s, frame);
    let rpy = rotation_matrix_to_euler_angles_rpy(&attitude_in_frame(&state.attitude, frame));
    let q = euler_angles_rpy_to_quaternion(&rpy);
    [p.x, p.y, p.z, q.w, q.i, q.j, q.k, v.x, v.y, v.z]
}

/// A world-frame vector re-expressed in the policy's frame.
#[inline]
fn position_in_frame(v: Vector3<f32>, frame: PolicyFrame) -> Vector3<f32> {
    match frame {
        PolicyFrame::Enu => v,
        PolicyFrame::LegacyNed => enu_to_ned(v),
    }
}

/// A gate course bound to a policy's observation convention: assembles the
/// observation vector and runs the gate state machine.
pub struct Track {
    cfg: PolicyConfig,
    /// Track as supplied, ENU. Used for the gate state machine.
    gates_enu: [Gate; MAX_GATES],
    /// Track pre-converted into the policy frame. Used for the observation.
    gates_policy: [Gate; MAX_GATES],
    /// Gate `i` expressed in gate `i-1`'s frame, policy frame.
    rel_pos: [Vector3<f32>; MAX_GATES],
    rel_yaw: [f32; MAX_GATES],
    n_gates: usize,
    target: usize,
    gates_passed: u32,
    gates_clipped: u32,
}

impl Track {
    pub fn new(gates: &[Gate], cfg: PolicyConfig) -> Result<Self, PolicyError> {
        let n = gates.len();
        if n == 0 || n > MAX_GATES {
            return Err(PolicyError::GateCount(n));
        }
        if cfg.gates_ahead == 0 || cfg.gates_ahead > MAX_GATES_AHEAD {
            return Err(PolicyError::GatesAhead(cfg.gates_ahead));
        }
        let mut seen = [false; NUM_MOTORS];
        for &m in &cfg.motor_map {
            let m = m as usize;
            if m >= NUM_MOTORS || seen[m] {
                return Err(PolicyError::MotorMap);
            }
            seen[m] = true;
        }

        let zero = Gate::new(Vector3::zeros(), 0.0);
        let mut gates_enu = [zero; MAX_GATES];
        let mut gates_policy = [zero; MAX_GATES];
        for (i, g) in gates.iter().enumerate() {
            gates_enu[i] = *g;
            gates_policy[i] = match cfg.frame {
                PolicyFrame::Enu => *g,
                PolicyFrame::LegacyNed => {
                    Gate::new(enu_to_ned(g.position_m), enu_heading_to_ned(g.heading_rad))
                }
            };
        }

        // Gate i expressed in gate (i-1)'s frame — the look-ahead features.
        // Wraps at i=0 to gate n-1, matching the looped-track convention.
        let mut rel_pos = [Vector3::zeros(); MAX_GATES];
        let mut rel_yaw = [0.0f32; MAX_GATES];
        for i in 0..n {
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let d = gates_policy[i].position_m - gates_policy[prev].position_m;
            let (s, c) = gates_policy[prev].heading_rad.sin_cos();
            rel_pos[i] = Vector3::new(c * d.x + s * d.y, -s * d.x + c * d.y, d.z);
            rel_yaw[i] = wrap_pi(gates_policy[i].heading_rad - gates_policy[prev].heading_rad);
        }

        Ok(Self {
            cfg,
            gates_enu,
            gates_policy,
            rel_pos,
            rel_yaw,
            n_gates: n,
            target: 0,
            gates_passed: 0,
            gates_clipped: 0,
        })
    }

    pub fn config(&self) -> &PolicyConfig {
        &self.cfg
    }

    pub fn obs_len(&self) -> usize {
        OBS_BASE + OBS_PER_GATE * self.cfg.gates_ahead
    }

    pub fn target_gate(&self) -> usize {
        self.target
    }

    pub fn gates_passed(&self) -> u32 {
        self.gates_passed
    }

    pub fn gates_clipped(&self) -> u32 {
        self.gates_clipped
    }

    pub fn laps_completed(&self) -> u32 {
        self.gates_passed / self.n_gates as u32
    }

    pub fn num_gates(&self) -> usize {
        self.n_gates
    }

    /// Reset to the first gate. Call on mode entry.
    pub fn reset(&mut self) {
        self.target = 0;
        self.gates_passed = 0;
        self.gates_clipped = 0;
    }

    /// Force the target gate (for entering the course mid-track).
    pub fn set_target_gate(&mut self, gate: usize) {
        self.target = gate % self.n_gates;
    }

    /// Assemble the observation vector. Writes `self.obs_len()` entries.
    pub fn observe(&self, state: &VehicleState, obs: &mut [f32]) {
        let g = self.gates_policy[self.target];

        // Vehicle state expressed in the policy's frame.
        let pos = position_in_frame(state.position_m, self.cfg.frame);
        let vel = position_in_frame(state.velocity_m_s, self.cfg.frame);
        let rate = match self.cfg.frame {
            PolicyFrame::Enu => state.body_rate_rad_s,
            PolicyFrame::LegacyNed => flu_to_frd(state.body_rate_rad_s),
        };
        let rpy = rotation_matrix_to_euler_angles_rpy(&attitude_in_frame(
            &state.attitude,
            self.cfg.frame,
        ));

        let (s, c) = g.heading_rad.sin_cos();
        let d = pos - g.position_m;

        // Position and velocity in the target gate's frame.
        obs[0] = c * d.x + s * d.y;
        obs[1] = -s * d.x + c * d.y;
        obs[2] = d.z;
        obs[3] = c * vel.x + s * vel.y;
        obs[4] = -s * vel.x + c * vel.y;
        obs[5] = vel.z;

        // Attitude: roll and pitch absolute, yaw relative to the gate.
        obs[6] = rpy[0];
        obs[7] = rpy[1];
        obs[8] = wrap_pi(rpy[2] - g.heading_rad);

        obs[9] = rate.x;
        obs[10] = rate.y;
        obs[11] = rate.z;

        let span = self.cfg.omega_norm_max - self.cfg.omega_norm_min;
        for i in 0..NUM_MOTORS {
            let w = state.rotor_omega_rad_s[self.cfg.motor_map[i] as usize];
            obs[12 + i] = (w - self.cfg.omega_norm_min) * 2.0 / span - 1.0;
        }

        for i in 0..self.cfg.gates_ahead {
            let idx = (self.target + i + 1) % self.n_gates;
            let base = OBS_BASE + OBS_PER_GATE * i;
            obs[base] = self.rel_pos[idx].x;
            obs[base + 1] = self.rel_pos[idx].y;
            obs[base + 2] = self.rel_pos[idx].z;
            obs[base + 3] = self.rel_yaw[idx];
        }
    }

    /// Advance the gate state machine over one motion segment, ENU.
    ///
    /// Call once per plant step with the positions bracketing that step.
    pub fn update_gate(&mut self, prev_pos: Vector3<f32>, new_pos: Vector3<f32>) -> GateEvent {
        let g = self.gates_enu[self.target];
        let (s, c) = g.heading_rad.sin_cos();
        let proj = |p: Vector3<f32>| {
            let d = p - g.position_m;
            c * d.x + s * d.y
        };
        if !(proj(prev_pos) < 0.0 && proj(new_pos) > 0.0) {
            return GateEvent::None;
        }
        let d = new_pos - g.position_m;
        let half = 0.5 * self.cfg.gate_size_m;
        let inside = d.x.abs() < half && d.y.abs() < half && d.z.abs() < half;
        let gate = self.target;
        if inside {
            self.gates_passed += 1;
            self.target = (self.target + 1) % self.n_gates;
            GateEvent::Passed { gate }
        } else {
            self.gates_clipped += 1;
            GateEvent::Clipped { gate }
        }
    }
}

/// A trained gate-racing policy: a [`Track`] plus an [`Mlp`] actor that
/// maps the observation straight to per-motor commands.
pub struct RacePolicy<'a> {
    pub track: Track,
    mlp: Mlp<'a>,
}

impl<'a> RacePolicy<'a> {
    pub fn new(
        weights: &'a [f32],
        shapes: &'a [LayerShape],
        gates: &[Gate],
        cfg: PolicyConfig,
    ) -> Result<Self, PolicyError> {
        let mlp = Mlp::new(weights, shapes, Activation::Relu).map_err(PolicyError::Mlp)?;
        let track = Track::new(gates, cfg)?;
        if mlp.input_dim() != track.obs_len() {
            return Err(PolicyError::ObsDim {
                network: mlp.input_dim(),
                config: track.obs_len(),
            });
        }
        if mlp.output_dim() != NUM_MOTORS {
            return Err(PolicyError::ObsDim { network: mlp.output_dim(), config: NUM_MOTORS });
        }
        Ok(Self { track, mlp })
    }

    /// Map a raw network output to normalized motor commands in `[0, 1]`.
    ///
    /// The policy's action space is `[-1, u_max]`; ESCs take `[0, 1]`.
    /// Deterministic — the training-time Gaussian exploration noise is
    /// **not** reproduced. That noise is an artifact of the stochastic
    /// policy PPO optimizes, not part of the controller.
    pub fn actions_to_commands(&self, raw: &[f32; NUM_MOTORS]) -> [f32; NUM_MOTORS] {
        let cfg = self.track.config();
        let u_max = 2.0 * cfg.motor_limit - 1.0;
        let mut out = [0.0f32; NUM_MOTORS];
        for i in 0..NUM_MOTORS {
            let v = if raw[i].is_finite() { raw[i] } else { -1.0 };
            out[cfg.motor_map[i] as usize] = (v.clamp(-1.0, u_max) + 1.0) * 0.5;
        }
        out
    }

    /// Observation → network → motor commands, in one call.
    pub fn step(&self, state: &VehicleState) -> Result<[f32; NUM_MOTORS], PolicyError> {
        let mut obs = [0.0f32; MAX_OBS];
        let n = self.track.obs_len();
        self.track.observe(state, &mut obs[..n]);
        let mut raw = [0.0f32; NUM_MOTORS];
        self.mlp.forward(&obs[..n], &mut raw).map_err(PolicyError::Mlp)?;
        Ok(self.actions_to_commands(&raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enu_ned_position_roundtrip_is_involutive() {
        let v = Vector3::new(1.0, -2.0, 3.0);
        assert_eq!(enu_to_ned(enu_to_ned(v)), v);
    }

    #[test]
    fn enu_ned_heading_roundtrip_is_involutive() {
        for &p in &[0.0f32, 0.7, -1.3, 3.0] {
            let back = enu_heading_to_ned(enu_heading_to_ned(p));
            assert!((back - p).abs() < 1e-6);
        }
    }

    /// A heading is the bearing of the gate normal. Converting the heading
    /// and converting the normal vector must agree — this is the identity
    /// that makes the gate-frame projection frame-invariant.
    #[test]
    fn heading_conversion_matches_normal_vector_conversion() {
        for &psi_enu in &[0.0f32, 0.5, 1.9, -2.4] {
            let n_enu = Vector3::new(psi_enu.cos(), psi_enu.sin(), 0.0);
            let n_ned = enu_to_ned(n_enu);
            let psi_ned = enu_heading_to_ned(psi_enu);
            assert!((n_ned.x - psi_ned.cos()).abs() < 1e-6, "psi={psi_enu}");
            assert!((n_ned.y - psi_ned.sin()).abs() < 1e-6, "psi={psi_enu}");
        }
    }

    /// Level flight facing East in ENU is heading π/2 in NED, with zero
    /// roll and pitch in both.
    #[test]
    fn level_attitude_converts_to_expected_ned_euler() {
        let q = UnitQuaternion::identity();
        let rpy = rotation_matrix_to_euler_angles_rpy(&attitude_in_frame(
            &q,
            PolicyFrame::LegacyNed,
        ));
        assert!(rpy[0].abs() < 1e-6, "roll {}", rpy[0]);
        assert!(rpy[1].abs() < 1e-6, "pitch {}", rpy[1]);
        assert!(
            (rpy[2] - core::f32::consts::FRAC_PI_2).abs() < 1e-6,
            "yaw {}",
            rpy[2]
        );
    }

    /// Nose-up pitch in FLU must read as nose-up in FRD too (the sign of
    /// pitch flips with the axis, and the Euler extraction must undo it).
    #[test]
    fn pitch_up_in_flu_reads_nose_up_in_frd() {
        // +0.2 rad about body y in FLU tilts the nose DOWN (right-hand rule
        // about +y_left is nose-down), so FRD pitch must be negative.
        let q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.2);
        let rpy = rotation_matrix_to_euler_angles_rpy(&attitude_in_frame(
            &q,
            PolicyFrame::LegacyNed,
        ));
        assert!((rpy[1] + 0.2).abs() < 1e-5, "frd pitch {}", rpy[1]);
    }

    #[test]
    fn body_rate_conversion_is_involutive() {
        let w = Vector3::new(0.3, -1.2, 0.7);
        assert_eq!(flu_to_frd(flu_to_frd(w)), w);
    }
}
