//! The differentiable MPC's internal prediction model: a 10-state CTBR
//! quadrotor.
//!
//! State `[p(3), q_wxyz(4), v(3)]`, control `[a_c, ω(3)]` — specific
//! collective thrust (m/s², along body **−z**) and body rates, assumed
//! perfectly tracked. Specific thrust makes the model mass-free.
//!
//! Frames follow the policy's convention ([`crate::nn::PolicyFrame`]): for
//! the published checkpoint that is NED world / FRD body, which is why
//! gravity enters as `+G` on the third axis and thrust as `−a_c·z_b`.
//!
//! This is deliberately **simpler than any plant**: no motor lag, no drag,
//! no rate-loop tracking dynamics. The ACMPC premise is that the learned
//! cost compensates for the model mismatch, so "fixing" the model here
//! would take the solver off the distribution its cost network was
//! trained against.
//!
//! [`DroneDx::forward`] integrates position and velocity with explicit
//! Euler and attitude with the exact quaternion exponential (so the
//! quaternion stays unit), while [`DroneDx::jacobian`] linearizes the
//! *Euler* step. The `O(dt²)` inconsistency between the two is inherited
//! from the reference implementation and is immaterial at one DDP
//! iteration.

use nalgebra::{SMatrix, SVector};
#[allow(unused_imports)]
use num_traits::Float;

/// Prediction state dimension.
pub const NX: usize = 10;
/// Control dimension: `[a_c, ωx, ωy, ωz]`.
pub const NU: usize = 4;
/// Stacked `τ = [x; u]` dimension.
pub const NTAU: usize = NX + NU;
/// Gravity the trained cost network was fitted against [m/s²].
pub const G: f32 = 9.81;

pub type State = SVector<f32, NX>;
pub type Control = SVector<f32, NU>;
/// `[A | B]` — the Euler-discretized jacobian block the DDP consumes.
pub type Jacobian = SMatrix<f32, NX, NTAU>;

/// Hamilton product of `w`-first quaternions.
#[inline]
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let ([w1, x1, y1, z1], [w2, x2, y2, z2]) = (a, b);
    [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
}

/// The body z axis in world coordinates — the third column of `R(q)`.
#[inline]
fn body_z_world(q: &[f32]) -> SVector<f32, 3> {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    SVector::<f32, 3>::new(
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        w * w - x * x - y * y + z * z,
    )
}

/// The 10-state CTBR model at a fixed prediction step.
///
/// `dt` may be coarser than the control period — the model only ever
/// predicts, it never integrates a plant.
#[derive(Clone, Copy, Debug)]
pub struct DroneDx {
    pub dt: f32,
}

impl DroneDx {
    pub const fn new(dt: f32) -> Self {
        Self { dt }
    }

    /// One prediction step.
    pub fn forward(&self, x: &State, u: &Control) -> State {
        let dt = self.dt;
        let (a_c, half) = (u[0], 0.5 * dt);

        // Quaternion exponential of ω·dt. `sin(θ)/‖ω‖` is the ω → 0 limit
        // `dt/2`, taken from the series so the model stays finite at rest
        // (a hover cold start hits this on the first call, every call).
        let w_norm = (u[1] * u[1] + u[2] * u[2] + u[3] * u[3]).sqrt();
        let theta = w_norm * half;
        let scale = if theta < 1e-4 {
            half * (1.0 - theta * theta / 6.0)
        } else {
            theta.sin() / w_norm
        };
        let dq = [theta.cos(), u[1] * scale, u[2] * scale, u[3] * scale];
        let q = quat_mul([x[3], x[4], x[5], x[6]], dq);

        let z_b = body_z_world(&x.as_slice()[3..7]);
        let mut out = State::zeros();
        for i in 0..3 {
            out[i] = x[i] + dt * x[7 + i];
            out[7 + i] = x[7 + i] - dt * a_c * z_b[i];
        }
        out[9] += dt * G;
        out.fixed_rows_mut::<4>(3).copy_from_slice(&q);
        out
    }

    /// `[A | B] = [I + dt·∂f/∂x | dt·∂f/∂u]` of the continuous dynamics
    /// `f = [v, ½·q ⊗ [0, ω], −a_c·R(q)e₃ + G·e₃]`.
    pub fn jacobian(&self, x: &State, u: &Control) -> Jacobian {
        let dt = self.dt;
        let (w, qx, qy, qz) = (x[3], x[4], x[5], x[6]);
        let (a2, wx, wy, wz) = (2.0 * u[0], u[1], u[2], u[3]);
        let mut j = Jacobian::zeros();

        // ∂ṗ/∂v = I₃
        for i in 0..3 {
            j[(i, 7 + i)] = dt;
        }
        // ∂q̇/∂q = ½·R(0, ω) — right-multiplication by the pure rate.
        let half_dt = 0.5 * dt;
        for (r, row) in [
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ]
        .into_iter()
        .enumerate()
        {
            for (c, v) in row.into_iter().enumerate() {
                j[(3 + r, 3 + c)] = half_dt * v;
            }
            // ∂q̇/∂ω = ½·L(q)[:, 1:4] — left-multiplication columns.
            j[(3 + r, NX + 1)] = half_dt * [-qx, w, qz, -qy][r];
            j[(3 + r, NX + 2)] = half_dt * [-qy, -qz, w, qx][r];
            j[(3 + r, NX + 3)] = half_dt * [-qz, qy, -qx, w][r];
        }
        // ∂v̇/∂q = −a_c·∂(R(q)e₃)/∂q
        for (r, row) in [
            [a2 * qy, a2 * qz, a2 * w, a2 * qx],
            [-a2 * qx, -a2 * w, a2 * qz, a2 * qy],
            [a2 * w, -a2 * qx, -a2 * qy, a2 * qz],
        ]
        .into_iter()
        .enumerate()
        {
            for (c, v) in row.into_iter().enumerate() {
                j[(7 + r, 3 + c)] = -dt * v;
            }
        }
        // ∂v̇/∂a_c = −R(q)e₃
        let z_b = body_z_world(&x.as_slice()[3..7]);
        for i in 0..3 {
            j[(7 + i, 10)] = -dt * z_b[i];
            j[(i, i)] += 1.0;
            j[(7 + i, 7 + i)] += 1.0;
        }
        for i in 3..7 {
            j[(i, i)] += 1.0;
        }
        j
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hover_state() -> State {
        State::from_column_slice(&[0.0, 0.0, -1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }

    /// Level hover at `a_c = G` is an equilibrium in velocity: thrust
    /// along body −z exactly cancels the `+G` on the NED down axis.
    #[test]
    fn hover_thrust_cancels_gravity() {
        let dx = DroneDx::new(0.02);
        let next = dx.forward(&hover_state(), &Control::new(G, 0.0, 0.0, 0.0));
        assert!(next.fixed_rows::<3>(7).norm() < 1e-6, "{next}");
    }

    /// The exponential map is exact, so a constant body rate integrates to
    /// a pure rotation with unit norm — no drift to renormalize away.
    #[test]
    fn attitude_integration_preserves_unit_norm() {
        let dx = DroneDx::new(0.02);
        let mut x = hover_state();
        let u = Control::new(G, 3.0, -2.0, 1.0);
        for _ in 0..100 {
            x = dx.forward(&x, &u);
        }
        assert!((x.fixed_rows::<4>(3).norm() - 1.0).abs() < 1e-5);
    }

    /// The analytic jacobian must agree with a central difference of the
    /// Euler step — the linearization the DDP actually consumes.
    #[test]
    fn jacobian_matches_central_differences() {
        let dx = DroneDx::new(0.02);
        let x = State::from_column_slice(&[
            0.3, -1.0, -1.7, 0.966, 0.13, -0.18, 0.1, 1.5, -0.4, 0.2,
        ]);
        let u = Control::new(14.0, 0.8, -0.5, 0.3);
        let j = dx.jacobian(&x, &u);

        // Euler step: the jacobian linearizes this, not `forward`'s exp map.
        let euler = |x: &State, u: &Control| {
            let mut out = *x;
            let q = [x[3], x[4], x[5], x[6]];
            let z_b = body_z_world(&q);
            let d = [
                0.5 * (-q[1] * u[1] - q[2] * u[2] - q[3] * u[3]),
                0.5 * (q[0] * u[1] - q[3] * u[2] + q[2] * u[3]),
                0.5 * (q[3] * u[1] + q[0] * u[2] - q[1] * u[3]),
                0.5 * (-q[2] * u[1] + q[1] * u[2] + q[0] * u[3]),
            ];
            for i in 0..3 {
                out[i] += dx.dt * x[7 + i];
                out[3 + i] += dx.dt * d[i];
                out[7 + i] -= dx.dt * u[0] * z_b[i];
            }
            out[6] += dx.dt * d[3];
            out[9] += dx.dt * G;
            out
        };

        const EPS: f32 = 1e-3;
        for c in 0..NTAU {
            let (mut xp, mut xm, mut up, mut um) = (x, x, u, u);
            if c < NX {
                xp[c] += EPS;
                xm[c] -= EPS;
            } else {
                up[c - NX] += EPS;
                um[c - NX] -= EPS;
            }
            let fd = (euler(&xp, &up) - euler(&xm, &um)) / (2.0 * EPS);
            for r in 0..NX {
                assert!(
                    (j[(r, c)] - fd[r]).abs() < 2e-3,
                    "({r},{c}): analytic {} vs fd {}",
                    j[(r, c)],
                    fd[r]
                );
            }
        }
    }
}
