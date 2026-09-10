//! One iteration of the control-limited DDP (box-iLQR) solver the ACMPC
//! actor is built on — a `no_std`, single-trajectory port of
//! `locuslab/mpc.pytorch` restricted to what the published method uses.
//!
//! # Scope
//!
//! The reference solver is a differentiable, batched, multi-iteration
//! solver. ACMPC as published runs it at `lqr_iter = 1` with a hover cold
//! start, and deployment needs no gradient. What remains after dropping
//! the batch dimension, the backward pass and the outer iteration loop is
//! the whole of this file:
//!
//! 1. roll the nominal control forward through the nonlinear model,
//! 2. linearize about that trajectory,
//! 3. a Riccati backward pass whose control step comes from a projected-
//!    Newton QP ([`pnqp`]) over the box, in *delta* space,
//! 4. a forward line search on the true nonlinear dynamics and true cost.
//!
//! Anything the reference would do across iterations — best-iterate
//! bookkeeping, the `full_du_norm` convergence test, `detach_unconverged`
//! — is unreachable at one iteration and is therefore absent rather than
//! written and never run.
//!
//! # Cost
//!
//! The cost is `Σₜ ½·τₜᵀCₜτₜ + cₜᵀτₜ` with `τ = [x; u]`. The cost network
//! only ever emits a **diagonal** `C`, so it is carried as a vector: the
//! quadratic form, the delta-space gradient and the `C + FᵀVF` update all
//! become `O(n)` where a dense `C` would cost `O(n²)`–`O(n³)`.

use nalgebra::{SMatrix, SVector};

use super::model::{Control, DroneDx, Jacobian, State, NTAU, NU, NX};

type Gain = SMatrix<f32, NU, NX>;
type Hessian = SMatrix<f32, NU, NU>;
type Tau = SVector<f32, NTAU>;

/// Box bounds on the control, constant over the horizon.
#[derive(Clone, Copy, Debug)]
pub struct ControlBox {
    pub lower: Control,
    pub upper: Control,
}

/// Line-search settings of the forward pass.
#[derive(Clone, Copy, Debug)]
pub struct LineSearch {
    pub decay: f32,
    pub max_iter: usize,
}

/// Per-step quadratic cost: the diagonal of `C` and the linear term `c`.
#[derive(Clone, Copy, Debug)]
pub struct QuadCost {
    pub diag: Tau,
    pub linear: Tau,
}

impl QuadCost {
    #[inline]
    fn value(&self, tau: &Tau) -> f32 {
        (0..NTAU)
            .map(|i| tau[i] * (0.5 * self.diag[i] * tau[i] + self.linear[i]))
            .sum()
    }
}

#[inline]
fn clamp(u: &Control, b: &ControlBox) -> Control {
    Control::from_fn(|i, _| u[i].clamp(b.lower[i], b.upper[i]))
}

#[inline]
fn tau(x: &State, u: &Control) -> Tau {
    let mut t = Tau::zeros();
    t.fixed_rows_mut::<NX>(0).copy_from(x);
    t.fixed_rows_mut::<NU>(NX).copy_from(u);
    t
}

/// Solve, and return the first control of the plan (receding horizon).
///
/// `u_nominal` is the cold start replicated over the horizon; `cost[t]`
/// is the stage cost the network emitted for step `t`.
pub fn solve<const T: usize>(
    x_init: &State,
    cost: &[QuadCost; T],
    dx: &DroneDx,
    bounds: &ControlBox,
    u_nominal: &Control,
    ls: &LineSearch,
) -> Control {
    // ── 1. nominal rollout ────────────────────────────────────────────
    let mut x = [State::zeros(); T];
    x[0] = *x_init;
    for t in 1..T {
        x[t] = dx.forward(&x[t - 1], u_nominal);
    }

    // ── 2. linearize about it ─────────────────────────────────────────
    // Only the jacobian is needed. The reference also forms the affine
    // offset `f = x⁺ − Ax − Bu`, but it reaches neither pass: the backward
    // pass works in delta space (where it cancels) and the forward pass
    // rolls the *true* dynamics.
    //
    // Entry T−1 is never read — the terminal stage has no successor — but
    // is still allocated, because `[_; T-1]` needs unstable const
    // arithmetic.
    let mut f_mat = [Jacobian::zeros(); T];
    for t in 0..T - 1 {
        f_mat[t] = dx.jacobian(&x[t], u_nominal);
    }

    // ── 3. backward pass ──────────────────────────────────────────────
    let mut gains = [Gain::zeros(); T];
    let mut steps = [Control::zeros(); T];
    let mut v_mat = SMatrix::<f32, NX, NX>::zeros();
    let mut v_vec = State::zeros();
    let mut warm: Option<Control> = None;
    // The nominal control is constant over the horizon, so the box in
    // delta space is too.
    let delta = ControlBox {
        lower: bounds.lower - u_nominal,
        upper: bounds.upper - u_nominal,
    };

    for t in (0..T).rev() {
        // Taylor-expand the objective about the nominal so the QP is
        // posed on `δu`, which is what makes the box bounds translate.
        let c = &cost[t];
        let tau_nom = tau(&x[t], u_nominal);
        let grad = Tau::from_fn(|i, _| c.diag[i] * tau_nom[i] + c.linear[i]);

        let (q_mat, q_vec) = if t == T - 1 {
            (SMatrix::<f32, NTAU, NTAU>::from_diagonal(&c.diag), grad)
        } else {
            let ft = f_mat[t].transpose();
            // `C` is diagonal, so it is added in place rather than
            // materialized as a second 14x14.
            let mut m = &ft * v_mat * f_mat[t];
            for i in 0..NTAU {
                m[(i, i)] += c.diag[i];
            }
            (m, grad + &ft * v_vec)
        };
        let q_xx = q_mat.fixed_view::<NX, NX>(0, 0).into_owned();
        let q_xu = q_mat.fixed_view::<NX, NU>(0, NX).into_owned();
        let q_ux = q_mat.fixed_view::<NU, NX>(NX, 0).into_owned();
        let q_uu = q_mat.fixed_view::<NU, NU>(NX, NX).into_owned();
        let q_x = q_vec.fixed_rows::<NX>(0).into_owned();
        let q_u = q_vec.fixed_rows::<NU>(NX).into_owned();

        let qp = pnqp(&q_uu, &q_u, &delta, warm.as_ref());
        warm = Some(qp.step);

        // Feedback is only taken on the free (unclamped) controls.
        let mut q_ux_free = q_ux;
        for i in 0..NU {
            if !qp.free[i] {
                q_ux_free.row_mut(i).fill(0.0);
            }
        }
        let k = -qp.free_hessian.solve(&q_ux_free).unwrap_or_else(Gain::zeros);
        let kt = k.transpose();

        v_mat = q_xx + q_xu * k + &kt * q_ux + &kt * q_uu * k;
        v_vec = q_x + q_xu * qp.step + &kt * q_u + &kt * q_uu * qp.step;
        gains[t] = k;
        steps[t] = qp.step;
    }

    // ── 4. forward line search on the true dynamics and true cost ─────
    let nominal_cost: f32 = (0..T).map(|t| cost[t].value(&tau(&x[t], u_nominal))).sum();
    let mut alpha = 1.0f32;
    let mut u0 = *u_nominal;
    for _ in 0..ls.max_iter {
        let (mut x_new, mut delta_x, mut total) = (*x_init, State::zeros(), 0.0f32);
        for t in 0..T {
            let u_new = clamp(&(gains[t] * delta_x + u_nominal + steps[t] * alpha), bounds);
            if t == 0 {
                u0 = u_new;
            }
            total += cost[t].value(&tau(&x_new, &u_new));
            if t + 1 < T {
                let next = dx.forward(&x_new, &u_new);
                delta_x = next - x[t + 1];
                x_new = next;
            }
        }
        // `!(total > nominal)` and not `total <= nominal`: a NaN cost must
        // keep shrinking the step, exactly as the reference's `any(>)` does.
        if !(total > nominal_cost) {
            break;
        }
        alpha *= ls.decay;
    }
    u0
}

struct Pnqp {
    step: Control,
    /// The Hessian restricted to the free set (constrained rows and
    /// columns zeroed, plus a `1e-11` ridge), pre-factored. Solving with
    /// it against a right-hand side whose constrained rows are zero is
    /// exactly the free-set Newton system.
    free_hessian: nalgebra::linalg::LU<f32, nalgebra::Const<NU>, nalgebra::Const<NU>>,
    free: [bool; NU],
}

/// Projected-Newton QP: `min ½·xᵀHx + qᵀx` subject to `lower ≤ x ≤ upper`.
///
/// Faithful to `mpc.pytorch`'s `pnqp`, including the exact-equality active
/// set test (the iterate is projected onto the box, so a clamped entry is
/// bit-equal to its bound) and the Armijo backtrack.
fn pnqp(h: &Hessian, q: &Control, b: &ControlBox, warm: Option<&Control>) -> Pnqp {
    const GAMMA: f32 = 0.1;
    const RIDGE: f32 = 1e-11;

    let obj = |x: &Control| 0.5 * x.dot(&(h * x)) + q.dot(x);
    let mut x = clamp(
        &match warm {
            Some(w) => *w,
            None => h.lu().solve(&-q).unwrap_or_else(Control::zeros),
        },
        b,
    );

    let mut iter = 0;
    loop {
        let g = h * x + q;
        // Exact equality is intentional: `x` was projected onto the box,
        // so a clamped coordinate *is* its bound bit for bit.
        let free: [bool; NU] = core::array::from_fn(|i| {
            !((x[i] == b.lower[i] && g[i] > 0.0) || (x[i] == b.upper[i] && g[i] < 0.0))
        });
        let g_free = Control::from_fn(|i, _| if free[i] { g[i] } else { 0.0 });
        let mut h_free = Hessian::from_fn(|r, c| if free[r] && free[c] { h[(r, c)] } else { 0.0 });
        for i in 0..NU {
            h_free[(i, i)] += RIDGE;
        }
        let lu = h_free.lu();
        let dx = lu.solve(&-g_free).unwrap_or_else(Control::zeros);

        let converged = dx.norm() < 1e-4;
        if !converged {
            // Armijo backtrack on the projected step. A non-finite ratio
            // compares false and ends the search, as in the reference.
            let (obj_x, mut alpha, mut armijo, mut candidate) = (obj(&x), 1.0f32, GAMMA, x);
            let mut count = 0;
            while armijo <= GAMMA && count < 10 {
                // Every trial steps from the *same* incumbent `x`.
                candidate = clamp(&(x + dx * alpha), b);
                armijo = (obj_x - obj(&candidate)) / g.dot(&(x - candidate));
                if armijo <= GAMMA {
                    alpha *= 0.1;
                }
                count += 1;
            }
            x = candidate;
        }
        iter += 1;
        if converged || iter == 20 {
            return Pnqp { step: x, free_hessian: lu, free };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wide_box() -> ControlBox {
        ControlBox {
            lower: Control::from_element(-1e3),
            upper: Control::from_element(1e3),
        }
    }

    /// Unconstrained, the projected-Newton QP must land on the exact
    /// stationary point `x = −H⁻¹q` in a single step.
    #[test]
    fn pnqp_solves_the_unconstrained_optimum() {
        let h = Hessian::from_diagonal(&Control::new(2.0, 3.0, 1.0, 5.0));
        let q = Control::new(1.0, -2.0, 0.5, 4.0);
        let got = pnqp(&h, &q, &wide_box(), None).step;
        let want = -h.lu().solve(&q).unwrap();
        assert!((got - want).norm() < 1e-5, "{got} vs {want}");
    }

    /// With the optimum outside the box, the solution sits on the active
    /// bound and that coordinate is reported as constrained — the flag
    /// the backward pass uses to suppress feedback.
    #[test]
    fn pnqp_clamps_to_the_active_bound() {
        let h = Hessian::identity();
        let q = Control::new(-10.0, 0.0, 0.0, 0.0);
        let b = ControlBox {
            lower: Control::from_element(-1.0),
            upper: Control::from_element(1.0),
        };
        let out = pnqp(&h, &q, &b, None);
        assert!((out.step[0] - 1.0).abs() < 1e-6, "{}", out.step[0]);
        assert!(!out.free[0], "clamped coordinate must leave the free set");
        assert!(out.free[1]);
    }

    /// The whole solver, on a cost that wants the drone at the origin at
    /// rest: the plan's first control must tilt thrust away from hover in
    /// the direction that closes the position error, and must respect the
    /// box.
    #[test]
    fn solve_respects_bounds_and_reacts_to_position_error() {
        use super::super::model::G;
        const T: usize = 5;
        let mut diag = Tau::from_element(0.0);
        for i in [0, 1, 2, 7, 8, 9] {
            diag[i] = 100.0;
        }
        for i in NX..NTAU {
            diag[i] = 1.0;
        }
        // Linear term pulls the plan toward hover thrust: p = −Q·x_ref.
        let mut linear = Tau::zeros();
        linear[NX] = -G;
        let cost = [QuadCost { diag, linear }; T];

        let bounds = ControlBox {
            lower: Control::new(0.0, -10.0, -10.0, -4.0),
            upper: Control::new(89.64, 10.0, 10.0, 4.0),
        };
        // 2 m above the target (NED: negative z is up) and level.
        let x0 = State::from_column_slice(&[0.0, 0.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let u = solve::<T>(
            &x0,
            &cost,
            &DroneDx::new(0.02),
            &bounds,
            &Control::new(G, 0.0, 0.0, 0.0),
            &LineSearch { decay: 0.2, max_iter: 5 },
        );
        for i in 0..NU {
            assert!(u[i] >= bounds.lower[i] && u[i] <= bounds.upper[i], "{u}");
        }
        // Too high: the plan must thrust *less* than hover to descend.
        assert!(u[0] < G, "expected below-hover thrust, got {}", u[0]);
    }
}
