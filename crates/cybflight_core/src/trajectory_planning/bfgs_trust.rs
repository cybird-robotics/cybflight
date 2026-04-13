//! Full BFGS optimizer with trust-region globalization (Cauchy-dogleg).
//! Zero heap allocations. Stores the full n×n Hessian approximation B.
//!
//! For n≤64, the Hessian is at most 64×64 = 32 KB — trivially fits on stack.
//! Cholesky solve per iteration: O(n³/6) ≈ 43K flops for n=64.

#[allow(unused_imports)]
use num_traits::Float;

pub use crate::params::BfgsTrustParams;

const MAX_VARS: usize = 4 * super::MAX_PIECES; // 64
const MAX_VARS_SQ: usize = MAX_VARS * MAX_VARS; // 4096

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BfgsTrustResult {
    Convergence,
    Stop,
    MaxIterations,
    InvalidValue,
}

/// Run full BFGS with trust-region optimization.
///
/// - `x`: initial guess, overwritten with solution
/// - `cost_grad`: closure `|x, grad| -> cost` computing cost and filling gradient
/// - `params`: optimizer parameters
///
/// Returns `(result, final_cost, iterations)`.
pub fn bfgs_trust_optimize<F>(
    x: &mut [f32],
    cost_grad: &mut F,
    params: &BfgsTrustParams,
) -> (BfgsTrustResult, f32, usize)
where
    F: FnMut(&[f32], &mut [f32]) -> f32,
{
    let n = x.len();
    debug_assert!(n <= MAX_VARS);

    // Hessian approximation B (n×n, row-major), initialized to identity
    let mut hess = [0.0f32; MAX_VARS_SQ];
    for i in 0..n {
        hess[i * n + i] = 1.0;
    }

    let mut g = [0.0f32; MAX_VARS];
    let mut g_new = [0.0f32; MAX_VARS];
    let mut p = [0.0f32; MAX_VARS]; // step
    let mut x_trial = [0.0f32; MAX_VARS];
    let mut s = [0.0f32; MAX_VARS]; // x_{k+1} - x_k
    let mut y = [0.0f32; MAX_VARS]; // g_{k+1} - g_k
    let mut bs = [0.0f32; MAX_VARS]; // B * s

    // Cached dogleg quantities (Newton point, Cauchy point, B*g).
    // Valid until B or g changes (i.e. until a step is accepted).
    let mut p_newton = [0.0f32; MAX_VARS];
    let mut p_cauchy = [0.0f32; MAX_VARS];
    let mut bg = [0.0f32; MAX_VARS];
    let mut newton_ok = false;
    let mut pn_norm = 0.0f32;
    let mut pc_norm = 0.0f32;
    let mut gtg = 0.0f32;
    let mut cache_valid = false;

    // Past function values for convergence test
    let mut past_f = [0.0f32; 16];

    let mut delta = params.delta_init;
    let mut fx = cost_grad(x, &mut g[..n]);
    if !fx.is_finite() {
        return (BfgsTrustResult::InvalidValue, fx, 0);
    }

    if params.past > 0 {
        past_f[0] = fx;
    }

    for k in 0..params.max_iterations.max(1000) {
        // Convergence check: ||g||_inf / max(1, ||x||_inf) < g_epsilon
        let xnorm = vec_norm_inf(&x[..n]).max(1.0);
        let gnorm = vec_norm_inf(&g[..n]);
        if gnorm / xnorm < params.g_epsilon {
            return (BfgsTrustResult::Convergence, fx, k);
        }

        if params.max_iterations > 0 && k >= params.max_iterations {
            return (BfgsTrustResult::MaxIterations, fx, k);
        }

        // Compute Newton and Cauchy points only when B or g has changed.
        if !cache_valid {
            // Newton point: p_n = -B^{-1} g via Cholesky (O(n³/6))
            newton_ok = cholesky_solve(&hess, &g[..n], &mut p_newton[..n], n);
            pn_norm = if newton_ok { vec_norm(&p_newton[..n]) } else { 0.0 };

            // Cauchy point: p_c = -alpha * g where alpha = ||g||² / (g^T B g)
            mat_vec(&hess, &g[..n], &mut bg[..n], n);
            gtg = vec_dot(&g[..n], &g[..n]);
            let g_bg = vec_dot(&g[..n], &bg[..n]);

            if g_bg > 0.0 && gtg >= 1e-7 {
                let alpha_c = gtg / g_bg;
                for i in 0..n { p_cauchy[i] = -alpha_c * g[i]; }
                pc_norm = vec_norm(&p_cauchy[..n]);
            } else {
                // Negative curvature or zero gradient
                pc_norm = 0.0;
            }

            cache_valid = true;
        }

        // Compute dogleg step from cached Newton/Cauchy points.
        let pred = dogleg_step_cached(
            &hess, &g[..n], delta,
            &p_newton[..n], newton_ok, pn_norm,
            &p_cauchy[..n], pc_norm, gtg,
            &mut p[..n], n,
        );

        if pred.abs() < 1e-7 {
            return (BfgsTrustResult::Convergence, fx, k);
        }

        // Evaluate trial point
        for i in 0..n {
            x_trial[i] = x[i] + p[i];
        }
        let fx_trial = cost_grad(&x_trial[..n], &mut g_new[..n]);

        let actual = fx - fx_trial;
        let rho = if pred.abs() > 1e-7 { actual / pred } else { 0.0 };

        // Update trust region radius
        if rho < 0.25 {
            delta *= 0.25;
        } else if rho > 0.75 {
            let pnorm = vec_norm(&p[..n]);
            if pnorm > 0.99 * delta {
                // Step was at boundary — expand
                delta = (2.0 * delta).min(params.delta_max);
            }
        }

        // Accept or reject
        if rho > params.eta && fx_trial.is_finite() {
            // Compute s and y for BFGS update
            for i in 0..n {
                s[i] = p[i];
                y[i] = g_new[i] - g[i];
            }

            // Accept step
            x[..n].copy_from_slice(&x_trial[..n]);
            g[..n].copy_from_slice(&g_new[..n]);
            fx = fx_trial;

            // BFGS update of Hessian B with Powell's damping for robustness
            bfgs_update_damped(&mut hess, &s[..n], &y[..n], &mut bs[..n], n);

            // B and g changed — invalidate cached Newton/Cauchy points.
            cache_valid = false;

            // Delta-based convergence test
            if params.past > 0 {
                let idx = (k + 1) % past_f.len().min(params.past + 1);
                past_f[idx] = fx;
                if k + 1 > params.past {
                    let old_idx = (k + 1 - params.past) % past_f.len().min(params.past + 1);
                    let rate = (past_f[old_idx] - fx).abs() / fx.abs().max(1.0);
                    if rate < params.delta_conv {
                        return (BfgsTrustResult::Stop, fx, k + 1);
                    }
                }
            }
        }
        // If rejected, delta was already shrunk; cache remains valid for next iteration.

        // Safety: if delta is too small, we've converged (or gotten stuck)
        if delta < 1e-7 {
            return (BfgsTrustResult::Convergence, fx, k + 1);
        }
    }

    (BfgsTrustResult::MaxIterations, fx, params.max_iterations)
}

/// Compute the dogleg step using pre-computed Newton and Cauchy points.
///
/// Avoids recomputing the Cholesky factorization and B*g product on rejected
/// steps where only delta changes.
fn dogleg_step_cached(
    b: &[f32],       // n×n Hessian (row-major)
    g: &[f32],       // gradient (length n)
    delta: f32,      // trust region radius
    p_newton: &[f32], // pre-computed Newton point (-B^{-1} g)
    newton_ok: bool,  // whether Cholesky succeeded
    pn_norm: f32,     // ||p_newton||
    p_cauchy: &[f32], // pre-computed Cauchy point
    pc_norm: f32,     // ||p_cauchy||
    gtg: f32,         // ||g||²
    p: &mut [f32],    // output step
    n: usize,
) -> f32 {
    // 1. Try Newton step if inside trust region
    if newton_ok && pn_norm <= delta {
        p[..n].copy_from_slice(&p_newton[..n]);
        return predicted_reduction(b, g, p, n);
    }

    // 2. Handle degenerate Cauchy (negative curvature or zero gradient)
    if pc_norm < 1e-7 {
        let gnorm = gtg.sqrt();
        if gnorm < 1e-7 {
            p[..n].fill(0.0);
            return 0.0;
        }
        let scale = -delta / gnorm;
        for i in 0..n { p[i] = scale * g[i]; }
        return predicted_reduction(b, g, p, n);
    }

    // 3. Cauchy outside trust region or Newton failed — cap Cauchy to delta
    if !newton_ok || pc_norm >= delta {
        let scale = delta / pc_norm;
        for i in 0..n { p[i] = scale * p_cauchy[i]; }
        return predicted_reduction(b, g, p, n);
    }

    // 4. Dogleg interpolation: find tau such that ||p_c + tau*(p_n - p_c)|| = delta
    let mut d = [0.0f32; MAX_VARS];
    for i in 0..n { d[i] = p_newton[i] - p_cauchy[i]; }
    let dd = vec_dot(&d[..n], &d[..n]);
    let pd = vec_dot(&p_cauchy[..n], &d[..n]);
    let pp_sq = pc_norm * pc_norm;

    let a = dd;
    let b_coeff = 2.0 * pd;
    let c = pp_sq - delta * delta;
    let disc = b_coeff * b_coeff - 4.0 * a * c;

    let tau = if disc >= 0.0 && a > 1e-7 {
        ((-b_coeff + disc.sqrt()) / (2.0 * a)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    for i in 0..n { p[i] = p_cauchy[i] + tau * d[i]; }
    predicted_reduction(b, g, p, n)
}

/// Predicted reduction of the quadratic model: pred = -(g^T p + 0.5 p^T B p)
fn predicted_reduction(b: &[f32], g: &[f32], p: &[f32], n: usize) -> f32 {
    let mut bp = [0.0f32; MAX_VARS];
    mat_vec(b, p, &mut bp[..n], n);
    let gp = vec_dot(&g[..n], &p[..n]);
    let pbp = vec_dot(&p[..n], &bp[..n]);
    -(gp + 0.5 * pbp)
}

/// Cholesky factorization and solve: B p = -g => p = -B^{-1} g.
/// Returns false if B is not positive definite.
fn cholesky_solve(b: &[f32], g: &[f32], p: &mut [f32], n: usize) -> bool {
    // Copy B to work buffer L (lower triangular will be stored here)
    let mut l = [0.0f32; MAX_VARS_SQ];
    l[..n * n].copy_from_slice(&b[..n * n]);

    // Cholesky factorization: L L^T = B
    for j in 0..n {
        let mut sum = l[j * n + j];
        for k in 0..j {
            sum -= l[j * n + k] * l[j * n + k];
        }
        if sum <= 0.0 {
            return false; // Not positive definite
        }
        // Regularize tiny diagonals to prevent ill-conditioned L factor
        if sum < 1e-6 {
            sum = 1e-6;
        }
        l[j * n + j] = sum.sqrt();
        let ljj = l[j * n + j];

        for i in (j + 1)..n {
            let mut sum = l[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = sum / ljj;
        }
    }

    // Solve L y = -g (forward substitution)
    let mut y = [0.0f32; MAX_VARS];
    for i in 0..n {
        let mut sum = -g[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }

    // Solve L^T p = y (backward substitution)
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * p[k];
        }
        p[i] = sum / l[i * n + i];
    }

    true
}

/// BFGS Hessian update with Powell's damping for non-convex robustness.
///
/// Standard BFGS: B' = B - (Bs)(Bs)^T/(s^T Bs) + yy^T/(y^T s)
/// Powell's damping: if y^T s < 0.2 * s^T Bs, use damped y.
fn bfgs_update_damped(b: &mut [f32], s: &[f32], y: &[f32], bs: &mut [f32], n: usize) {
    mat_vec(b, s, bs, n);
    let s_bs = vec_dot(&s[..n], &bs[..n]);
    let y_s = vec_dot(&y[..n], &s[..n]);

    if s_bs < 1e-7 {
        return; // Skip update if s^T B s is too small
    }

    // Powell's damping
    let mut y_damped = [0.0f32; MAX_VARS];
    let theta = if y_s < 0.2 * s_bs {
        let th = 0.8 * s_bs / (s_bs - y_s);
        for i in 0..n {
            y_damped[i] = th * y[i] + (1.0 - th) * bs[i];
        }
        th
    } else {
        y_damped[..n].copy_from_slice(&y[..n]);
        1.0
    };
    let _ = theta;

    let yd_s = vec_dot(&y_damped[..n], &s[..n]);
    if yd_s < 1e-7 {
        return; // Skip if damped y^T s is too small
    }

    // B' = B - (Bs)(Bs)^T / (s^T Bs) + (yd)(yd)^T / (yd^T s)
    let inv_sbs = 1.0 / s_bs;
    let inv_yds = 1.0 / yd_s;
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] += -bs[i] * bs[j] * inv_sbs + y_damped[i] * y_damped[j] * inv_yds;
        }
    }
}

/// Matrix-vector product: y = A * x (A is n×n row-major)
#[inline]
fn mat_vec(a: &[f32], x: &[f32], y: &mut [f32], n: usize) {
    for i in 0..n {
        let mut sum = 0.0;
        let row = i * n;
        for j in 0..n {
            sum += a[row + j] * x[j];
        }
        y[i] = sum;
    }
}

#[inline]
fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn vec_norm(v: &[f32]) -> f32 {
    vec_dot(v, v).sqrt()
}

#[inline]
fn vec_norm_inf(v: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for &x in v {
        m = m.max(x.abs());
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic() {
        let mut x = [3.0, -4.0];
        let mut eval = |xv: &[f32], g: &mut [f32]| -> f32 {
            g[0] = 2.0 * xv[0];
            g[1] = 2.0 * xv[1];
            xv[0] * xv[0] + xv[1] * xv[1]
        };
        let (status, cost, iters) = bfgs_trust_optimize(&mut x, &mut eval, &BfgsTrustParams::default());
        eprintln!("Quadratic: status={status:?}, x=[{:.6}, {:.6}], cost={cost:.2e}, iters={iters}", x[0], x[1]);
        assert!(x[0].abs() < 1e-6);
        assert!(x[1].abs() < 1e-6);
        assert!(cost < 1e-10);
    }

    #[test]
    fn test_rosenbrock() {
        let mut x = [-1.0, 1.0];
        let mut eval = |xv: &[f32], g: &mut [f32]| -> f32 {
            let a = 1.0 - xv[0];
            let b = xv[1] - xv[0] * xv[0];
            g[0] = -2.0 * a - 400.0 * xv[0] * b;
            g[1] = 200.0 * b;
            a * a + 100.0 * b * b
        };
        // Rosenbrock needs more iterations than the planner default (pathological test function)
        let params = BfgsTrustParams { max_iterations: 200, ..BfgsTrustParams::default() };
        let (status, cost, iters) = bfgs_trust_optimize(&mut x, &mut eval, &params);
        eprintln!("Rosenbrock: status={status:?}, x=[{:.6}, {:.6}], cost={cost:.2e}, iters={iters}", x[0], x[1]);
        assert!((x[0] - 1.0).abs() < 1e-3);
        assert!((x[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_ill_conditioned() {
        // f(x) = 0.5 * (x1^2 + 1000*x2^2), condition number = 1000
        let mut x = [10.0, 10.0];
        let mut eval = |xv: &[f32], g: &mut [f32]| -> f32 {
            g[0] = xv[0];
            g[1] = 1000.0 * xv[1];
            0.5 * (xv[0] * xv[0] + 1000.0 * xv[1] * xv[1])
        };
        let (status, cost, iters) = bfgs_trust_optimize(&mut x, &mut eval, &BfgsTrustParams::default());
        eprintln!("Ill-cond (κ=1000): status={status:?}, cost={cost:.2e}, iters={iters}");
        assert!(cost < 1e-10);
    }
}
