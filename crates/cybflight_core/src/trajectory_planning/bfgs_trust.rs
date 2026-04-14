//! Full BFGS optimizer with trust-region globalization (Cauchy-dogleg).
//!
//! Zero heap allocations — all scratch memory lives in [`BfgsWorkspace`]
//! (caller-owned) rather than on the `bfgs_trust_optimize` stack frame.
//! This keeps the single-function-frame size tiny, preventing stack
//! overflow on embedded targets (e.g. STM32H743 task stacks ≈ 4-8 KB).
//!
//! The workspace itself is ~35 KB for MAX_VARS = 64; place it on the
//! main stack, in a `static`, or in DTCM as the deployment requires.

#[allow(unused_imports)]
use num_traits::Float;

pub use crate::params::BfgsTrustParams;

const MAX_VARS: usize = 4 * super::MAX_PIECES; // 64
const MAX_VARS_SQ: usize = MAX_VARS * MAX_VARS; // 4096
const MAX_PAST: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BfgsTrustResult {
    Convergence,
    Stop,
    MaxIterations,
    InvalidValue,
}

/// All mutable working memory for a BFGS trust-region solve.
///
/// Construct once and reuse across calls — nothing here is problem-specific.
/// Sized for the worst-case `MAX_VARS` so the struct has a fixed footprint.
pub struct BfgsWorkspace {
    /// Hessian approximation B (n×n row-major).
    hess: [f32; MAX_VARS_SQ],
    /// Cholesky factor L scratch (n×n row-major).
    l: [f32; MAX_VARS_SQ],
    /// Current gradient.
    g: [f32; MAX_VARS],
    /// Trial-point gradient.
    g_new: [f32; MAX_VARS],
    /// Step direction.
    p: [f32; MAX_VARS],
    /// Trial point.
    x_trial: [f32; MAX_VARS],
    /// s = x_{k+1} − x_k (accepted step).
    s: [f32; MAX_VARS],
    /// y = g_{k+1} − g_k.
    y: [f32; MAX_VARS],
    /// B · s (BFGS update scratch).
    bs: [f32; MAX_VARS],
    /// B · g (Cauchy / dogleg scratch).
    bg: [f32; MAX_VARS],
    /// Cached Newton point −B⁻¹g.
    p_newton: [f32; MAX_VARS],
    /// Cached Cauchy point.
    p_cauchy: [f32; MAX_VARS],
    /// Powell-damped y (BFGS update scratch).
    y_damped: [f32; MAX_VARS],
    /// Dogleg interpolation direction (p_n − p_c).
    d_dogleg: [f32; MAX_VARS],
    /// B · p scratch (predicted-reduction).
    bp: [f32; MAX_VARS],
    /// Cholesky forward-substitution intermediate.
    chol_y: [f32; MAX_VARS],
    /// Past cost-value ring buffer for the "cost stagnation" convergence test.
    past_f: [f32; MAX_PAST],
}

impl BfgsWorkspace {
    pub const fn new() -> Self {
        Self {
            hess: [0.0; MAX_VARS_SQ],
            l: [0.0; MAX_VARS_SQ],
            g: [0.0; MAX_VARS],
            g_new: [0.0; MAX_VARS],
            p: [0.0; MAX_VARS],
            x_trial: [0.0; MAX_VARS],
            s: [0.0; MAX_VARS],
            y: [0.0; MAX_VARS],
            bs: [0.0; MAX_VARS],
            bg: [0.0; MAX_VARS],
            p_newton: [0.0; MAX_VARS],
            p_cauchy: [0.0; MAX_VARS],
            y_damped: [0.0; MAX_VARS],
            d_dogleg: [0.0; MAX_VARS],
            bp: [0.0; MAX_VARS],
            chol_y: [0.0; MAX_VARS],
            past_f: [0.0; MAX_PAST],
        }
    }
}

impl Default for BfgsWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Run full BFGS with trust-region optimization.
///
/// - `x`: initial guess, overwritten with solution
/// - `cost_grad`: closure `|x, grad| -> cost` computing cost and filling gradient
/// - `params`: optimizer parameters
/// - `ws`: preallocated working memory (see [`BfgsWorkspace`])
///
/// Returns `(result, final_cost, iterations)`.
pub fn bfgs_trust_optimize<F>(
    x: &mut [f32],
    cost_grad: &mut F,
    params: &BfgsTrustParams,
    ws: &mut BfgsWorkspace,
) -> (BfgsTrustResult, f32, usize)
where
    F: FnMut(&[f32], &mut [f32]) -> f32,
{
    let n = x.len();
    debug_assert!(n <= MAX_VARS);

    // Initialize B to identity (reuse ws.hess; zero the working portion first).
    for i in 0..(n * n) {
        ws.hess[i] = 0.0;
    }
    for i in 0..n {
        ws.hess[i * n + i] = 1.0;
    }

    // Cached dogleg quantities — valid until B or g changes.
    let mut newton_ok = false;
    let mut pn_norm = 0.0f32;
    let mut pc_norm = 0.0f32;
    let mut gtg = 0.0f32;
    let mut cache_valid = false;

    // Past-f ring buffer (bounded by MAX_PAST regardless of user `past` config).
    let past_len = (params.past + 1).min(MAX_PAST);
    for v in &mut ws.past_f[..past_len] {
        *v = 0.0;
    }
    // `accepted` counts accepted steps; the ring index should advance on
    // accepts only, so the "cost from `past` iters ago" is physically correct.
    let mut accepted: usize = 0;

    let mut delta = params.delta_init;
    let mut fx = cost_grad(x, &mut ws.g[..n]);
    if !fx.is_finite() {
        return (BfgsTrustResult::InvalidValue, fx, 0);
    }

    if params.past > 0 {
        ws.past_f[0] = fx;
    }

    let max_iter = if params.max_iterations > 0 {
        params.max_iterations
    } else {
        usize::MAX
    };

    let mut k: usize = 0;
    while k < max_iter {
        // Convergence: ||g||_inf / max(1, ||x||_inf) < g_epsilon
        let xnorm = vec_norm_inf(&x[..n]).max(1.0);
        let gnorm = vec_norm_inf(&ws.g[..n]);
        if gnorm / xnorm < params.g_epsilon {
            return (BfgsTrustResult::Convergence, fx, k);
        }

        if !cache_valid {
            // Newton point via Cholesky solve.
            newton_ok = cholesky_solve(
                &ws.hess,
                &ws.g[..n],
                &mut ws.p_newton[..n],
                &mut ws.l,
                &mut ws.chol_y,
                n,
            );
            pn_norm = if newton_ok {
                vec_norm(&ws.p_newton[..n])
            } else {
                0.0
            };

            // Cauchy point: p_c = -(||g||² / g^T B g) g
            mat_vec(&ws.hess, &ws.g[..n], &mut ws.bg[..n], n);
            gtg = vec_dot(&ws.g[..n], &ws.g[..n]);
            let g_bg = vec_dot(&ws.g[..n], &ws.bg[..n]);

            if g_bg > 0.0 && gtg >= 1e-7 {
                let alpha_c = gtg / g_bg;
                for i in 0..n {
                    ws.p_cauchy[i] = -alpha_c * ws.g[i];
                }
                pc_norm = vec_norm(&ws.p_cauchy[..n]);
            } else {
                pc_norm = 0.0;
            }
            cache_valid = true;
        }

        // Dogleg step (uses cached Newton & Cauchy).
        let pred = dogleg_step_cached(
            &ws.hess,
            &ws.g[..n],
            &ws.bg[..n],
            delta,
            &ws.p_newton[..n],
            newton_ok,
            pn_norm,
            &ws.p_cauchy[..n],
            pc_norm,
            gtg,
            &mut ws.p[..n],
            &mut ws.d_dogleg[..n],
            &mut ws.bp[..n],
            n,
        );

        if pred.abs() < 1e-7 {
            return (BfgsTrustResult::Convergence, fx, k);
        }

        // Evaluate trial point.
        for i in 0..n {
            ws.x_trial[i] = x[i] + ws.p[i];
        }
        let fx_trial = cost_grad(&ws.x_trial[..n], &mut ws.g_new[..n]);

        let actual = fx - fx_trial;
        let rho = if pred.abs() > 1e-7 { actual / pred } else { 0.0 };

        // Update trust radius.
        if rho < 0.25 {
            delta *= 0.25;
        } else if rho > 0.75 {
            let pnorm = vec_norm(&ws.p[..n]);
            if pnorm > 0.99 * delta {
                delta = (2.0 * delta).min(params.delta_max);
            }
        }

        // Accept or reject.
        if rho > params.eta && fx_trial.is_finite() {
            // s, y for BFGS update.
            for i in 0..n {
                ws.s[i] = ws.p[i];
                ws.y[i] = ws.g_new[i] - ws.g[i];
            }

            x[..n].copy_from_slice(&ws.x_trial[..n]);
            ws.g[..n].copy_from_slice(&ws.g_new[..n]);
            fx = fx_trial;

            bfgs_update_damped(
                &mut ws.hess,
                &ws.s[..n],
                &ws.y[..n],
                &mut ws.bs[..n],
                &mut ws.y_damped,
                n,
            );

            cache_valid = false;
            accepted += 1;

            // Cost-stagnation convergence test using a ring of past accepted
            // cost values. Index advances only on accepts → `past` counts
            // accepted iterations regardless of how many trials were rejected.
            if params.past > 0 && past_len > 0 {
                let idx = accepted % past_len;
                ws.past_f[idx] = fx;
                if accepted > params.past {
                    let old_idx = (accepted - params.past) % past_len;
                    let rate = (ws.past_f[old_idx] - fx).abs() / fx.abs().max(1.0);
                    if rate < params.delta_conv {
                        return (BfgsTrustResult::Stop, fx, k + 1);
                    }
                }
            }
        }

        if delta < 1e-7 {
            return (BfgsTrustResult::Convergence, fx, k + 1);
        }

        k += 1;
    }

    (BfgsTrustResult::MaxIterations, fx, k)
}

/// Dogleg step from pre-computed Newton & Cauchy points.
#[allow(clippy::too_many_arguments)]
fn dogleg_step_cached(
    b: &[f32],
    g: &[f32],
    bg: &[f32],
    delta: f32,
    p_newton: &[f32],
    newton_ok: bool,
    pn_norm: f32,
    p_cauchy: &[f32],
    pc_norm: f32,
    gtg: f32,
    p: &mut [f32],
    d_scratch: &mut [f32],
    bp_scratch: &mut [f32],
    n: usize,
) -> f32 {
    // 1. Newton step lies inside trust region: analytic predicted reduction.
    //
    //    B·p_newton = −g exactly (by Newton-point definition), so
    //    pred = −(g·p + ½ p·B·p) = −(g·p − ½ g·p) = −½ g·p
    if newton_ok && pn_norm <= delta {
        p[..n].copy_from_slice(&p_newton[..n]);
        return -0.5 * vec_dot(&g[..n], &p[..n]);
    }

    // 2. Degenerate Cauchy: take a steepest-descent step to the trust radius.
    if pc_norm < 1e-7 {
        let gnorm = gtg.sqrt();
        if gnorm < 1e-7 {
            p[..n].fill(0.0);
            return 0.0;
        }
        let scale = -delta / gnorm;
        for i in 0..n {
            p[i] = scale * g[i];
        }
        return predicted_reduction(b, g, p, bp_scratch, n);
    }

    // 3. Cauchy already past the trust boundary (or Newton failed): scale it in.
    //
    //    p = σ · p_cauchy = (σ · −α_c) g. Since bg = B·g is cached,
    //    B·p = (σ · −α_c) · bg, avoiding a fresh mat_vec.
    if !newton_ok || pc_norm >= delta {
        let scale = delta / pc_norm;
        // p_cauchy = −α_c · g, so p = scale · p_cauchy = (scale · −α_c) · g.
        let coeff = -scale * (vec_dot(&g[..n], &p_cauchy[..n]) / gtg.max(1e-20));
        // Equivalent direct form:
        for i in 0..n {
            p[i] = scale * p_cauchy[i];
        }
        // pred = −(g·p + ½ p·B·p); B·p = coeff · bg.
        let gp = vec_dot(&g[..n], &p[..n]);
        let mut pbp = 0.0;
        for i in 0..n {
            pbp += p[i] * (coeff * bg[i]);
        }
        return -(gp + 0.5 * pbp);
    }

    // 4. Dogleg: find τ so that ‖p_c + τ·d‖ = delta, d = p_n − p_c.
    for i in 0..n {
        d_scratch[i] = p_newton[i] - p_cauchy[i];
    }
    let dd = vec_dot(&d_scratch[..n], &d_scratch[..n]);
    let pd = vec_dot(&p_cauchy[..n], &d_scratch[..n]);
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

    for i in 0..n {
        p[i] = p_cauchy[i] + tau * d_scratch[i];
    }
    predicted_reduction(b, g, p, bp_scratch, n)
}

/// Predicted reduction: −(g·p + ½ p·B·p). Writes `bp = B·p` into `bp_scratch`.
#[inline]
fn predicted_reduction(b: &[f32], g: &[f32], p: &[f32], bp_scratch: &mut [f32], n: usize) -> f32 {
    mat_vec(b, p, &mut bp_scratch[..n], n);
    let gp = vec_dot(&g[..n], &p[..n]);
    let pbp = vec_dot(&p[..n], &bp_scratch[..n]);
    -(gp + 0.5 * pbp)
}

/// Cholesky factorization and solve: B p = −g ⇒ p = −B⁻¹ g.
///
/// `l_scratch` and `y_scratch` are caller-provided working buffers.
/// Returns false if B is not positive definite.
fn cholesky_solve(
    b: &[f32],
    g: &[f32],
    p: &mut [f32],
    l_scratch: &mut [f32],
    y_scratch: &mut [f32],
    n: usize,
) -> bool {
    // Copy B into L (lower triangular will be built in place).
    l_scratch[..n * n].copy_from_slice(&b[..n * n]);

    // Cholesky factorization: L Lᵀ = B.
    for j in 0..n {
        let mut sum = l_scratch[j * n + j];
        for k in 0..j {
            let lj = l_scratch[j * n + k];
            sum -= lj * lj;
        }
        if sum <= 0.0 {
            return false;
        }
        if sum < 1e-6 {
            sum = 1e-6;
        }
        let ljj = sum.sqrt();
        l_scratch[j * n + j] = ljj;
        let inv_ljj = 1.0 / ljj;

        for i in (j + 1)..n {
            let mut sum = l_scratch[i * n + j];
            for k in 0..j {
                sum -= l_scratch[i * n + k] * l_scratch[j * n + k];
            }
            l_scratch[i * n + j] = sum * inv_ljj;
        }
    }

    // Forward-substitute: L y = −g.
    for i in 0..n {
        let mut sum = -g[i];
        for k in 0..i {
            sum -= l_scratch[i * n + k] * y_scratch[k];
        }
        y_scratch[i] = sum / l_scratch[i * n + i];
    }

    // Back-substitute: Lᵀ p = y.
    for i in (0..n).rev() {
        let mut sum = y_scratch[i];
        for k in (i + 1)..n {
            sum -= l_scratch[k * n + i] * p[k];
        }
        p[i] = sum / l_scratch[i * n + i];
    }

    true
}

/// BFGS Hessian update with Powell's damping for non-convex robustness.
fn bfgs_update_damped(
    b: &mut [f32],
    s: &[f32],
    y: &[f32],
    bs: &mut [f32],
    y_damped: &mut [f32],
    n: usize,
) {
    mat_vec(b, s, bs, n);
    let s_bs = vec_dot(&s[..n], &bs[..n]);
    let y_s = vec_dot(&y[..n], &s[..n]);

    if s_bs < 1e-7 {
        return;
    }

    if y_s < 0.2 * s_bs {
        let th = 0.8 * s_bs / (s_bs - y_s);
        for i in 0..n {
            y_damped[i] = th * y[i] + (1.0 - th) * bs[i];
        }
    } else {
        y_damped[..n].copy_from_slice(&y[..n]);
    }

    let yd_s = vec_dot(&y_damped[..n], &s[..n]);
    if yd_s < 1e-7 {
        return;
    }

    let inv_sbs = 1.0 / s_bs;
    let inv_yds = 1.0 / yd_s;
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] += -bs[i] * bs[j] * inv_sbs + y_damped[i] * y_damped[j] * inv_yds;
        }
    }
}

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
        let mut ws = BfgsWorkspace::new();
        let (status, cost, iters) =
            bfgs_trust_optimize(&mut x, &mut eval, &BfgsTrustParams::default(), &mut ws);
        eprintln!(
            "Quadratic: status={status:?}, x=[{:.6}, {:.6}], cost={cost:.2e}, iters={iters}",
            x[0], x[1]
        );
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
        let params = BfgsTrustParams {
            max_iterations: 200,
            ..BfgsTrustParams::default()
        };
        let mut ws = BfgsWorkspace::new();
        let (status, cost, iters) = bfgs_trust_optimize(&mut x, &mut eval, &params, &mut ws);
        eprintln!(
            "Rosenbrock: status={status:?}, x=[{:.6}, {:.6}], cost={cost:.2e}, iters={iters}",
            x[0], x[1]
        );
        assert!((x[0] - 1.0).abs() < 1e-3);
        assert!((x[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_ill_conditioned() {
        let mut x = [10.0, 10.0];
        let mut eval = |xv: &[f32], g: &mut [f32]| -> f32 {
            g[0] = xv[0];
            g[1] = 1000.0 * xv[1];
            0.5 * (xv[0] * xv[0] + 1000.0 * xv[1] * xv[1])
        };
        let mut ws = BfgsWorkspace::new();
        let (status, cost, iters) =
            bfgs_trust_optimize(&mut x, &mut eval, &BfgsTrustParams::default(), &mut ws);
        eprintln!("Ill-cond (κ=1000): status={status:?}, cost={cost:.2e}, iters={iters}");
        assert!(cost < 1e-10);
    }
}
