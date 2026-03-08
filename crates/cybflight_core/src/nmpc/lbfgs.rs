// L-BFGS optimizer with Lewis-Overton line search.
// Ported from LBFGSSolver.h (FSC Lab / cyblib).
//
// Uses fixed-size arrays (no heap) for embedded no_std.
// DIM = optimization variable dimension (NU*N = 80).
// M   = L-BFGS memory size (reduced from 128 to 10 for embedded).

use nalgebra::ComplexField as _;

pub const DIM: usize = 80; // must equal N * NU (10 * 4); see assert in solver.rs
pub const M: usize = 5;
pub const PAST: usize = 3;

// L-BFGS return codes (matches C++ enum)
pub const LBFGS_CONVERGENCE: i32 = 0;
pub const LBFGS_STOP: i32 = 1;
pub const LBFGSERR_MAXIMUMITERATION: i32 = -1;
pub const LBFGSERR_LINESEARCH: i32 = -2;
pub const LBFGSERR_INVALID_FUNCVAL: i32 = -3;

// Parameters — matches NMPCSolver.h:lbfgs_params_ setup
pub struct LbfgsParams {
    pub max_iterations: i32,
    pub max_linesearch: i32,
    pub past: i32,
    pub delta: f32,
    pub g_epsilon: f32,
    pub min_step: f32,
    pub max_step: f32,
    pub f_dec_coeff: f32,
    pub s_curv_coeff: f32,
    pub cautious_factor: f32,
    pub machine_prec: f32,
}

impl LbfgsParams {
    pub fn default_nmpc() -> Self {
        Self {
            max_iterations: 5,
            max_linesearch: 20,
            past: 0,
            delta: 0.0,
            g_epsilon: 1e-4,
            min_step: 1e-20,
            max_step: 1e20,
            f_dec_coeff: 1e-4,
            s_curv_coeff: 0.9,
            cautious_factor: 1e-6,
            machine_prec: 1e-7, // f32 machine epsilon ≈ 1.19e-7
        }
    }
}

// Persistent L-BFGS working memory — store in NmpcSolver to avoid stack allocation.
pub struct LbfgsWorkspace {
    pub lm_s: [[f32; DIM]; M],
    pub lm_y: [[f32; DIM]; M],
    pub lm_ys: [f32; M],
    pub lm_alpha: [f32; M],
    pub pf: [f32; PAST],
    // scratch vectors
    pub xp: [f32; DIM],
    pub g: [f32; DIM],
    pub gp: [f32; DIM],
    pub d: [f32; DIM],
    // output
    pub last_k: i32,
}

impl LbfgsWorkspace {
    pub fn zeroed() -> Self {
        Self {
            lm_s: [[0.0; DIM]; M],
            lm_y: [[0.0; DIM]; M],
            lm_ys: [0.0; M],
            lm_alpha: [0.0; M],
            pf: [0.0; PAST],
            xp: [0.0; DIM],
            g: [0.0; DIM],
            gp: [0.0; DIM],
            d: [0.0; DIM],
            last_k: 0,
        }
    }
}

// ── vector helpers ────────────────────────────────────────────────────────────

#[inline(always)]
fn dot(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    let mut s = 0.0;
    for i in 0..DIM {
        s += a[i] * b[i];
    }
    s
}

#[inline(always)]
fn norm(a: &[f32; DIM]) -> f32 {
    dot(a, a).sqrt()
}

#[inline(always)]
fn max_abs(a: &[f32; DIM]) -> f32 {
    let mut m = 0.0_f32;
    for i in 0..DIM {
        let v = a[i].abs();
        if v > m {
            m = v;
        }
    }
    m
}

// a += alpha * b
#[inline(always)]
fn axpy(alpha: f32, b: &[f32; DIM], a: &mut [f32; DIM]) {
    for i in 0..DIM {
        a[i] += alpha * b[i];
    }
}

// out = a + alpha * b
#[inline(always)]
fn xpay(a: &[f32; DIM], alpha: f32, b: &[f32; DIM], out: &mut [f32; DIM]) {
    for i in 0..DIM {
        out[i] = a[i] + alpha * b[i];
    }
}

// ── Lewis-Overton line search ─────────────────────────────────────────────────
//
// Matches LBFGSSolver.h:LineSearchLewisOverton exactly.

fn line_search<F>(
    x: &mut [f32; DIM],
    f: &mut f32,
    g: &mut [f32; DIM],
    stp: &mut f32,
    s: &[f32; DIM],
    xp: &[f32; DIM],
    gp: &[f32; DIM],
    param: &LbfgsParams,
    eval: &mut F,
) -> i32
where
    F: FnMut(&[f32; DIM], &mut [f32; DIM]) -> f32,
{
    let mut count = 0_i32;
    let mut brackt = false;
    let mut touched = false;
    let mut mu = 0.0_f32;
    let mut nu = param.max_step;

    if !(*stp > 0.0) {
        return LBFGSERR_LINESEARCH;
    }

    let dginit = dot(gp, s);
    if dginit >= 0.0 {
        return LBFGSERR_LINESEARCH;
    }

    let finit = *f;
    let dgtest = param.f_dec_coeff * dginit;
    let dstest = param.s_curv_coeff * dginit;

    loop {
        xpay(xp, *stp, s, x);
        *f = eval(x, g);
        count += 1;

        if f.is_nan() || f.is_infinite() {
            return LBFGSERR_INVALID_FUNCVAL;
        }

        if *f > finit + *stp * dgtest {
            nu = *stp;
            brackt = true;
        } else {
            if dot(g, s) < dstest {
                mu = *stp;
            } else {
                return count;
            }
        }

        if count >= param.max_linesearch {
            return LBFGSERR_LINESEARCH;
        }

        if brackt && (nu - mu) < param.machine_prec * nu {
            return LBFGSERR_LINESEARCH;
        }

        *stp = if brackt { 0.5 * (mu + nu) } else { *stp * 2.0 };

        if *stp < param.min_step {
            return LBFGSERR_LINESEARCH;
        }
        if *stp > param.max_step {
            if touched {
                return LBFGSERR_LINESEARCH;
            }
            touched = true;
            *stp = param.max_step;
        }
    }
}

// ── Main L-BFGS loop ──────────────────────────────────────────────────────────
//
// Matches LBFGSSolver.h:LBFGSOptimize exactly.
// `eval(x, g) -> f`: writes gradient into g, returns cost.
// `ws`: persistent workspace (avoids stack allocation).

pub fn lbfgs_optimize<F>(
    x: &mut [f32; DIM],
    f_out: &mut f32,
    param: &LbfgsParams,
    ws: &mut LbfgsWorkspace,
    eval: &mut F,
) -> i32
where
    F: FnMut(&[f32; DIM], &mut [f32; DIM]) -> f32,
{
    let mut fx = eval(x, &mut ws.g);
    ws.pf[0] = fx;

    // Initial search direction: d = -g
    for i in 0..DIM {
        ws.d[i] = -ws.g[i];
    }

    let gnorm = max_abs(&ws.g);
    let xnorm = max_abs(x).max(1.0);
    if gnorm / xnorm < param.g_epsilon {
        *f_out = fx;
        return LBFGS_CONVERGENCE;
    }

    let nd = norm(&ws.d);
    let mut step = if nd > 0.0 { 1.0 / nd } else { 1.0 };
    let mut k = 1_i32;
    let mut end = 0_usize;
    let mut bound = 0_usize;

    // Reset L-BFGS memory
    for m in 0..M {
        ws.lm_s[m] = [0.0; DIM];
        ws.lm_y[m] = [0.0; DIM];
        ws.lm_ys[m] = 0.0;
    }

    let ret = loop {
        ws.xp.copy_from_slice(x);
        ws.gp.copy_from_slice(&ws.g);

        let ls = line_search(
            x,
            &mut fx,
            &mut ws.g,
            &mut step,
            &ws.d.clone(),
            &ws.xp.clone(),
            &ws.gp.clone(),
            param,
            eval,
        );
        if ls < 0 {
            x.copy_from_slice(&ws.xp);
            ws.g.copy_from_slice(&ws.gp);
            break ls;
        }

        let gnorm = max_abs(&ws.g);
        let xnorm = max_abs(x).max(1.0);
        if gnorm / xnorm < param.g_epsilon {
            break LBFGS_CONVERGENCE;
        }

        if param.past > 0 {
            let past = param.past as usize;
            let idx = ((k - 1) as usize) % past;
            if k as usize >= past {
                let prev = ws.pf[idx];
                let rate = (prev - fx).abs() / fx.abs().max(1.0);
                if rate < param.delta {
                    break LBFGS_STOP;
                }
            }
            ws.pf[idx] = fx;
        }

        if param.max_iterations > 0 && k >= param.max_iterations {
            break LBFGSERR_MAXIMUMITERATION;
        }

        // Update L-BFGS curvature pairs
        // s = x - xp,  y = g - gp
        let mut ys = 0.0_f32;
        let mut yy = 0.0_f32;
        for i in 0..DIM {
            ws.lm_s[end][i] = x[i] - ws.xp[i];
            ws.lm_y[end][i] = ws.g[i] - ws.gp[i];
        }
        for i in 0..DIM {
            ys += ws.lm_y[end][i] * ws.lm_s[end][i];
        }
        for i in 0..DIM {
            yy += ws.lm_y[end][i] * ws.lm_y[end][i];
        }
        ws.lm_ys[end] = ys;

        // d = -g  (reset direction)
        for i in 0..DIM {
            ws.d[i] = -ws.g[i];
        }

        // Cautious update: skip if curvature condition not met
        let s_sq: f32 = ws.lm_s[end].iter().map(|v| v * v).sum();
        let gp_n = norm(&ws.gp);
        let cau = s_sq * gp_n * param.cautious_factor;

        if ys > cau {
            bound = (bound + 1).min(M);
            end = (end + 1) % M;

            // Two-loop L-BFGS recursion
            let mut j = end;
            for _ in 0..bound {
                j = (j + M - 1) % M;
                ws.lm_alpha[j] = dot_slices(&ws.lm_s[j], &ws.d) / ws.lm_ys[j];
                let a = -ws.lm_alpha[j];
                let y = ws.lm_y[j];
                axpy(a, &y, &mut ws.d);
            }

            let scale = ys / yy;
            for i in 0..DIM {
                ws.d[i] *= scale;
            }

            for _ in 0..bound {
                let beta = dot_slices(&ws.lm_y[j], &ws.d) / ws.lm_ys[j];
                let diff = ws.lm_alpha[j] - beta;
                let s = ws.lm_s[j];
                axpy(diff, &s, &mut ws.d);
                j = (j + 1) % M;
            }
        }

        step = 1.0;
        k += 1;
    };

    ws.last_k = k;
    *f_out = fx;
    ret
}

// Helper: dot product of two fixed-size slices (used where ownership makes array refs awkward)
#[inline(always)]
fn dot_slices(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    dot(a, b)
}
