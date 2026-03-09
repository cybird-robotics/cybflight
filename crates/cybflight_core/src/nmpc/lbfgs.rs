// L-BFGS optimizer with Lewis-Overton line search.
// Ported from LBFGSSolver.h (FSC Lab / cyblib).
//
// Uses fixed-size arrays (no heap) for embedded no_std.
// DIM = optimization variable dimension (NU*N = 80).
// M   = L-BFGS memory size (reduced from 128 to 10 for embedded).

use core::default::Default;
use core::marker::Copy;
use core::ops::FnMut;
use core::result::{
    Result,
    Result::{Err, Ok},
};
use nalgebra as na;
use num_traits::NumCast;

pub const DIM: usize = 80; // must equal N * NU (10 * 4); see assert in solver.rs
pub const M: usize = 5;
pub const PAST: usize = 3;

// L-BFGS return codes (matches C++ enum)

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LbfgsError {
    NonFiniteValue,
    LinesearchNegativeStep,
    LinesearchPositiveGradient,
    LineSearchMaximumIteration,
    LinesearchStepTooLarge,
    LinesearchStepTooSmall,
    MaximumIteration,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LbfgsSuccessReason {
    Convergence,
    Stop,
}

// Parameters — matches NMPCSolver.h:lbfgs_params_ setup
pub struct LbfgsParams<T: NumCast> {
    pub max_iterations: i32,
    pub max_linesearch: i32,
    pub past: i32,
    pub delta: T,
    pub g_epsilon: T,
    pub min_step: T,
    pub max_step: T,
    pub f_dec_coeff: T,
    pub s_curv_coeff: T,
    pub cautious_factor: T,
    pub machine_prec: T,
}

impl<T: NumCast> Default for LbfgsParams<T> {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            max_linesearch: 20,
            past: 0,
            delta: T::from(0.0).unwrap(),
            g_epsilon: T::from(1e-4).unwrap(),
            min_step: T::from(1e-20).unwrap(),
            max_step: T::from(1e20).unwrap(),
            f_dec_coeff: T::from(1e-4).unwrap(),
            s_curv_coeff: T::from(0.9).unwrap(),
            cautious_factor: T::from(1e-6).unwrap(),
            machine_prec: T::from(1e-7).unwrap(), // T machine epsilon ≈ 1.19e-7
        }
    }
}

// Persistent L-BFGS working memory — store in NmpcSolver to avoid stack allocation.
pub struct LbfgsWorkspace<T: na::RealField, const DIM: usize> {
    pub lm_s: na::SMatrix<T, DIM, M>,
    pub lm_y: na::SMatrix<T, DIM, M>,
    pub lm_ys: na::SVector<T, M>,
    pub lm_alpha: na::SVector<T, M>,
    pub pf: na::SVector<T, PAST>,
    // scratch vectors
    pub xp: na::SVector<T, DIM>,
    pub g: na::SVector<T, DIM>,
    pub gp: na::SVector<T, DIM>,
    pub d: na::SVector<T, DIM>,
    // output
    pub last_k: i32,
}

impl<T: na::RealField + Copy, const DIM: usize> Default for LbfgsWorkspace<T, DIM> {
    fn default() -> Self {
        Self {
            lm_s: na::Matrix::zeros(),
            lm_y: na::Matrix::zeros(),
            lm_ys: na::Vector::zeros(),
            lm_alpha: na::Vector::zeros(),
            pf: na::Vector::zeros(),
            xp: na::Vector::zeros(),
            g: na::Vector::zeros(),
            gp: na::Vector::zeros(),
            d: na::Vector::zeros(),
            last_k: 0,
        }
    }
}

// ── vector helpers ────────────────────────────────────────────────────────────

// ── Lewis-Overton line search ─────────────────────────────────────────────────
pub enum LineSearchSuccessReason {
    ArmijoWolfeSatisfied,
    BracketTooSmall,
}

struct LineSearchOutput<T, const DIM: usize> {
    pub x: na::SVector<T, DIM>,
    pub f: T,
    pub g: na::SVector<T, DIM>,
    pub reason: LineSearchSuccessReason,
}

pub struct ValueAndGrad<T, const DIM: usize>(pub T, pub na::SVector<T, DIM>);

/// Lewis-Overton line search.
///
/// # Arguments
/// - `xp`: Current point
/// - `gp`: Gradient at current point
/// - `finit`: Function value at current point
/// - `step`: Initial step size
/// - `s`: Search direction
/// - `param`: Line search parameters
/// - `eval`: Function to evaluate f and g at a given x
fn line_search<F, T: na::RealField + Copy + NumCast, const DIM: usize>(
    xp: &na::SVector<T, DIM>,
    gp: &na::SVector<T, DIM>,
    finit: T,
    step: T,
    s: &na::SVector<T, DIM>,
    param: &LbfgsParams<T>,
    eval: &mut F,
) -> Result<LineSearchOutput<T, DIM>, LbfgsError>
where
    F: FnMut(&na::SVector<T, DIM>) -> ValueAndGrad<T, DIM>,
{
    let mut count = 0_i32;
    let mut brackt = false;
    let mut touched = false;
    let mut mu = T::zero();
    let mut nu = param.max_step;

    if step <= T::zero() {
        return Err(LbfgsError::LinesearchNegativeStep);
    }

    let dginit = gp.dot(s);
    if dginit >= T::zero() {
        return Err(LbfgsError::LinesearchPositiveGradient);
    }

    let dgtest = param.f_dec_coeff * dginit;
    let dstest = param.s_curv_coeff * dginit;

    let mut step = step;
    loop {
        let x = (*xp) + (*s) * step;
        let ValueAndGrad(f, g) = eval(&x);
        count += 1;

        if !f.is_finite() {
            return Err(LbfgsError::NonFiniteValue);
        }

        if f > finit + step * dgtest {
            nu = step;
            brackt = true;
        } else if g.dot(s) < dstest {
            mu = step;
        } else {
            return Ok(LineSearchOutput {
                x,
                f,
                g,
                reason: LineSearchSuccessReason::ArmijoWolfeSatisfied,
            });
        }

        if count >= param.max_linesearch {
            return Err(LbfgsError::LineSearchMaximumIteration);
        }

        if brackt && (nu - mu) < param.machine_prec * nu {
            return Ok(LineSearchOutput {
                x,
                f,
                g,
                reason: LineSearchSuccessReason::BracketTooSmall,
            });
        }

        step = if brackt {
            T::from(0.5).unwrap() * (mu + nu)
        } else {
            step * T::from(2.0).unwrap()
        };

        if step < param.min_step {
            return Err(LbfgsError::LinesearchStepTooSmall);
        }

        if step > param.max_step {
            if touched {
                return Err(LbfgsError::LinesearchStepTooLarge);
            }
            touched = true;
            step = param.max_step;
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Solution<T, const DIM: usize> {
    pub x: na::SVector<T, DIM>,
    pub f: T,
    pub reason: LbfgsSuccessReason,
}

// ── Main L-BFGS loop ──────────────────────────────────────────────────────────
//
// Matches LBFGSSolver.h:LBFGSOptimize exactly.
// `eval(x, g) -> f`: writes gradient into g, returns cost.
// `ws`: persistent workspace (avoids stack allocation).
pub fn lbfgs_optimize<F, T: na::RealField + Copy + NumCast, const DIM: usize>(
    x: &na::SVector<T, DIM>,
    param: &LbfgsParams<T>,
    ws: &mut LbfgsWorkspace<T, DIM>,
    eval: &mut F,
) -> Result<Solution<T, DIM>, LbfgsError>
where
    F: FnMut(&na::SVector<T, DIM>) -> ValueAndGrad<T, DIM>,
{
    let ValueAndGrad(v, g) = eval(x);
    let mut fx = v;
    ws.g = g;
    ws.pf[0] = fx;

    // Initial search direction: d = -g
    ws.d = -ws.g;

    let gnorm = ws.g.abs().max();
    let xnorm = x.abs().max().max(T::one());
    if gnorm / xnorm < param.g_epsilon {
        return Ok(Solution {
            x: *x,
            f: fx,
            reason: LbfgsSuccessReason::Convergence,
        });
    }

    let nd = ws.d.norm();
    let mut step = if nd > T::zero() {
        T::one() / nd
    } else {
        T::one()
    };
    let mut k = 1_i32;
    let mut end = 0_usize;
    let mut bound = 0_usize;

    // Reset L-BFGS memory
    ws.lm_s = na::Matrix::zeros();
    ws.lm_y = na::Matrix::zeros();
    ws.lm_ys = na::Vector::zeros();

    let mut x = *x;
    loop {
        ws.xp = x;
        ws.gp = ws.g;

        let res = line_search(&ws.xp, &ws.gp, fx, step, &ws.d, param, eval)?;
        x = res.x;
        fx = res.f;
        ws.g = res.g; // Update the workspace gradient
        let gnorm = ws.g.abs().max();
        let xnorm = x.abs().max().max(T::one());
        if gnorm / xnorm < param.g_epsilon {
            return Ok(Solution {
                x,
                f: fx,
                reason: LbfgsSuccessReason::Convergence,
            });
        }

        if param.past > 0 {
            let past = param.past as usize;
            let idx = ((k - 1) as usize) % past;
            if k as usize >= past {
                let prev = ws.pf[idx];
                let rate = (prev - fx).abs() / fx.abs().max(T::one());
                if rate < param.delta {
                    return Ok(Solution {
                        x,
                        f: fx,
                        reason: LbfgsSuccessReason::Stop,
                    });
                }
            }
            ws.pf[idx] = fx;
        }

        if param.max_iterations > 0 && k >= param.max_iterations {
            return Err(LbfgsError::MaximumIteration);
        }

        // Update L-BFGS curvature pairs
        // s = x - xp,  y = g - gp
        ws.lm_s.column_mut(end).copy_from(&(x - ws.xp));
        ws.lm_y.column_mut(end).copy_from(&(ws.g - ws.gp));
        let ys = ws.lm_y.column(end).dot(&ws.lm_s.column(end));
        let yy = ws.lm_y.column(end).dot(&ws.lm_y.column(end));
        ws.lm_ys[end] = ys;

        // d = -g  (reset direction)
        ws.d = -ws.g;

        // Cautious update: skip if curvature condition not met
        let s_sq: T = ws.lm_s.column(end).norm_squared();
        let gp_n = ws.gp.norm();
        let cau = s_sq * gp_n * param.cautious_factor;

        if ys > cau {
            bound = (bound + 1).min(M);
            end = (end + 1) % M;

            // Two-loop L-BFGS recursion
            let mut j = end;
            for _ in 0..bound {
                j = (j + M - 1) % M;
                ws.lm_alpha[j] = ws.lm_s.column(j).dot(&ws.d) / ws.lm_ys[j];
                let a = -ws.lm_alpha[j];
                let y = ws.lm_y.column(j);
                ws.d += y * a;
            }

            let scale = ys / yy;
            ws.d *= scale;

            for _ in 0..bound {
                let beta = ws.lm_y.column(j).dot(&ws.d) / ws.lm_ys[j];
                let diff = ws.lm_alpha[j] - beta;
                let s = ws.lm_s.column(j);
                ws.d += s * diff;
                j = (j + 1) % M;
            }
        }

        step = T::one();
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::{assert, matches};

    // 1. Verify Vector Primitives (The first targets for functional refactoring)

    // 2. Convergence Test: The Sphere Function
    // f(x) = sum(x_i^2), min at x = [0, 0, ... 0]
    #[test]
    fn test_rosenbrock_optimization() {
        let x = na::SVector::<f32, DIM>::from_element(-1.2); // Classic Rosenbrock starting point
        let mut params = LbfgsParams::default();

        // Rosenbrock needs more iterations than the default Sphere
        params.max_iterations = 100;
        params.g_epsilon = 1e-3_f32;

        let mut ws = LbfgsWorkspace::default();

        let mut eval = |x: &na::SVector<f32, DIM>| -> ValueAndGrad<f32, DIM> {
            let mut fx = 0.0;
            // Reset gradient
            let g = &mut na::SVector::<f32, DIM>::zeros();

            for i in 0..(DIM / 2) {
                let x_odd = x[2 * i];
                let x_even = x[2 * i + 1];

                let t1 = 1.0 - x_odd;
                let t2 = x_even - x_odd * x_odd;

                fx += t1 * t1 + 100.0 * t2 * t2;

                // Gradient components
                g[2 * i] = -2.0 * t1 - 400.0 * x_odd * t2;
                g[2 * i + 1] = 200.0 * t2;
            }
            ValueAndGrad(fx, *g)
        };

        let result = lbfgs_optimize(&x, &params, &mut ws, &mut eval);

        // Verify it didn't fail due to Line Search or NaNs
        assert!(
            matches!(result, Ok(_)),
            "Optimization failed with error: {:?}",
            result
        );
        let Solution {
            x,
            f: f_out,
            reason,
        } = result.unwrap();

        // The global minimum is at x = [1.0, 1.0, ... 1.0] where f(x) = 0
        // L-BFGS should get reasonably close even with T
        assert!(f_out < 0.1, "Final cost too high: {}", f_out);

        for val in x.iter() {
            assert!((val - 1.0).abs() < 0.5, "Coordinate far from 1.0: {}", val);
        }
        assert_eq!(
            reason,
            LbfgsSuccessReason::Convergence,
            "Did not converge: {:?}",
            reason,
        );
    }

    // 3. Robustness: Handle NaN from objective function
    // #[test]
    // fn test_nan_handling() {
    //     let mut x = [1.0; DIM];
    //     let mut f_out = 0.0;
    //     let params = LbfgsParams::default();
    //     let mut ws = LbfgsWorkspace::default();
    //
    //     let mut eval = |_x: &[T; DIM], _g: &mut [T; DIM]| -> T { T::NAN };
    //
    //     let result = lbfgs_optimize(&mut x, &mut f_out, &params, &mut ws, &mut eval);
    //     assert_eq!(result, LBFGSERR_INVALID_FUNCVAL);
    // }
}
