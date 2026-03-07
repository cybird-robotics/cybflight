// LBFGSSolver.h - L-BFGS optimization solver
// Migrated from FSC Lab's cyblib for UE5
// Copyright 2024 FSC Lab - MIT License

#pragma once

#include "NMPCTypes.h"

namespace NMPC
{

//=============================================================================
// L-BFGS Parameters
//=============================================================================

struct LBFGSParams
{
    int mem_size = 256;
    int past = 3;
    Scalar g_epsilon = 0.0;
    Scalar delta = 1.0e-5;
    int max_iterations = 0;
    int max_linesearch = 64;
    Scalar min_step = 1.0e-32;
    Scalar max_step = 1.0e+20;
    Scalar f_dec_coeff = 1.0e-4;
    Scalar s_curv_coeff = 0.9;
    Scalar cautious_factor = 1.0e-6;
    Scalar machine_prec = 1.0e-16;
};

//=============================================================================
// L-BFGS Return Codes
//=============================================================================

enum LBFGSReturnCode
{
    LBFGS_CONVERGENCE = 0,
    LBFGS_STOP,
    LBFGS_CANCELED,
    LBFGSERR_MAXIMUMITERATION,

    LBFGSERR_UNKNOWNERROR = -1024,
    LBFGSERR_INVALID_N,
    LBFGSERR_INVALID_MEMSIZE,
    LBFGSERR_INVALID_GEPSILON,
    LBFGSERR_INVALID_TESTPERIOD,
    LBFGSERR_INVALID_DELTA,
    LBFGSERR_INVALID_MINSTEP,
    LBFGSERR_INVALID_MAXSTEP,
    LBFGSERR_INVALID_FDECCOEFF,
    LBFGSERR_INVALID_SCURVCOEFF,
    LBFGSERR_INVALID_MACHINEPREC,
    LBFGSERR_INVALID_MAXLINESEARCH,
    LBFGSERR_INVALID_FUNCVAL,
    LBFGSERR_MINIMUMSTEP,
    LBFGSERR_MAXIMUMSTEP,
    LBFGSERR_MAXIMUMLINESEARCH,
    LBFGSERR_WIDTHTOOSMALL,
    LBFGSERR_INVALIDPARAMETERS,
    LBFGSERR_INCREASEGRADIENT,
};

//=============================================================================
// Callback Types
//=============================================================================

typedef Scalar (*lbfgs_evaluate_t)(void* instance, const Vector<>& x, Vector<>& g);

typedef Scalar (*lbfgs_stepbound_t)(void* instance, const Vector<>& xp, const Vector<>& d);

typedef int (*lbfgs_progress_t)(void* instance, const Vector<>& x, const Vector<>& g,
                                const Scalar fx, const Scalar step, const int k, const int ls);

struct CallbackData
{
    void* instance = nullptr;
    lbfgs_evaluate_t proc_evaluate = nullptr;
    lbfgs_stepbound_t proc_stepbound = nullptr;
    lbfgs_progress_t proc_progress = nullptr;
};

//=============================================================================
// Line Search (Lewis-Overton method)
//=============================================================================

inline int LineSearchLewisOverton(Vector<>& x, Scalar& f, Vector<>& g,
                                   Scalar& stp, const Vector<>& s,
                                   const Vector<>& xp, const Vector<>& gp,
                                   const Scalar stpmin, const Scalar stpmax,
                                   const CallbackData& cd,
                                   const LBFGSParams& param)
{
    int count = 0;
    bool brackt = false, touched = false;
    Scalar finit, dginit, dgtest, dstest;
    Scalar mu = 0.0, nu = stpmax;

    if (!(stp > 0.0))
    {
        return LBFGSERR_INVALIDPARAMETERS;
    }

    dginit = gp.dot(s);

    if (0.0 < dginit)
    {
        return LBFGSERR_INCREASEGRADIENT;
    }

    finit = f;
    dgtest = param.f_dec_coeff * dginit;
    dstest = param.s_curv_coeff * dginit;

    while (true)
    {
        x = xp + stp * s;

        f = cd.proc_evaluate(cd.instance, x, g);
        ++count;

        if (std::isinf(f) || std::isnan(f))
        {
            return LBFGSERR_INVALID_FUNCVAL;
        }

        if (f > finit + stp * dgtest)
        {
            nu = stp;
            brackt = true;
        }
        else
        {
            if (g.dot(s) < dstest)
            {
                mu = stp;
            }
            else
            {
                return count;
            }
        }

        if (param.max_linesearch <= count)
        {
            return LBFGSERR_MAXIMUMLINESEARCH;
        }

        if (brackt && (nu - mu) < param.machine_prec * nu)
        {
            return LBFGSERR_WIDTHTOOSMALL;
        }

        if (brackt)
        {
            stp = 0.5 * (mu + nu);
        }
        else
        {
            stp *= 2.0;
        }

        if (stp < stpmin)
        {
            return LBFGSERR_MINIMUMSTEP;
        }

        if (stp > stpmax)
        {
            if (touched)
            {
                return LBFGSERR_MAXIMUMSTEP;
            }
            else
            {
                touched = true;
                stp = stpmax;
            }
        }
    }
}

//=============================================================================
// L-BFGS Optimization
//=============================================================================

inline int LBFGSOptimize(Vector<>& x, Scalar& f,
                          lbfgs_evaluate_t proc_evaluate,
                          lbfgs_stepbound_t proc_stepbound,
                          lbfgs_progress_t proc_progress,
                          void* instance,
                          const LBFGSParams& param)
{
    int ret, i, j, k, ls, end, bound;
    Scalar step, step_min, step_max, fx, ys, yy;
    Scalar gnorm_inf, xnorm_inf, beta, rate, cau;

    const int n = static_cast<int>(x.size());
    const int m = param.mem_size;

    if (n <= 0) return LBFGSERR_INVALID_N;
    if (m <= 0) return LBFGSERR_INVALID_MEMSIZE;
    if (param.g_epsilon < 0.0) return LBFGSERR_INVALID_GEPSILON;
    if (param.past < 0) return LBFGSERR_INVALID_TESTPERIOD;
    if (param.delta < 0.0) return LBFGSERR_INVALID_DELTA;
    if (param.min_step < 0.0) return LBFGSERR_INVALID_MINSTEP;
    if (param.max_step < param.min_step) return LBFGSERR_INVALID_MAXSTEP;
    if (!(param.f_dec_coeff > 0.0 && param.f_dec_coeff < 1.0)) return LBFGSERR_INVALID_FDECCOEFF;
    if (!(param.s_curv_coeff < 1.0 && param.s_curv_coeff > param.f_dec_coeff)) return LBFGSERR_INVALID_SCURVCOEFF;
    if (!(param.machine_prec > 0.0)) return LBFGSERR_INVALID_MACHINEPREC;
    if (param.max_linesearch <= 0) return LBFGSERR_INVALID_MAXLINESEARCH;

    Vector<> xp(n);
    Vector<> g(n);
    Vector<> gp(n);
    Vector<> d(n);
    Vector<> pf(std::max(1, param.past));

    Vector<> lm_alpha = Vector<>::Zero(m);
    Matrix<> lm_s = Matrix<>::Zero(n, m);
    Matrix<> lm_y = Matrix<>::Zero(n, m);
    Vector<> lm_ys = Vector<>::Zero(m);

    CallbackData cd;
    cd.instance = instance;
    cd.proc_evaluate = proc_evaluate;
    cd.proc_stepbound = proc_stepbound;
    cd.proc_progress = proc_progress;

    fx = cd.proc_evaluate(cd.instance, x, g);
    pf(0) = fx;

    d = -g;

    gnorm_inf = g.cwiseAbs().maxCoeff();
    xnorm_inf = x.cwiseAbs().maxCoeff();

    if (gnorm_inf / std::max<Scalar>(1.0, xnorm_inf) < param.g_epsilon)
    {
        ret = LBFGS_CONVERGENCE;
    }
    else
    {
        step = 1.0 / d.norm();

        k = 1;
        end = 0;
        bound = 0;

        while (true)
        {
            xp = x;
            gp = g;

            step_min = param.min_step;
            step_max = param.max_step;
            if (cd.proc_stepbound)
            {
                step_max = cd.proc_stepbound(cd.instance, xp, d);
                step_max = step_max < param.max_step ? step_max : param.max_step;
                step = step < step_max ? step : 0.5 * step_max;
            }

            ls = LineSearchLewisOverton(x, fx, g, step, d, xp, gp, step_min, step_max, cd, param);

            if (ls < 0)
            {
                x = xp;
                g = gp;
                ret = ls;
                break;
            }

            if (cd.proc_progress)
            {
                if (cd.proc_progress(cd.instance, x, g, fx, step, k, ls))
                {
                    ret = LBFGS_CANCELED;
                    break;
                }
            }

            gnorm_inf = g.cwiseAbs().maxCoeff();
            xnorm_inf = x.cwiseAbs().maxCoeff();
            if (gnorm_inf / std::max<Scalar>(1.0, xnorm_inf) < param.g_epsilon)
            {
                ret = LBFGS_CONVERGENCE;
                break;
            }

            if (0 < param.past)
            {
                if (param.past <= k)
                {
                    rate = std::fabs(pf(k % param.past) - fx) /
                           std::max<Scalar>(1.0, std::fabs(fx));

                    if (rate < param.delta)
                    {
                        ret = LBFGS_STOP;
                        break;
                    }
                }
                pf(k % param.past) = fx;
            }

            if (param.max_iterations != 0 && param.max_iterations <= k)
            {
                ret = LBFGSERR_MAXIMUMITERATION;
                break;
            }

            ++k;

            lm_s.col(end) = x - xp;
            lm_y.col(end) = g - gp;

            ys = lm_y.col(end).dot(lm_s.col(end));
            yy = lm_y.col(end).squaredNorm();
            lm_ys(end) = ys;

            d = -g;

            cau = lm_s.col(end).squaredNorm() * gp.norm() * param.cautious_factor;

            if (ys > cau)
            {
                ++bound;
                bound = m < bound ? m : bound;
                end = (end + 1) % m;

                j = end;
                for (i = 0; i < bound; ++i)
                {
                    j = (j + m - 1) % m;
                    lm_alpha(j) = lm_s.col(j).dot(d) / lm_ys(j);
                    d += (-lm_alpha(j)) * lm_y.col(j);
                }

                d *= ys / yy;

                for (i = 0; i < bound; ++i)
                {
                    beta = lm_y.col(j).dot(d) / lm_ys(j);
                    d += (lm_alpha(j) - beta) * lm_s.col(j);
                    j = (j + 1) % m;
                }
            }

            step = 1.0;
        }
    }

    f = fx;
    return ret;
}

} // namespace NMPC
