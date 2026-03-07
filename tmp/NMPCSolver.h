// NMPCSolver.h - NMPC optimization solver
// Migrated from FSC Lab's cyblib for UE5
// Copyright 2024 FSC Lab - MIT License

#pragma once

#include "NMPCTypes.h"
#include "LBFGSSolver.h"
#include <vector>
#include <iostream>

namespace NMPC
{

//=============================================================================
// NMPC Solver Template
//=============================================================================

template <typename Model>
class NMPCSolver
{
private:
    typedef typename Model::VXX VXX;
    typedef typename Model::VXU VXU;
    typedef typename Model::MXA MXA;
    typedef typename Model::MXB MXB;

    Model model_;
    VXX x_init_;
    Vector<> x_prev_;
    bool warmstart_ = false;
    LBFGSParams lbfgs_params_;
    Scalar residual_cost_;
    Scalar residual_const_;
    int last_iterations_{0};

public:
    NMPCSolver() = default;

    explicit NMPCSolver(const Model& model) : model_(model) {}

    void SetModel(const Model& model)
    {
        model_ = model;
    }

    Model& GetModel()
    {
        return model_;
    }

    int LastIterations() const { return last_iterations_; }

    Scalar Solve(const VXX& x_init,
                 const Scalar& relCostTol,
                 std::vector<VXX>& states,
                 std::vector<VXU>& controls)
    {
        x_init_ = x_init;

        // Check for trivial case
        constexpr Scalar kTrivialCaseTol = 1e-6;
        Scalar initial_cost = 0.0;
        for (size_t i = 0; i <= static_cast<size_t>(model_.N()); ++i)
        {
            initial_cost += (x_init - model_.x_refs_[i]).squaredNorm();
        }

        if (initial_cost < kTrivialCaseTol)
        {
            UE_LOG(LogTemp, Log, TEXT("NMPCSolver: Trivial case detected, initial state close to references."));
            states.resize(static_cast<size_t>(model_.N()) + 1);
            controls.resize(static_cast<size_t>(model_.N()));
            for (size_t i = 0; i <= static_cast<size_t>(model_.N()); ++i)
            {
                states[i] = x_init;
                if (i < static_cast<size_t>(model_.N()))
                {
                    controls[i] = model_.u_refs_[i];
                }
            }
            // warmstart_ = false;
            last_iterations_ = 0;
            return 0.0;
        }

        Scalar minObjectiveFunctional;
        lbfgs_params_.mem_size = 128;
        lbfgs_params_.past = 3;
        lbfgs_params_.g_epsilon = 0.0;
        lbfgs_params_.min_step = 1.0e-32;
        lbfgs_params_.delta = relCostTol;
        lbfgs_params_.max_iterations = 60;

        Vector<> x(static_cast<Eigen::Index>(model_.nu()) * model_.N());

        if (!warmstart_)
        {
            for (size_t i = 0; i < static_cast<size_t>(model_.N()); ++i)
            {
                x.segment(static_cast<Eigen::Index>(i * static_cast<size_t>(model_.nu())), model_.nu()) = model_.u_refs_[i];
            }
        }
        else
        {
            x = x_prev_;
            warmstart_ = false;
        }

        last_iterations_ = 0;

        int ret = LBFGSOptimize(x,
                                 minObjectiveFunctional,
                                 NMPCSolver::ObjectiveFunctional,
                                 nullptr,
                                 NMPCSolver::ProgressCallback,
                                 this,
                                 lbfgs_params_);

        if (ret >= 0)
        {
            Map<const Matrix<>> U(x.data(), static_cast<Eigen::Index>(model_.nu()), static_cast<Eigen::Index>(model_.N()));

            states.resize(static_cast<size_t>(model_.N()) + 1);
            controls.resize(static_cast<size_t>(model_.N()));
            states[0] = x_init_;

            for (size_t i = 0; i < static_cast<size_t>(model_.N()); ++i)
            {
                states[i + 1] = model_.PropagateRK4(states[i], U.col(static_cast<Eigen::Index>(i)));
                controls[i] = U.col(static_cast<Eigen::Index>(i));
            }

            x_prev_ = x;
            warmstart_ = true;
        }
        else
        {
            minObjectiveFunctional = INFINITY;
        }

        return minObjectiveFunctional;
    }

private:
    static constexpr Eigen::Index kMaxHorizon = 50;
    static constexpr Eigen::Index kNx = 10;
    static constexpr Eigen::Index kNu = 4;

    static inline Scalar ObjectiveFunctional(void* ptr, const Vector<>& x, Vector<>& g)
    {
        NMPCSolver& obj = *static_cast<NMPCSolver*>(ptr);
        const Eigen::Index N = static_cast<Eigen::Index>(obj.GetModel().N());

        Map<const Matrix<kNu, Eigen::Dynamic>> U(x.data(), kNu, N);
        Map<Matrix<kNu, Eigen::Dynamic>> grad_U(g.data(), kNu, N);

        Eigen::Matrix<Scalar, kNx, kMaxHorizon + 1> X;
        X.col(0) = obj.x_init_;
        for (Eigen::Index i = 0; i < N; ++i)
        {
            X.col(i + 1) = obj.GetModel().PropagateRK4(X.col(i), U.col(i));
        }

        VXX grad_xk = VXX::Zero();
        VXX grad_xk_1 = VXX::Zero();
        VXU grad_uk = VXU::Zero();

        VXX grad_xk_cost = VXX::Zero();
        VXX grad_hxk_const = VXX::Zero();
        VXU grad_uk_cost = VXU::Zero();
        VXU grad_huk_const = VXU::Zero();

        MXA grad_Fx;
        MXB grad_Fu;

        Scalar total_cost = 0.0;
        obj.residual_cost_ = 0.0;
        obj.residual_const_ = 0.0;

        obj.residual_cost_ += obj.GetModel().TerminalCostGrad(X.col(N),
                                                               obj.GetModel().x_refs_[static_cast<size_t>(N)],
                                                               grad_xk);

        grad_xk_1 = grad_xk;
        for (Eigen::Index i = N - 1; i >= 0; --i)
        {
            obj.residual_cost_ += obj.GetModel().PathCostGrad(X.col(i), U.col(i),
                                                               obj.GetModel().x_refs_[static_cast<size_t>(i)],
                                                               obj.GetModel().u_refs_[static_cast<size_t>(i)],
                                                               grad_xk_cost, grad_uk_cost);

            obj.residual_const_ += obj.GetModel().AddGeneralConstraintGrad(X.col(i), U.col(i),
                                                                            grad_hxk_const, grad_huk_const);

            grad_xk = grad_xk_cost + grad_hxk_const;
            grad_uk = grad_uk_cost + grad_huk_const;

            obj.GetModel().PropagateRK4Grad(X.col(i), U.col(i), grad_Fx, grad_Fu);

            grad_U.col(i).noalias() = grad_uk + grad_Fu.transpose() * grad_xk_1;
            grad_xk_1.noalias() = grad_Fx.transpose() * grad_xk_1 + grad_xk;
        }

        total_cost = obj.residual_cost_ + obj.residual_const_;
        return total_cost;
    }

    static inline int ProgressCallback(void* ptr,
                                        const Vector<>& /*x*/,
                                        const Vector<>& /*g*/,
                                        const Scalar /*fx*/,
                                        const Scalar /*step*/,
                                        const int k,
                                        const int /*ls*/)
    {
        NMPCSolver& obj = *static_cast<NMPCSolver*>(ptr);
        obj.last_iterations_ = k;
        return 0;
    }
};

} // namespace NMPC
