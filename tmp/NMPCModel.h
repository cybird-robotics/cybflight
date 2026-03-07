// NMPCModel.h - Quadrotor dynamics model for NMPC
// Migrated from FSC Lab's cyblib for UE5
// Copyright 2024 FSC Lab - MIT License

#pragma once

#include "NMPCTypes.h"
#include <vector>

namespace NMPC
{

//=============================================================================
// NMPC Quadrotor Dynamics Model
//=============================================================================

class NMPCModel
{
private:
    static const size_t nx_ = 10;  // State dimension: [pos(3), quat(4), vel(3)]
    static const size_t nu_ = 4;   // Control dimension: [thrust, omega_x, omega_y, omega_z]

public:
    typedef Matrix<nx_, 1> VXX;
    typedef Matrix<nu_, 1> VXU;
    typedef Matrix<nx_, nx_> MXA;
    typedef Matrix<nx_, nu_> MXB;

    // RK4 intermediate variables (preallocated for performance)
    VXX k0, k1, k2, k3;
    MXA df_x0, df_x1, df_x2, df_x3;
    MXB df_u0, df_u1, df_u2, df_u3;
    MXA grad_k0_xk, grad_k1_xk, grad_k2_xk, grad_k3_xk;
    MXB grad_k0_uk, grad_k1_uk, grad_k2_uk, grad_k3_uk;

    // Cost weights
    Vector3 weightPos_;
    Vector3 weightAtt_;
    Vector3 weightVel_;
    VXU weightU_;

    // State and control references
    std::vector<VXX> x_refs_;
    std::vector<VXU> u_refs_;

    // State constraints
    std::vector<int> x_idx_;
    std::vector<Scalar> x_lb_;
    std::vector<Scalar> x_ub_;

    // Control constraints
    std::vector<int> u_idx_;
    std::vector<Scalar> u_lb_;
    std::vector<Scalar> u_ub_;

    // Model parameters
    int N_{20};
    Scalar dt_{0.05};
    Scalar dt_half_{0.025};
    Scalar dt_sixth_{0.05 / 6.0};
    Scalar rho_{1e4};

    Scalar mass_{1.0};
    Scalar grav_{9.81};

    size_t nx() const { return nx_; }
    size_t nu() const { return nu_; }
    int N() const { return N_; }
    Scalar dt() const { return dt_; }

    void Initialize(const int& N,
                    const Scalar& dt,
                    const Scalar& mass,
                    const Scalar& grav,
                    const Vector3& weightPos,
                    const Vector3& weightAtt,
                    const Vector3& weightVel,
                    const VXU& weightU)
    {
        N_ = N;
        dt_ = dt;
        dt_half_ = 0.5 * dt;
        dt_sixth_ = dt / 6.0;
        mass_ = mass;
        grav_ = grav;
        weightPos_ = weightPos;
        weightAtt_ = weightAtt;
        weightVel_ = weightVel;
        weightU_ = weightU;
    }

    void SetReferences(const std::vector<VXX>& x_refs, const std::vector<VXU>& u_refs)
    {
        x_refs_ = x_refs;
        u_refs_ = u_refs;
    }

    void SetXConstraints(const std::vector<int>& x_idx,
                         const std::vector<Scalar>& x_lb,
                         const std::vector<Scalar>& x_ub)
    {
        x_idx_ = x_idx;
        x_lb_ = x_lb;
        x_ub_ = x_ub;
    }

    void SetUConstraints(const std::vector<int>& u_idx,
                         const std::vector<Scalar>& u_lb,
                         const std::vector<Scalar>& u_ub)
    {
        u_idx_ = u_idx;
        u_lb_ = u_lb;
        u_ub_ = u_ub;
    }

    // Continuous-time dynamics: x_dot = f(x, u)
    VXX Dynamics(const VXX& x, const VXU& u)
    {
        VXX x_dot;

        auto& q_x = x(3);
        auto& q_y = x(4);
        auto& q_z = x(5);
        auto& q_w = x(6);
        auto& v_x = x(7);
        auto& v_y = x(8);
        auto& v_z = x(9);

        auto& c = u(0);
        auto& w_x = u(1);
        auto& w_y = u(2);
        auto& w_z = u(3);

        x_dot << v_x,
                 v_y,
                 v_z,
                 0.5 * (w_x * q_w + w_z * q_y - w_y * q_z),
                 0.5 * (w_y * q_w - w_z * q_x + w_x * q_z),
                 0.5 * (w_z * q_w + w_y * q_x - w_x * q_y),
                 0.5 * (-w_x * q_x - w_y * q_y - w_z * q_z),
                 2 * (q_w * q_y + q_x * q_z) * c / mass_,
                 2 * (q_y * q_z - q_w * q_x) * c / mass_,
                 (1 - 2 * q_x * q_x - 2 * q_y * q_y) * c / mass_ - grav_;

        return x_dot;
    }

    // Dynamics with Jacobians
    VXX DynamicsJac(const VXX& x, const VXU& u, MXA& jacX, MXB& jacU)
    {
        VXX x_dot;

        auto& q_x = x(3);
        auto& q_y = x(4);
        auto& q_z = x(5);
        auto& q_w = x(6);
        auto& v_x = x(7);
        auto& v_y = x(8);
        auto& v_z = x(9);

        auto& c = u(0);
        auto& w_x = u(1);
        auto& w_y = u(2);
        auto& w_z = u(3);

        const auto a_1 = 2 * (q_w * q_y + q_x * q_z) / mass_;
        const auto a_2 = 2 * (q_y * q_z - q_w * q_x) / mass_;
        const auto a_3 = (1 - 2 * q_x * q_x - 2 * q_y * q_y) / mass_;

        x_dot << v_x,
                 v_y,
                 v_z,
                 0.5 * (w_x * q_w + w_z * q_y - w_y * q_z),
                 0.5 * (w_y * q_w - w_z * q_x + w_x * q_z),
                 0.5 * (w_z * q_w + w_y * q_x - w_x * q_y),
                 0.5 * (-w_x * q_x - w_y * q_y - w_z * q_z),
                 a_1 * c,
                 a_2 * c,
                 a_3 * c - grav_;

        const auto h_w_x = 0.5 * w_x;
        const auto h_w_y = 0.5 * w_y;
        const auto h_w_z = 0.5 * w_z;
        const auto h_q_w = 0.5 * q_w;
        const auto h_q_x = 0.5 * q_x;
        const auto h_q_y = 0.5 * q_y;
        const auto h_q_z = 0.5 * q_z;

        const auto dc_q_w = 2.0 * c * q_w / mass_;
        const auto dc_q_x = 2.0 * c * q_x / mass_;
        const auto dc_q_y = 2.0 * c * q_y / mass_;
        const auto dc_q_z = 2.0 * c * q_z / mass_;

        jacX.setZero();

        for (int i = 0; i < 3; ++i)
        {
            jacX(i, i + 7) = 1.0;
        }

        jacX(3, 3) = 0.0;
        jacX(3, 4) = h_w_z;
        jacX(3, 5) = -h_w_y;
        jacX(3, 6) = h_w_x;

        jacX(4, 3) = -h_w_z;
        jacX(4, 4) = 0.0;
        jacX(4, 5) = h_w_x;
        jacX(4, 6) = h_w_y;

        jacX(5, 3) = h_w_y;
        jacX(5, 4) = -h_w_x;
        jacX(5, 5) = 0.0;
        jacX(5, 6) = h_w_z;

        jacX(6, 3) = -h_w_x;
        jacX(6, 4) = -h_w_y;
        jacX(6, 5) = -h_w_z;
        jacX(6, 6) = 0.0;

        jacX(7, 3) = dc_q_z;
        jacX(7, 4) = dc_q_w;
        jacX(7, 5) = dc_q_x;
        jacX(7, 6) = dc_q_y;

        jacX(8, 3) = -dc_q_w;
        jacX(8, 4) = dc_q_z;
        jacX(8, 5) = dc_q_y;
        jacX(8, 6) = -dc_q_x;

        jacX(9, 3) = -2 * dc_q_x;
        jacX(9, 4) = -2 * dc_q_y;
        jacX(9, 5) = 0.0;
        jacX(9, 6) = 0.0;

        jacU.setZero();

        jacU(3, 1) = h_q_w;
        jacU(3, 2) = -h_q_z;
        jacU(3, 3) = h_q_y;

        jacU(4, 1) = h_q_z;
        jacU(4, 2) = h_q_w;
        jacU(4, 3) = -h_q_x;

        jacU(5, 1) = -h_q_y;
        jacU(5, 2) = h_q_x;
        jacU(5, 3) = h_q_w;

        jacU(6, 1) = -h_q_x;
        jacU(6, 2) = -h_q_y;
        jacU(6, 3) = -h_q_z;

        jacU(7, 0) = a_1;
        jacU(8, 0) = a_2;
        jacU(9, 0) = a_3;

        return x_dot;
    }

    // RK4 integration
    VXX PropagateRK4(const VXX& xk, const VXU& uk)
    {
        k0 = Dynamics(xk, uk);
        k1 = Dynamics(xk + dt_half_ * k0, uk);
        k2 = Dynamics(xk + dt_half_ * k1, uk);
        k3 = Dynamics(xk + dt_ * k2, uk);

        VXX xk_1 = xk + (k0 + 2.0 * k1 + 2.0 * k2 + k3) * dt_sixth_;
        return xk_1;
    }

    // RK4 integration with gradients
    VXX PropagateRK4Grad(const VXX& xk, const VXU& uk, MXA& grad_Fx, MXB& grad_Fu)
    {
        static const MXA II = MXA::Identity();

        k0 = DynamicsJac(xk, uk, df_x0, df_u0);
        k1 = DynamicsJac(xk + dt_half_ * k0, uk, df_x1, df_u1);
        k2 = DynamicsJac(xk + dt_half_ * k1, uk, df_x2, df_u2);
        k3 = DynamicsJac(xk + dt_ * k2, uk, df_x3, df_u3);

        grad_k0_xk = df_x0;
        grad_k0_uk = df_u0;
        grad_k1_xk.noalias() = df_x1 * (II + dt_half_ * grad_k0_xk);
        grad_k1_uk.noalias() = df_u1 + dt_half_ * df_x1 * grad_k0_uk;
        grad_k2_xk.noalias() = df_x2 * (II + dt_half_ * grad_k1_xk);
        grad_k2_uk.noalias() = df_u2 + dt_half_ * df_x2 * grad_k1_uk;
        grad_k3_xk.noalias() = df_x3 * (II + dt_ * grad_k2_xk);
        grad_k3_uk.noalias() = df_u3 + dt_ * df_x3 * grad_k2_uk;

        VXX xk_1 = xk + (k0 + 2.0 * k1 + 2.0 * k2 + k3) * dt_sixth_;
        grad_Fx.noalias() = II + (grad_k0_xk + 2.0 * grad_k1_xk + 2.0 * grad_k2_xk + grad_k3_xk) * dt_sixth_;
        grad_Fu.noalias() = (grad_k0_uk + 2.0 * grad_k1_uk + 2.0 * grad_k2_uk + grad_k3_uk) * dt_sixth_;

        return xk_1;
    }

    // State cost with angle-based attitude error
    Scalar StateCostGrad(const VXX& x, const VXX& xref, VXX& grad_x)
    {
        Vector3 e_p = x.segment<3>(0) - xref.segment<3>(0);
        Vector3 e_v = x.segment<3>(7) - xref.segment<3>(7);

        Vector4 q = x.segment<4>(3);
        Vector4 q_ref = xref.segment<4>(3);

        Scalar qx = q(0), qy = q(1), qz = q(2), qw = q(3);
        Scalar qx_ref = q_ref(0), qy_ref = q_ref(1), qz_ref = q_ref(2), qw_ref = q_ref(3);

        Vector4 q_aux;
        q_aux(0) = -qx * qw_ref + qw * qx_ref + qz * qy_ref - qy * qz_ref;
        q_aux(1) = -qy * qw_ref - qz * qx_ref + qw * qy_ref + qx * qz_ref;
        q_aux(2) = -qz * qw_ref + qy * qx_ref - qx * qy_ref + qw * qz_ref;
        q_aux(3) = qw * qw_ref + qx * qx_ref + qy * qy_ref + qz * qz_ref;

        // Ensure shortest rotation path: if q_aux.w < 0, flip the quaternion
        // This prevents the quaternion double-cover issue and ensures numerical stability
        Scalar sign_flip = 1.0;
        if (q_aux(3) < 0.0) {
            q_aux = -q_aux;
            sign_flip = -1.0;
        }

        constexpr Scalar eps = 1e-3;
        Scalar denom_sq = q_aux(3) * q_aux(3) + q_aux(2) * q_aux(2) + eps;
        Scalar denom = std::sqrt(denom_sq);
        Scalar inv_denom = 1.0 / denom;

        Scalar num_roll = q_aux(3) * q_aux(0) - q_aux(1) * q_aux(2);
        Scalar num_pitch = q_aux(3) * q_aux(1) + q_aux(0) * q_aux(2);
        Scalar num_yaw = q_aux(2);

        Vector3 e_att;
        e_att(0) = 2.0 * num_roll * inv_denom;
        e_att(1) = 2.0 * num_pitch * inv_denom;
        e_att(2) = 2.0 * num_yaw * inv_denom;

        Scalar cost_pos = e_p.cwiseProduct(e_p).cwiseProduct(weightPos_).sum() * dt();
        Scalar cost_att = e_att.cwiseProduct(e_att).cwiseProduct(weightAtt_).sum() * dt();
        Scalar cost_vel = e_v.cwiseProduct(e_v).cwiseProduct(weightVel_).sum() * dt();
        Scalar cost = cost_pos + cost_att + cost_vel;

        grad_x.segment<3>(0) = 2 * e_p.cwiseProduct(weightPos_) * dt();
        grad_x.segment<3>(7) = 2 * e_v.cwiseProduct(weightVel_) * dt();

        Scalar inv_denom_sq = inv_denom * inv_denom;
        Scalar d_denom_dqaux2 = q_aux(2) * inv_denom;
        Scalar d_denom_dqaux3 = q_aux(3) * inv_denom;

        Eigen::Matrix<Scalar, 3, 4> de_att_dqaux;
        de_att_dqaux.setZero();

        de_att_dqaux(0, 0) = 2.0 * (q_aux(3) * inv_denom);
        de_att_dqaux(0, 1) = 2.0 * (-q_aux(2) * inv_denom);
        de_att_dqaux(0, 2) = 2.0 * (-q_aux(1) * inv_denom - num_roll * d_denom_dqaux2 * inv_denom_sq);
        de_att_dqaux(0, 3) = 2.0 * (q_aux(0) * inv_denom - num_roll * d_denom_dqaux3 * inv_denom_sq);

        de_att_dqaux(1, 0) = 2.0 * (q_aux(2) * inv_denom);
        de_att_dqaux(1, 1) = 2.0 * (q_aux(3) * inv_denom);
        de_att_dqaux(1, 2) = 2.0 * (q_aux(0) * inv_denom - num_pitch * d_denom_dqaux2 * inv_denom_sq);
        de_att_dqaux(1, 3) = 2.0 * (q_aux(1) * inv_denom - num_pitch * d_denom_dqaux3 * inv_denom_sq);

        de_att_dqaux(2, 0) = 0.0;
        de_att_dqaux(2, 1) = 0.0;
        de_att_dqaux(2, 2) = 2.0 * (inv_denom - num_yaw * d_denom_dqaux2 * inv_denom_sq);
        de_att_dqaux(2, 3) = 2.0 * (-num_yaw * d_denom_dqaux3 * inv_denom_sq);

        Eigen::Matrix<Scalar, 4, 4> dqaux_dq;
        dqaux_dq(0, 0) = -qw_ref; dqaux_dq(0, 1) = -qz_ref; dqaux_dq(0, 2) = qy_ref;  dqaux_dq(0, 3) = qx_ref;
        dqaux_dq(1, 0) = qz_ref;  dqaux_dq(1, 1) = -qw_ref; dqaux_dq(1, 2) = -qx_ref; dqaux_dq(1, 3) = qy_ref;
        dqaux_dq(2, 0) = -qy_ref; dqaux_dq(2, 1) = qx_ref;  dqaux_dq(2, 2) = -qw_ref; dqaux_dq(2, 3) = qz_ref;
        dqaux_dq(3, 0) = qx_ref;  dqaux_dq(3, 1) = qy_ref;  dqaux_dq(3, 2) = qz_ref;  dqaux_dq(3, 3) = qw_ref;

        // Apply sign flip to Jacobian if quaternion was flipped
        dqaux_dq *= sign_flip;

        Vector3 weighted_e_att = e_att.cwiseProduct(weightAtt_);
        Vector4 grad_qaux;
        grad_qaux.noalias() = de_att_dqaux.transpose() * weighted_e_att;
        Vector4 grad_q;
        grad_q.noalias() = 2.0 * dt_ * dqaux_dq.transpose() * grad_qaux;

        grad_x.segment<4>(3) = grad_q;

        return cost;
    }

    Scalar InputCostGrad(const VXU& u, const VXU& uref, VXU& grad_u)
    {
        VXU eu = u - uref;
        Scalar cost = eu.cwiseProduct(eu).cwiseProduct(weightU_).sum() * dt();
        grad_u = 2 * eu.cwiseProduct(weightU_) * dt();
        return cost;
    }

    Scalar PathCostGrad(const VXX& x, const VXU& u,
                        const VXX& xref, const VXU& uref,
                        VXX& grad_x, VXU& grad_u)
    {
        Scalar state_cost = StateCostGrad(x, xref, grad_x);
        Scalar input_cost = InputCostGrad(u, uref, grad_u);
        return state_cost + input_cost;
    }

    Scalar TerminalCostGrad(const VXX& x, const VXX& xref, VXX& grad_x)
    {
        return StateCostGrad(x, xref, grad_x);
    }

    Scalar BoxConstraint(const Scalar& x, const Scalar& l, const Scalar& u, Scalar& grad)
    {
        Scalar lpen = l - x;
        if (lpen > 0)
        {
            Scalar lpen2 = lpen * lpen;
            grad = -rho_ * 3.0 * lpen2;
            return rho_ * lpen2 * lpen;
        }
        Scalar upen = x - u;
        if (upen > 0)
        {
            Scalar upen2 = upen * upen;
            grad = rho_ * 3.0 * upen2;
            return rho_ * upen2 * upen;
        }
        grad = 0.0;
        return 0.0;
    }

    Scalar AddGeneralConstraintGrad([[maybe_unused]] const VXX& x, const VXU& u,
                                     VXX& grad_x, VXU& grad_u)
    {
        Scalar penalty = 0.0;
        grad_x.setZero();
        grad_u.setZero();

        for (size_t i = 0; i < u_idx_.size(); ++i)
        {
            const Scalar u_val = u(u_idx_[i]);
            const Scalar lb = u_lb_[i];
            const Scalar ub = u_ub_[i];

            if (u_val >= lb && u_val <= ub)
            {
                continue;
            }

            Scalar gradControl = 0.0;
            penalty += BoxConstraint(u_val, lb, ub, gradControl);
            grad_u(u_idx_[i]) += gradControl;
        }

        return penalty;
    }
};

} // namespace NMPC
