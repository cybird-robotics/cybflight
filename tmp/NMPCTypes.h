// NMPCTypes.h - Core type definitions for NMPC controller
// Migrated from FSC Lab's cyblib for UE5
// Copyright 2024 FSC Lab - MIT License

#pragma once

#include <limits>
#include <type_traits>
#include <functional>
#include <cmath>
#include <algorithm>

// Disable Eigen warnings for UE5 compatibility
THIRD_PARTY_INCLUDES_START
#include <Eigen/Dense>
THIRD_PARTY_INCLUDES_END

namespace NMPC
{

//=============================================================================
// Scalar and Integer Types
//=============================================================================

using Scalar = double;
using Integer = int;

using Eigen::Index;

inline constexpr Scalar kInf = std::numeric_limits<Scalar>::infinity();
inline constexpr Scalar kNaN = std::numeric_limits<Scalar>::quiet_NaN();

//=============================================================================
// Matrix and Vector Type Definitions
//=============================================================================

static constexpr int kDynamic = Eigen::Dynamic;

template <int Rows = kDynamic, int Cols = Rows>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols>;

using Matrix2 = Matrix<2, 2>;
using Matrix3 = Matrix<3, 3>;
using Matrix4 = Matrix<4, 4>;
using Matrix6 = Matrix<6, 6>;

template <int Rows = kDynamic>
using Vector = Matrix<Rows, 1>;

using Vector2 = Vector<2>;
using Vector3 = Vector<3>;
using Vector4 = Vector<4>;
using Vector6 = Vector<6>;
using Vector7 = Vector<7>;

template <int Cols = kDynamic>
using RowVector = Matrix<1, Cols>;

using Quaternion = Eigen::Quaternion<Scalar>;
using AngleAxis = Eigen::AngleAxis<Scalar>;

template <int Rows = kDynamic, int Cols = Rows>
using Array = Eigen::Array<Scalar, Rows, Cols>;

template <int Rows = kDynamic>
using ArrayVector = Array<Rows, 1>;

template <class Derived>
using Ref = Eigen::Ref<Derived>;

using VectorRef = Ref<Vector<>>;
using VectorCRef = Ref<const Vector<>>;

using MatrixRef = Ref<Matrix<>>;
using MatrixCRef = Ref<const Matrix<>>;

template <class Derived>
using Map = Eigen::Map<Derived>;

//=============================================================================
// Mathematical Constants
//=============================================================================

inline constexpr Scalar kPi = 3.14159265358979323846;
inline constexpr Scalar kSqrt2 = 1.41421356237309504880;
inline constexpr Scalar kGravity = 9.80665;
inline constexpr Scalar kRadPerDeg = 0.017453292519943295;
inline constexpr Scalar kDegPerRad = 57.2957795130823229;

static Vector3 kGravityVec{0, 0, -kGravity};
static const Matrix2 I2{Matrix2::Identity()};
static const Matrix3 I3{Matrix3::Identity()};
static const Matrix4 I4{Matrix4::Identity()};

//=============================================================================
// Tolerance Constants
//=============================================================================

template <typename T>
constexpr T kToleranceRelative = 1e-5;

template <typename T>
constexpr T kToleranceAbsolute = 1e-8;

//=============================================================================
// Math Utility Functions
//=============================================================================

template <typename T>
constexpr T wrapToPi(T angle)
{
    const T result = std::fmod(angle + kPi, 2.0 * kPi);
    return result <= 0.0 ? result + kPi : result - kPi;
}

template <typename T>
constexpr T wrapTo2Pi(T angle)
{
    T res = std::fmod(angle, T{2} * kPi);
    return res < T{0} ? res + T{2} * kPi : res;
}

template <typename T>
constexpr T deg2rad(T deg) noexcept
{
    return deg * kRadPerDeg;
}

template <typename T>
constexpr T rad2deg(T rad) noexcept
{
    return rad * kDegPerRad;
}

template <typename Derived, typename LBDerived, typename UBDerived>
auto Clip(const Eigen::MatrixBase<Derived>& v,
          const Eigen::MatrixBase<LBDerived>& lb,
          const Eigen::MatrixBase<UBDerived>& ub)
{
    return v.cwiseMin(ub).cwiseMax(lb);
}

template <typename Derived>
auto Clip(const Eigen::MatrixBase<Derived>& v,
          typename Derived::Scalar lb,
          typename Derived::Scalar ub)
{
    return v.cwiseMin(ub).cwiseMax(lb);
}

} // namespace NMPC
