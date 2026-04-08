// Deeper algebraic rotation utilities not even covered by nalgebra

use core::marker::Copy;
use nalgebra::{
    Matrix, Matrix3, Matrix4, Quaternion, RealField, Rotation3, Storage, U1, U3, UnitQuaternion,
    Vector3,
};
use num_traits::NumCast;

/// Small-magnitude check used as the branch predicate in the C++ source.
/// Mirrors `cyb::IsClose(a, b)` for the Taylor-vs-closed-form switches in the
/// rotation utilities below. Tolerance is fixed at `1e-6`, matching float-cmp's
/// default `epsilon` for `f32` (the only realistic scalar in this firmware).
#[inline]
fn is_close<T: RealField + Copy + NumCast>(a: T, b: T) -> bool {
    (a - b).abs() <= cast::<T>(1e-6)
}

#[inline]
fn cast<T: RealField + Copy + NumCast>(v: f64) -> T {
    T::from(v).unwrap()
}

pub fn hat<T, S>(v: &Matrix<T, U3, U1, S>) -> Matrix3<T>
where
    T: RealField + Copy,
    S: Storage<T, U3, U1>,
{
    Matrix3::new(
        T::zero(),
        -v[2],
        v[1],
        v[2],
        T::zero(),
        -v[0],
        -v[1],
        v[0],
        T::zero(),
    )
}

pub fn vee<T, S>(m: &Matrix<T, U3, U3, S>) -> Vector3<T>
where
    T: RealField + Copy,
    S: Storage<T, U3, U3>,
{
    Vector3::new(m[(2, 1)], m[(0, 2)], m[(1, 0)])
}

/// Left-multiplication matrix `[q]_L`. Defining identity (Hamilton):
/// `coeffs(q · p) = [q]_L · coeffs(p)`, where `coeffs` is `(x, y, z, w)`
/// (matching `nalgebra::Quaternion::coords` and `cyb::qLeft`).
pub fn left_quaternion_matrix<T: RealField + Copy>(q: &UnitQuaternion<T>) -> Matrix4<T> {
    let mut m = Matrix4::zeros();
    let w = q.scalar();
    let v = q.vector();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&hat(&v));
    m.fixed_view_mut::<3, 1>(0, 3).copy_from(&v);
    m.fixed_view_mut::<1, 3>(3, 0).copy_from(&-v.transpose());
    // NOTE: nalgebra's `Matrix::diagonal()` returns the diagonal *by value*,
    // so `m.diagonal().fill(w)` writes to a temporary and is silently a
    // no-op. Set the diagonal entries explicitly instead.
    m[(0, 0)] = w;
    m[(1, 1)] = w;
    m[(2, 2)] = w;
    m[(3, 3)] = w;
    m
}

/// Right-multiplication matrix `[q]_R`. Defining identity (Hamilton):
/// `coeffs(p · q) = [q]_R · coeffs(p)`, again with `coeffs` in `(x, y, z, w)`
/// order. Layout matches `cyb::qRight`.
pub fn right_quaternion_matrix<T: RealField + Copy>(q: &UnitQuaternion<T>) -> Matrix4<T> {
    let mut m = Matrix4::zeros();
    let w = q.scalar();
    let v = q.vector();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&-hat(&v));
    m.fixed_view_mut::<3, 1>(0, 3).copy_from(&v);
    m.fixed_view_mut::<1, 3>(3, 0).copy_from(&-v.transpose());
    m[(0, 0)] = w;
    m[(1, 1)] = w;
    m[(2, 2)] = w;
    m[(3, 3)] = w;
    m
}

// ─────────────────────────────────────────────────────────────────────────────
// Translations of cyblib/math/rotation.hpp functions not already covered by
// nalgebra's UnitQuaternion / Rotation3 APIs.
//
// `skew`/`unskew` map to `hat`/`vee` above; `qLeft`/`qRight` map to
// `left_quaternion_matrix`/`right_quaternion_matrix`. Everything else lives
// here.
// ─────────────────────────────────────────────────────────────────────────────

/// Logarithmic map of a unit quaternion: `q ↦ θ·n` where `q = (cos θ/2,
/// n sin θ/2)`. Returns the rotation vector (axis times angle, in `[-π, π]`).
///
/// Uses a 3rd-order series in `‖vec(q)‖²` near identity to dodge the `0/0`
/// in `2·atan2(n, w) / n`. Picks the negative-w branch via `atan2(-n, -w)`
/// so the wrap to `(-π, π]` happens inside the trig call.
pub fn quaternion_to_angle_axis<T: RealField + Copy + NumCast>(
    quaternion: &UnitQuaternion<T>,
) -> Vector3<T> {
    let v = quaternion.vector();
    let squared_n = v.dot(&v);
    let w = quaternion.scalar();

    let two_atan_nbyw_by_n = if is_close(squared_n, T::zero()) {
        // n=0 ⇒ for a normalized quaternion, w=±1; series of 2·atan2(n,w)/n in n²
        let two: T = cast(2.0);
        let two_thirds: T = cast(2.0 / 3.0);
        let squared_w = w * w;
        two / w - two_thirds * squared_n / (w * squared_w)
    } else {
        let n = squared_n.sqrt();
        // w<0 ⇒ θ>π; the wrap to (-π, π] is folded into atan2(-n, -w).
        let atan_nbyw = if w < T::zero() {
            (-n).atan2(-w)
        } else {
            n.atan2(w)
        };
        let two: T = cast(2.0);
        two * atan_nbyw / n
    };

    v * two_atan_nbyw_by_n
}

/// Exponential map of a rotation vector to a 3×3 rotation matrix
/// (Rodrigues' formula). Series-expanded near zero to avoid `sin θ / θ`
/// blowing up.
pub fn angle_axis_to_rotation_matrix<T, S>(angle_axis: &Matrix<T, U3, U1, S>) -> Matrix3<T>
where
    T: RealField + Copy + NumCast,
    S: Storage<T, U3, U1>,
{
    let theta_sq = angle_axis.dot(angle_axis);
    let hat_phi = hat(angle_axis);
    let hat_phi_sq = hat_phi * hat_phi;
    let identity = Matrix3::<T>::identity();

    if is_close(theta_sq, T::zero()) {
        let half: T = cast(0.5);
        identity + hat_phi + hat_phi_sq * half
    } else {
        let theta = theta_sq.sqrt();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        identity + hat_phi_sq * ((T::one() - cos_theta) / theta_sq) + hat_phi * (sin_theta / theta)
    }
}

/// Exponential map of a rotation vector to a unit quaternion. Uses a
/// 4th-order series in `‖θ‖²` near zero to dodge the `0/0` in
/// `sin(θ/2)/θ`, and the closed form everywhere else.
pub fn angle_axis_to_quaternion<T, S>(angle_axis: &Matrix<T, U3, U1, S>) -> UnitQuaternion<T>
where
    T: RealField + Copy + NumCast,
    S: Storage<T, U3, U1>,
{
    let angle_sq = angle_axis.dot(angle_axis);
    let (real_factor, imag_factor) = if is_close(angle_sq, T::zero()) {
        let theta_po4 = angle_sq * angle_sq;
        let imag = cast::<T>(0.5) - cast::<T>(1.0 / 48.0) * angle_sq
            + cast::<T>(1.0 / 3840.0) * theta_po4;
        let real = T::one() - cast::<T>(1.0 / 8.0) * angle_sq
            + cast::<T>(1.0 / 384.0) * theta_po4;
        (real, imag)
    } else {
        let theta = angle_sq.sqrt();
        let half_theta = cast::<T>(0.5) * theta;
        (half_theta.cos(), half_theta.sin() / theta)
    };

    let q = Quaternion::new(
        real_factor,
        imag_factor * angle_axis[0],
        imag_factor * angle_axis[1],
        imag_factor * angle_axis[2],
    );
    UnitQuaternion::new_unchecked(q)
}

/// Roll/pitch/yaw (XYZ intrinsic, applied as Rz·Ry·Rx) → unit quaternion.
/// Closed form, no Euler-angle gimbal handling needed.
pub fn euler_angles_rpy_to_quaternion<T, S>(rpy: &Matrix<T, U3, U1, S>) -> UnitQuaternion<T>
where
    T: RealField + Copy + NumCast,
    S: Storage<T, U3, U1>,
{
    let half: T = cast(0.5);
    let r = rpy[0] * half;
    let p = rpy[1] * half;
    let y = rpy[2] * half;
    let (sr, cr) = (r.sin(), r.cos());
    let (sp, cp) = (p.sin(), p.cos());
    let (sy, cy) = (y.sin(), y.cos());

    let qw = cr * cp * cy + sr * sp * sy;
    let qx = sr * cp * cy - cr * sp * sy;
    let qy = cr * sp * cy + sr * cp * sy;
    let qz = cr * cp * sy - sr * sp * cy;
    UnitQuaternion::new_unchecked(Quaternion::new(qw, qx, qy, qz))
}

/// Unit quaternion → roll/pitch/yaw (XYZ intrinsic). Uses the
/// `2·atan2(√(1+s), √(1−s)) − π/2` form for pitch so it stays well-defined
/// at the gimbal-lock point (`s = ±1`) instead of NaN-ing through `asin`.
pub fn quaternion_to_euler_angles_rpy<T: RealField + Copy + NumCast>(
    q: &UnitQuaternion<T>,
) -> Vector3<T> {
    let one = T::one();
    let two: T = cast(2.0);
    let half_pi = T::frac_pi_2();

    let qw = q.scalar();
    let qx = q.i;
    let qy = q.j;
    let qz = q.k;

    // roll (x)
    let sinr_cosp = two * (qw * qx + qy * qz);
    let cosr_cosp = one - two * (qx * qx + qy * qy);
    let roll = sinr_cosp.atan2(cosr_cosp);

    // pitch (y) — clamp to [0, 2] guards against tiny FP overshoot.
    let s = two * (qw * qy - qx * qz);
    let one_plus = (one + s).max(T::zero());
    let one_minus = (one - s).max(T::zero());
    let sinp = one_plus.sqrt();
    let cosp = one_minus.sqrt();
    let pitch = two * sinp.atan2(cosp) - half_pi;

    // yaw (z)
    let siny_cosp = two * (qw * qz + qx * qy);
    let cosy_cosp = one - two * (qy * qy + qz * qz);
    let yaw = siny_cosp.atan2(cosy_cosp);

    Vector3::new(roll, pitch, yaw)
}

/// Direct port of the rotation-matrix → roll/pitch/yaw extractor in the
/// C++ source. Note the `-atan2(R(2,0), …)` for pitch — the negative sign
/// is preserved verbatim from `rotation.hpp`.
pub fn rotation_matrix_to_euler_angles_rpy<T, S>(r: &Matrix<T, U3, U3, S>) -> Vector3<T>
where
    T: RealField + Copy,
    S: Storage<T, U3, U3>,
{
    let r21 = r[(2, 1)];
    let r22 = r[(2, 2)];
    let r20 = r[(2, 0)];
    let r10 = r[(1, 0)];
    let r00 = r[(0, 0)];

    let roll = r21.atan2(r22);
    let pitch = -r20.atan2((r21 * r21 + r22 * r22).sqrt());
    let yaw = r10.atan2(r00);
    Vector3::new(roll, pitch, yaw)
}

/// Roll/pitch/yaw → 3×3 rotation matrix (R = Rz·Ry·Rx). Direct port of
/// the closed-form expansion in the C++ source.
pub fn euler_angles_rpy_to_rotation_matrix<T, S>(rpy: &Matrix<T, U3, U1, S>) -> Matrix3<T>
where
    T: RealField + Copy,
    S: Storage<T, U3, U1>,
{
    let r = rpy[0];
    let p = rpy[1];
    let y = rpy[2];
    let (sr, cr) = (r.sin(), r.cos());
    let (sp, cp) = (p.sin(), p.cos());
    let (sy, cy) = (y.sin(), y.cos());

    Matrix3::new(
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    )
}

/// Yaw of `q` defined as the angle between world `x̂` and the projection of
/// `q·x̂` onto the world XY plane. Returns `default_yaw` when the body
/// x-axis is too close to vertical for the projection to be meaningful.
pub fn quaternion_to_yaw<T: RealField + Copy + NumCast>(
    q: &UnitQuaternion<T>,
    default_yaw: T,
) -> T {
    let tolerance: T = cast(1e-3);
    let x_b = q * Vector3::x();
    let x_proj = Vector3::new(x_b.x, x_b.y, T::zero());
    if x_proj.norm() < tolerance {
        return default_yaw;
    }
    let x_proj_norm = x_proj.normalize();
    let cross = Vector3::x().cross(&x_proj_norm);
    // Clamp to defend against tiny FP overshoot pushing asin into NaN territory.
    let z = cross.z.max(-T::one()).min(T::one());
    let angle = z.asin();
    if x_proj_norm.x >= T::zero() {
        angle
    } else if x_proj_norm.y >= T::zero() {
        T::pi() - angle
    } else {
        -T::pi() - angle
    }
}

/// Shortest-arc rotation taking world `ẑ` into the unit vector `v`. Yaw of
/// the result is zero. Singular at `v = -ẑ` (denominator goes to zero).
pub fn quaternion_from_unit_z_to_v<T, S>(v: &Matrix<T, U3, U1, S>) -> UnitQuaternion<T>
where
    T: RealField + Copy + NumCast,
    S: Storage<T, U3, U1>,
{
    let v_norm = v / v.norm();
    let two: T = cast(2.0);
    let half: T = cast(0.5);
    let tilt_den = (two * (T::one() + v_norm[2])).sqrt();

    let qx = -v_norm[1] / tilt_den;
    let qy = v_norm[0] / tilt_den;
    let qz = T::zero();
    let qw = half * tilt_den;
    UnitQuaternion::new_unchecked(Quaternion::new(qw, qx, qy, qz))
}

/// Build a unit quaternion whose body z-axis is `v` and whose yaw is
/// `yaw`. Two parameterizations:
///
/// * `use_tilt = true`: tilt-then-yaw closed form (singular at
///   `v.z == -1`, fast).
/// * `use_tilt = false`: cross-product construction (singular when the
///   yaw-aligned y-axis is parallel to `v`).
pub fn quaternion_from_zb_and_yaw<T, S>(
    v: &Matrix<T, U3, U1, S>,
    yaw: T,
    use_tilt: bool,
) -> UnitQuaternion<T>
where
    T: RealField + Copy + NumCast,
    S: Storage<T, U3, U1>,
{
    let mut z_b: Vector3<T> = Vector3::new(v[0], v[1], v[2]);
    if !is_close(z_b.norm(), T::one()) {
        z_b.normalize_mut();
    }

    if use_tilt {
        let half: T = cast(0.5);
        let two: T = cast(2.0);
        let c_half_yaw = (half * yaw).cos();
        let s_half_yaw = (half * yaw).sin();

        let tilt_den = (two * (T::one() + z_b.z)).sqrt();
        let tilt0 = half * tilt_den;
        let tilt1 = -z_b.y / tilt_den;
        let tilt2 = z_b.x / tilt_den;

        let qx = tilt1 * c_half_yaw + tilt2 * s_half_yaw;
        let qy = tilt2 * c_half_yaw - tilt1 * s_half_yaw;
        let qz = tilt0 * s_half_yaw;
        let qw = tilt0 * c_half_yaw;
        UnitQuaternion::new_unchecked(Quaternion::new(qw, qx, qy, qz))
    } else {
        let q_yaw = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw);
        let y_c = q_yaw * Vector3::y();
        let x_b = y_c.cross(&z_b).normalize();
        let y_b = z_b.cross(&x_b).normalize();
        let r_wb = Matrix3::from_columns(&[x_b, y_b, z_b]);
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_wb))
    }
}

/// Rotation vector taking unit normal `a` to unit normal `b`. When `a` and
/// `b` are antiparallel, returns `π · a_perp` (caller supplies an in-plane
/// perpendicular to disambiguate the otherwise undefined axis).
pub fn angle_axis_from_two_normals<T: RealField + Copy + NumCast>(
    a: &Vector3<T>,
    b: &Vector3<T>,
    a_perp: &Vector3<T>,
) -> Vector3<T> {
    let cross = a.cross(b);
    let cross_norm = cross.norm();
    let c = a.dot(b);
    // Clamp guards against |c| > 1 from FP error → acos NaN.
    let c_clamped = c.max(-T::one()).min(T::one());
    let angle = c_clamped.acos();
    if is_close(cross_norm, T::zero()) {
        if c > T::zero() {
            cross
        } else {
            a_perp * T::pi()
        }
    } else {
        cross * (angle / cross_norm)
    }
}

/// Returns true when `q1` and `-q2` are closer than `q1` and `q2`.
/// Useful before measurement updates that compare quaternion residuals
/// componentwise (q and -q encode the same rotation but break naive diffs).
pub fn should_flip_quaternion<T: RealField + Copy>(
    q1: &UnitQuaternion<T>,
    q2: &UnitQuaternion<T>,
) -> bool {
    let c1 = q1.as_vector();
    let c2 = q2.as_vector();
    let diff = c1 - c2;
    let sum = c1 + c2;
    diff.dot(&diff) > sum.dot(&sum)
}
