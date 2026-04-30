// Deeper algebraic rotation utilities not even covered by nalgebra

use core::marker::Copy;
use nalgebra::{
    Matrix, Matrix3, Matrix4, Quaternion, RealField, Rotation3, Storage, UnitQuaternion, Vector3,
    U1, U3,
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
/// * `use_tilt = true`: tilt-then-yaw closed form. The unique
///   singularity is at `v.z == -1` (drone fully inverted), where the
///   construction substitutes the canonical yaw-consistent 180° flip
///   `q = (0, cos(yaw/2), sin(yaw/2), 0)` — body-x heading remains
///   `(cos yaw, sin yaw, 0)`, matching the upright convention for the
///   same `yaw` input. Discontinuous in `v` at the pole (unavoidable;
///   SO(3) has no continuous global parameterisation), but finite and
///   meaningful there.
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

        // Singularity guard at `z_b.z == -1` (drone fully inverted). The
        // closed form below has `tilt_den = sqrt(2·(1 + z_b.z))` which
        // collapses to 0/0 there, returning NaN. Substitute the canonical
        // "yaw-consistent" 180° flip:
        //
        //   q = (0, cos(yaw/2), sin(yaw/2), 0)
        //
        // This is the 180° rotation about a horizontal axis at angle
        // `yaw/2` from world-x, equivalently the composition
        // `q_tilt(180° about world x) ⊗ q_yaw(-yaw)`. The yaw is *flipped*
        // so the body-x heading at the inverted pose remains
        // `(cos yaw, sin yaw, 0)` — the same world-frame direction the
        // upright closed form gives for the same `yaw` input. Without
        // the flip the controller's yaw command would invert direction
        // when the body inverts; with it the yaw input keeps a
        // consistent meaning across the pole.
        //
        // Discontinuous in `z_b` at the pole — unavoidable; SO(3) admits
        // no continuous global parameterisation over the unit sphere.
        let one_plus_zbz = T::one() + z_b.z;
        let pole_eps: T = cast(1.0e-6);
        if one_plus_zbz < pole_eps {
            return UnitQuaternion::new_unchecked(Quaternion::new(
                T::zero(),
                c_half_yaw,
                s_half_yaw,
                T::zero(),
            ));
        }

        let tilt_den = (two * one_plus_zbz).sqrt();
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
