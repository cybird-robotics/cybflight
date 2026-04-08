//! Convention & correctness tests for `cybflight_core::rotation`.
//!
//! Two goals:
//!
//! 1. Self-consistency: round-trips, known closed-form values, and the
//!    defining algebraic identities for each utility we ported from
//!    `cyblib/math/rotation.hpp`.
//! 2. Cross-check against nalgebra's built-in `UnitQuaternion` / `Rotation3`
//!    APIs. Where the conventions agree we assert numerical equivalence;
//!    where they intentionally diverge (gimbal-lock branches, Taylor-vs-
//!    closed-form near identity) we document the divergence with a test
//!    that pins down *both* answers.
//!
//! Run: `cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
//!       --features std --test rotation_convention`

use cybflight_core::rotation::{
    angle_axis_from_two_normals, angle_axis_to_quaternion, angle_axis_to_rotation_matrix,
    euler_angles_rpy_to_quaternion, euler_angles_rpy_to_rotation_matrix, hat,
    left_quaternion_matrix, quaternion_from_unit_z_to_v, quaternion_from_zb_and_yaw,
    quaternion_to_angle_axis, quaternion_to_euler_angles_rpy, quaternion_to_yaw,
    right_quaternion_matrix, rotation_matrix_to_euler_angles_rpy, should_flip_quaternion, vee,
};
use nalgebra::{Matrix3, Quaternion, Rotation3, UnitQuaternion, Vector3, Vector4};

// ─── tolerances ─────────────────────────────────────────────────────────────
const TOL_TIGHT: f32 = 1e-6; // exact-ish algebraic identities
const TOL: f32 = 1e-5; // single trig round-trip
const TOL_LOOSE: f32 = 1e-4; // chained trig / large angles
const TOL_TAYLOR: f32 = 5e-4; // 4th-order Taylor truncation in angle_axis_to_quaternion

// ─── helpers ────────────────────────────────────────────────────────────────

/// Compare two unit quaternions modulo sign (q and -q encode the same rotation).
fn quat_close(a: &UnitQuaternion<f32>, b: &UnitQuaternion<f32>, tol: f32) -> bool {
    let d_minus = (a.coords - b.coords).norm();
    let d_plus = (a.coords + b.coords).norm();
    d_minus.min(d_plus) < tol
}

fn vec_close(a: &Vector3<f32>, b: &Vector3<f32>, tol: f32) -> bool {
    (a - b).norm() < tol
}

fn mat_close(a: &Matrix3<f32>, b: &Matrix3<f32>, tol: f32) -> bool {
    (a - b).norm() < tol
}

/// Sample roll/pitch/yaw triples covering several octants and one each
/// near the poles. Pitch deliberately stays away from ±π/2 — gimbal lock
/// is exercised in its own dedicated test.
fn sample_rpys() -> Vec<Vector3<f32>> {
    use core::f32::consts::PI;
    vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.1, 0.2, 0.3),
        Vector3::new(-0.4, 0.5, -0.6),
        Vector3::new(1.0, -0.7, 2.0),
        Vector3::new(-1.2, 0.8, -2.5),
        Vector3::new(PI / 6.0, PI / 4.0, PI / 3.0),
        Vector3::new(-PI / 6.0, -PI / 4.0, -PI / 3.0),
        Vector3::new(0.0, 0.4, PI - 0.1),
    ]
}

/// Sample rotation vectors from "essentially zero" through "near π".
fn sample_axes() -> Vec<Vector3<f32>> {
    // NOTE: exact zero is deliberately omitted. The C++ source's
    // `angleAxisToQuaternion` uses `sin(half_theta)/theta` in the small-angle
    // branch, which is `0/0 = NaN` at θ = 0. Per the "do not challenge
    // rotation.hpp" instruction the Rust port preserves this verbatim, so we
    // simply avoid the singular input here.
    vec![
        Vector3::new(1e-9, 0.0, 0.0), // tickles the small-angle branch
        Vector3::new(0.0, 1e-4, 0.0),
        Vector3::new(0.1, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 0.5),
        Vector3::new(0.3, -0.4, 0.5),
        Vector3::new(-0.6, 0.7, -0.8),
        Vector3::new(1.0, 1.0, 1.0),                              // ≈1.73 rad
        Vector3::new(2.0, 0.0, 0.0),                              // ≈2 rad
        Vector3::new(2.5, -1.0, 0.5).normalize() * (2.9_f32),     // close to π
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// hat / vee
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hat_is_skew_symmetric() {
    for v in [
        Vector3::new(1.0, 2.0, 3.0),
        Vector3::new(-0.5, 0.7, -0.2),
        Vector3::zeros(),
    ] {
        let h = hat(&v);
        assert!(mat_close(&h, &(-h.transpose()), TOL_TIGHT));
        // hat(v) * v == 0  (cross product of v with itself)
        assert!(vec_close(&(h * v), &Vector3::zeros(), TOL_TIGHT));
    }
}

#[test]
fn hat_v_acts_as_cross_product() {
    let a = Vector3::new(0.3, -0.4, 0.5);
    let b = Vector3::new(1.1, 0.7, -0.2);
    assert!(vec_close(&(hat(&a) * b), &a.cross(&b), TOL_TIGHT));
}

#[test]
fn vee_is_left_inverse_of_hat() {
    for v in [
        Vector3::new(1.0, 2.0, 3.0),
        Vector3::new(-0.5, 0.7, -0.2),
    ] {
        assert!(vec_close(&vee(&hat(&v)), &v, TOL_TIGHT));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// left_quaternion_matrix / right_quaternion_matrix
//
// Defining identity (Hamilton convention, matching the C++ source):
//   coeffs(q1 * q2)  ==  qLeft(q1)  * coeffs(q2)
//   coeffs(q1 * q2)  ==  qRight(q2) * coeffs(q1)
// where coeffs is laid out as (x, y, z, w) — nalgebra's `Quaternion::coords`.
// ═══════════════════════════════════════════════════════════════════════════

fn random_unit_quaternions() -> Vec<UnitQuaternion<f32>> {
    sample_rpys()
        .into_iter()
        .map(|rpy| UnitQuaternion::from_euler_angles(rpy.x, rpy.y, rpy.z))
        .collect()
}

#[test]
fn left_quaternion_matrix_realizes_left_multiplication() {
    let qs = random_unit_quaternions();
    for q1 in &qs {
        for q2 in &qs {
            let prod = q1 * q2;
            let lhs: Vector4<f32> = left_quaternion_matrix(q1) * q2.as_vector();
            let rhs: Vector4<f32> = *prod.as_vector();
            assert!(
                (lhs - rhs).norm() < TOL,
                "qLeft(q1)·q2 != q1·q2:\n  q1={:?}\n  q2={:?}\n  lhs={:?}\n  rhs={:?}",
                q1.coords.as_slice(),
                q2.coords.as_slice(),
                lhs.as_slice(),
                rhs.as_slice(),
            );
        }
    }
}

#[test]
fn right_quaternion_matrix_realizes_right_multiplication() {
    let qs = random_unit_quaternions();
    for q1 in &qs {
        for q2 in &qs {
            let prod = q1 * q2;
            let lhs: Vector4<f32> = right_quaternion_matrix(q2) * q1.as_vector();
            let rhs: Vector4<f32> = *prod.as_vector();
            assert!(
                (lhs - rhs).norm() < TOL,
                "qRight(q2)·q1 != q1·q2:\n  q1={:?}\n  q2={:?}\n  lhs={:?}\n  rhs={:?}",
                q1.coords.as_slice(),
                q2.coords.as_slice(),
                lhs.as_slice(),
                rhs.as_slice(),
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// quaternion_to_angle_axis  ↔  angle_axis_to_quaternion  ↔  nalgebra
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn aa_to_quat_is_unit_norm_for_small_angles() {
    // The C++ source uses a 4th-order Taylor expansion in `θ²` for the
    // exp map's real/imag scale factors as the *default* branch. The
    // truncation error in ‖q‖ scales as `O(θ⁶)`, so the unit-norm
    // guarantee tightens dramatically as θ shrinks. We test the regime
    // where the firmware actually operates: |θ| ≲ 1 rad.
    for axis in sample_axes() {
        if axis.norm() > 1.0 {
            continue;
        }
        let q = angle_axis_to_quaternion(&axis);
        assert!(
            (q.norm() - 1.0).abs() < TOL_TAYLOR,
            "non-unit quat for axis={:?}: ‖q‖={}",
            axis,
            q.norm()
        );
    }
}

#[test]
fn aa_to_quat_norm_drift_pinned_for_large_angles() {
    // Pin (don't fight) the known Taylor-truncation drift at large angles.
    // At ‖θ‖ = 2 rad the 4th-order series under-normalizes by ~1e-3 — this
    // is a documented property of the C++ source's branch choice.
    let axis = Vector3::new(2.0_f32, 0.0, 0.0);
    let q = angle_axis_to_quaternion(&axis);
    let drift = (q.norm() - 1.0).abs();
    // Sanity bracket: drift is in the milli-units, not micro and not unity.
    assert!(drift > 1e-4 && drift < 1e-2, "unexpected drift: {}", drift);
}

#[test]
fn quat_to_aa_round_trip_through_quaternion() {
    // ours(ours⁻¹(θ)) == θ. Restrict to ‖θ‖ ≤ 1 rad where the Taylor
    // truncation in `angle_axis_to_quaternion` is well within tolerance.
    for axis in sample_axes() {
        if axis.norm() > 1.0 {
            continue;
        }
        let q = angle_axis_to_quaternion(&axis);
        let back = quaternion_to_angle_axis(&q);
        assert!(
            vec_close(&back, &axis, TOL_LOOSE),
            "round trip failed: axis={:?} back={:?}",
            axis,
            back
        );
    }
}

#[test]
fn quat_to_aa_round_trip_via_nalgebra_exp_holds_at_large_angles() {
    // Sidestep our exp map's truncation by building the quaternion with
    // nalgebra's exact closed-form `from_scaled_axis`, then verify our
    // log map (`quaternion_to_angle_axis`) recovers the original axis to
    // tight tolerance even at near-π rotations.
    for axis in sample_axes() {
        if axis.norm() < 1e-6 {
            continue;
        }
        let q = UnitQuaternion::from_scaled_axis(axis);
        let back = quaternion_to_angle_axis(&q);
        assert!(
            (back - axis).norm() < TOL_LOOSE,
            "log-map round trip failed: axis={:?} back={:?}",
            axis,
            back
        );
    }
}

#[test]
fn quat_to_aa_matches_nalgebra_scaled_axis() {
    // Build the quaternion the *exact* way (nalgebra), then check our log
    // map against nalgebra's. Avoids contaminating the comparison with the
    // 4th-order truncation in our exp map.
    for axis in sample_axes() {
        if axis.norm() < 1e-6 {
            // Near identity nalgebra returns a hard-cut zero (Unit::try_new
            // fails); ours returns a tiny Taylor value. Pinned in its own test.
            continue;
        }
        let q = UnitQuaternion::from_scaled_axis(axis);
        let ours = quaternion_to_angle_axis(&q);
        let theirs = q.scaled_axis();
        assert!(
            (ours - theirs).norm() < TOL_LOOSE,
            "axis={:?}: ours={:?} theirs={:?}",
            axis,
            ours,
            theirs
        );
    }
}

#[test]
fn quat_to_aa_pins_near_identity_branch() {
    // At the identity quaternion, both implementations return 0.
    let id = UnitQuaternion::<f32>::identity();
    assert!(vec_close(
        &quaternion_to_angle_axis(&id),
        &Vector3::zeros(),
        TOL_TIGHT
    ));
    assert!(vec_close(&id.scaled_axis(), &Vector3::zeros(), TOL_TIGHT));

    // For a tiny but nonzero rotation, both should be close to the input
    // angle-axis. Ours uses a 3rd-order series; nalgebra uses the closed
    // form. They should agree to ~1e-7.
    let tiny = Vector3::new(1e-5, 0.0, 0.0);
    let q = UnitQuaternion::from_scaled_axis(tiny);
    let ours = quaternion_to_angle_axis(&q);
    let theirs = q.scaled_axis();
    assert!(
        (ours - theirs).norm() < 1e-7,
        "tiny-angle branch disagrees: ours={:?} theirs={:?}",
        ours,
        theirs
    );
}

#[test]
fn aa_to_quat_matches_nalgebra_for_small_to_moderate_angles() {
    // Where the 4th-order Taylor truncation is well within tolerance.
    for axis in sample_axes() {
        if axis.norm() > 1.0 {
            continue;
        }
        let ours = angle_axis_to_quaternion(&axis);
        let theirs = UnitQuaternion::from_scaled_axis(axis);
        assert!(
            quat_close(&ours, &theirs, TOL),
            "axis={:?} ours={:?} theirs={:?}",
            axis,
            ours.coords.as_slice(),
            theirs.coords.as_slice(),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// angle_axis_to_rotation_matrix
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn aa_to_rmat_matches_rotation3() {
    for axis in sample_axes() {
        let ours = angle_axis_to_rotation_matrix(&axis);
        let theirs = *Rotation3::from_scaled_axis(axis).matrix();
        assert!(
            mat_close(&ours, &theirs, TOL_LOOSE),
            "axis={:?}\nours=\n{}\ntheirs=\n{}",
            axis,
            ours,
            theirs
        );
    }
}

#[test]
fn aa_to_rmat_is_orthogonal_with_unit_determinant() {
    for axis in sample_axes() {
        let r = angle_axis_to_rotation_matrix(&axis);
        let rtr = r.transpose() * r;
        assert!(
            mat_close(&rtr, &Matrix3::identity(), TOL_LOOSE),
            "RᵀR ≠ I for axis={:?}",
            axis
        );
        assert!(
            (r.determinant() - 1.0).abs() < TOL_LOOSE,
            "det(R) ≠ 1 for axis={:?}",
            axis
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// euler_angles_rpy_to_quaternion  ↔  UnitQuaternion::from_euler_angles
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rpy_to_quat_matches_nalgebra() {
    for rpy in sample_rpys() {
        let ours = euler_angles_rpy_to_quaternion(&rpy);
        let theirs = UnitQuaternion::from_euler_angles(rpy.x, rpy.y, rpy.z);
        assert!(
            quat_close(&ours, &theirs, TOL),
            "rpy={:?}\nours={:?}\ntheirs={:?}",
            rpy,
            ours.coords.as_slice(),
            theirs.coords.as_slice(),
        );
    }
}

#[test]
fn rpy_to_quat_known_values() {
    use core::f32::consts::FRAC_PI_2;
    // 90° about x: (cos45°, sin45°, 0, 0)
    let q = euler_angles_rpy_to_quaternion(&Vector3::new(FRAC_PI_2, 0.0, 0.0));
    let expect = UnitQuaternion::new_unchecked(Quaternion::new(
        (FRAC_PI_2 / 2.0).cos(),
        (FRAC_PI_2 / 2.0).sin(),
        0.0,
        0.0,
    ));
    assert!(quat_close(&q, &expect, TOL_TIGHT));

    // Identity
    let q = euler_angles_rpy_to_quaternion(&Vector3::zeros());
    assert!(quat_close(&q, &UnitQuaternion::identity(), TOL_TIGHT));
}

// ═══════════════════════════════════════════════════════════════════════════
// quaternion_to_euler_angles_rpy  ↔  UnitQuaternion::euler_angles
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quat_to_rpy_round_trips() {
    // Build a quat from RPY and see if our extractor recovers it. Skip
    // gimbal lock — pinned separately.
    for rpy in sample_rpys() {
        let q = euler_angles_rpy_to_quaternion(&rpy);
        let back = quaternion_to_euler_angles_rpy(&q);
        assert!(
            vec_close(&back, &rpy, TOL_LOOSE),
            "rpy={:?} back={:?}",
            rpy,
            back
        );
    }
}

#[test]
fn quat_to_rpy_matches_nalgebra_away_from_singularity() {
    for rpy in sample_rpys() {
        let q = UnitQuaternion::from_euler_angles(rpy.x, rpy.y, rpy.z);
        let ours = quaternion_to_euler_angles_rpy(&q);
        let (r, p, y) = q.euler_angles();
        let theirs = Vector3::new(r, p, y);
        assert!(
            vec_close(&ours, &theirs, TOL_LOOSE),
            "rpy={:?} ours={:?} theirs={:?}",
            rpy,
            ours,
            theirs
        );
    }
}

#[test]
fn quat_to_rpy_handles_gimbal_lock_without_nan() {
    use core::f32::consts::FRAC_PI_2;
    // Pitch = +π/2 (nose-up): roll and yaw cease to be independent — only
    // a single linear combination of them is observable. Our extractor's
    // `2·atan2(√(1+s), √(1−s)) - π/2` form is well-defined at s = 1 (it
    // returns π/2 exactly) where a naive `asin` would NaN under FP
    // overshoot. We assert only the things that *are* defined at gimbal
    // lock: finite output and the correct pitch. We do NOT assert
    // round-trip recovery, because our formula collapses the residual
    // (roll, yaw) freedom rather than concentrating it in one slot — a
    // valid choice (matches the C++ source) but it loses the
    // pre-singularity (roll, yaw) split.
    // Precision note: at the singularity the formula reduces to
    // `2·atan2(√(1+s), √(1−s)) − π/2` with `s = ±1`. f32 round-off in
    // `s` (~1e-7) is *amplified* by the `sqrt` to ~3e-4 in the extracted
    // pitch. We allow a 1e-3 envelope here — that's the inherent f32
    // floor for this closed form at the pole, not a defect of the port.
    const POLE_TOL: f32 = 1e-3;

    let q = UnitQuaternion::from_euler_angles(0.3, FRAC_PI_2, 0.5);
    let ours = quaternion_to_euler_angles_rpy(&q);
    assert!(
        ours.x.is_finite() && ours.y.is_finite() && ours.z.is_finite(),
        "non-finite output at gimbal lock: {:?}",
        ours
    );
    assert!((ours.y - FRAC_PI_2).abs() < POLE_TOL);

    // Sanity check the negative pole (pitch = -π/2) too.
    let q_neg = UnitQuaternion::from_euler_angles(0.0, -FRAC_PI_2, 0.0);
    let ours_neg = quaternion_to_euler_angles_rpy(&q_neg);
    assert!(
        ours_neg.x.is_finite() && ours_neg.y.is_finite() && ours_neg.z.is_finite(),
        "non-finite output at -π/2 pole: {:?}",
        ours_neg
    );
    assert!((ours_neg.y + FRAC_PI_2).abs() < POLE_TOL);
}

// ═══════════════════════════════════════════════════════════════════════════
// rotation_matrix_to_euler_angles_rpy  ↔  euler_angles_rpy_to_rotation_matrix
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rpy_rmat_round_trip() {
    for rpy in sample_rpys() {
        let r = euler_angles_rpy_to_rotation_matrix(&rpy);
        let back = rotation_matrix_to_euler_angles_rpy(&r);
        assert!(
            vec_close(&back, &rpy, TOL_LOOSE),
            "rpy={:?} back={:?}",
            rpy,
            back
        );
    }
}

#[test]
fn rpy_to_rmat_matches_rotation3() {
    for rpy in sample_rpys() {
        let ours = euler_angles_rpy_to_rotation_matrix(&rpy);
        let theirs = *Rotation3::from_euler_angles(rpy.x, rpy.y, rpy.z).matrix();
        assert!(
            mat_close(&ours, &theirs, TOL_LOOSE),
            "rpy={:?}\nours=\n{}\ntheirs=\n{}",
            rpy,
            ours,
            theirs
        );
    }
}

#[test]
fn rmat_to_rpy_matches_rotation3_away_from_singularity() {
    for rpy in sample_rpys() {
        let r = Rotation3::from_euler_angles(rpy.x, rpy.y, rpy.z);
        let ours = rotation_matrix_to_euler_angles_rpy(r.matrix());
        let (rr, pp, yy) = r.euler_angles();
        let theirs = Vector3::new(rr, pp, yy);
        assert!(
            vec_close(&ours, &theirs, TOL_LOOSE),
            "rpy={:?} ours={:?} theirs={:?}",
            rpy,
            ours,
            theirs
        );
    }
}

#[test]
fn rpy_to_rmat_columns_are_body_axes() {
    // R = [x_b, y_b, z_b] in world. For yaw-only rotation, x_b should
    // lie in the world XY plane.
    use core::f32::consts::FRAC_PI_4;
    let r = euler_angles_rpy_to_rotation_matrix(&Vector3::new(0.0, 0.0, FRAC_PI_4));
    let x_b = r.column(0);
    assert!((x_b.z).abs() < TOL_TIGHT);
    assert!((x_b.x - FRAC_PI_4.cos()).abs() < TOL);
    assert!((x_b.y - FRAC_PI_4.sin()).abs() < TOL);
}

// ═══════════════════════════════════════════════════════════════════════════
// quaternion_to_yaw
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn yaw_extractor_matches_pure_yaw_rotations() {
    use core::f32::consts::PI;
    for &y in &[-PI + 0.1, -1.0, -0.3, 0.0, 0.3, 1.0, PI - 0.1] {
        let q = UnitQuaternion::from_euler_angles(0.0, 0.0, y);
        let extracted = quaternion_to_yaw(&q, 0.0);
        assert!(
            (extracted - y).abs() < TOL,
            "yaw mismatch: in={} out={}",
            y,
            extracted
        );
    }
}

#[test]
fn yaw_extractor_ignores_small_roll_pitch() {
    // Adding modest roll/pitch should not perturb the extracted yaw much
    // — the projection of x_b onto the world XY plane is what we measure.
    let y = 0.7_f32;
    let q = UnitQuaternion::from_euler_angles(0.2, 0.15, y);
    let extracted = quaternion_to_yaw(&q, 0.0);
    // Geometric definition of "yaw" used here doesn't equal the Euler-yaw
    // when there's nontrivial roll/pitch — but it should be close for
    // small tilts. Just sanity-check finite + bounded.
    assert!(extracted.is_finite());
    assert!((extracted - y).abs() < 0.2);
}

#[test]
fn yaw_extractor_returns_default_when_x_b_is_vertical() {
    // Pitch = +π/2 puts the body x-axis along world -z. The projection
    // onto XY shrinks to zero, so the extractor must fall back to the
    // supplied default.
    use core::f32::consts::FRAC_PI_2;
    let q = UnitQuaternion::from_euler_angles(0.0, FRAC_PI_2, 0.0);
    let default_yaw: f32 = 1.234;
    let extracted = quaternion_to_yaw(&q, default_yaw);
    assert!((extracted - default_yaw).abs() < TOL_TIGHT);
}

// ═══════════════════════════════════════════════════════════════════════════
// quaternion_from_unit_z_to_v
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn unit_z_to_v_actually_maps_z_to_v() {
    for v in [
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.1, 0.2, 0.97).normalize(),
        Vector3::new(-0.3, 0.5, 0.8).normalize(),
        Vector3::new(0.6, -0.4, 0.7).normalize(),
    ] {
        let q = quaternion_from_unit_z_to_v(&v);
        let mapped = q * Vector3::z();
        assert!(
            vec_close(&mapped, &v, TOL_LOOSE),
            "q·ẑ ≠ v: v={:?} mapped={:?}",
            v,
            mapped
        );
        // Should also be unit-norm.
        assert!((q.norm() - 1.0).abs() < TOL);
    }
}

#[test]
fn unit_z_to_v_zero_yaw() {
    // The result of unit_z_to_v has, by construction, qz = 0 (no yaw
    // about world z). For a tilt that already lies in the XZ plane, the
    // resulting rotation is purely about world y.
    let v = Vector3::new(0.5_f32.sin(), 0.0, 0.5_f32.cos());
    let q = quaternion_from_unit_z_to_v(&v);
    assert!(q.k.abs() < TOL_TIGHT, "expected qz=0, got {}", q.k);
}

// ═══════════════════════════════════════════════════════════════════════════
// quaternion_from_zb_and_yaw  (both branches)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zb_and_yaw_tilt_branch_realizes_zb_and_yaw() {
    use core::f32::consts::FRAC_PI_4;
    let v = Vector3::new(0.2, -0.3, 0.93).normalize();
    let yaw = FRAC_PI_4;
    let q = quaternion_from_zb_and_yaw(&v, yaw, true);

    // Body z-axis = q · ẑ should equal v.
    let zb = q * Vector3::z();
    assert!(vec_close(&zb, &v, TOL_LOOSE), "zb={:?} v={:?}", zb, v);
    // Yaw, as defined by quaternion_to_yaw, should round-trip.
    let extracted = quaternion_to_yaw(&q, 0.0);
    assert!(
        (extracted - yaw).abs() < 0.05,
        "yaw mismatch: in={} out={}",
        yaw,
        extracted
    );
}

#[test]
fn zb_and_yaw_cross_branch_realizes_zb_and_yaw() {
    use core::f32::consts::FRAC_PI_3;
    let v = Vector3::new(-0.4, 0.2, 0.89).normalize();
    let yaw = FRAC_PI_3;
    let q = quaternion_from_zb_and_yaw(&v, yaw, false);

    let zb = q * Vector3::z();
    assert!(vec_close(&zb, &v, TOL_LOOSE), "zb={:?} v={:?}", zb, v);

    // Body x-axis should be perpendicular to v and lie on the side
    // selected by yaw (i.e. y_c × v normalized, where y_c is the
    // yaw-rotated world ŷ).
    let y_c = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw) * Vector3::y();
    let expected_xb = y_c.cross(&v).normalize();
    let xb = q * Vector3::x();
    assert!(
        vec_close(&xb, &expected_xb, TOL_LOOSE),
        "xb={:?} expected={:?}",
        xb,
        expected_xb
    );
}

#[test]
fn zb_and_yaw_branches_agree_on_body_z() {
    // The two branches use *different* yaw conventions internally (the
    // tilt branch yaws about the body z after tilting; the cross-product
    // branch picks the body x as `y_c × z_b` where `y_c` is the yawed
    // world ŷ). The resulting frames therefore generally differ by an
    // about-z rotation. The one invariant they must share is the body z
    // axis, which is the input `v`. Verify that and nothing more.
    use core::f32::consts::FRAC_PI_6;
    for v in [
        Vector3::new(0.1, 0.1, 0.99).normalize(),
        Vector3::new(-0.2, 0.3, 0.93).normalize(),
        Vector3::new(0.4, -0.3, 0.87).normalize(),
    ] {
        let yaw = FRAC_PI_6;
        let q_tilt = quaternion_from_zb_and_yaw(&v, yaw, true);
        let q_cross = quaternion_from_zb_and_yaw(&v, yaw, false);
        let zb_tilt = q_tilt * Vector3::z();
        let zb_cross = q_cross * Vector3::z();
        assert!(vec_close(&zb_tilt, &v, TOL_LOOSE));
        assert!(vec_close(&zb_cross, &v, TOL_LOOSE));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// angle_axis_from_two_normals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn aa_from_two_normals_rotates_a_to_b() {
    for (a, b) in [
        (Vector3::z(), Vector3::new(0.2, -0.3, 0.93).normalize()),
        (Vector3::x(), Vector3::new(0.5, 0.6, 0.6244).normalize()),
        (Vector3::y(), Vector3::new(-0.4, 0.6, 0.6928).normalize()),
    ] {
        let perp = Vector3::x(); // unused away from antiparallel case
        let rvec = angle_axis_from_two_normals(&a, &b, &perp);
        let r = angle_axis_to_rotation_matrix(&rvec);
        let mapped = r * a;
        assert!(
            vec_close(&mapped, &b, TOL_LOOSE),
            "R·a ≠ b: a={:?} b={:?} mapped={:?}",
            a,
            b,
            mapped
        );
    }
}

#[test]
fn aa_from_two_normals_handles_parallel_inputs() {
    // a == b: rotation vector should be (essentially) zero.
    let a: Vector3<f32> = Vector3::new(0.3, -0.4, 0.5).normalize();
    let perp: Vector3<f32> = Vector3::x();
    let rvec = angle_axis_from_two_normals(&a, &a, &perp);
    assert!(rvec.norm() < TOL_LOOSE, "expected ~0, got {:?}", rvec);
}

#[test]
fn aa_from_two_normals_handles_antiparallel_inputs() {
    use core::f32::consts::PI;
    // a == -b: there's no unique axis; the function returns π · a_perp.
    // Use the supplied perpendicular as the axis.
    let a: Vector3<f32> = Vector3::z();
    let b: Vector3<f32> = -a;
    let perp: Vector3<f32> = Vector3::x();
    let rvec = angle_axis_from_two_normals(&a, &b, &perp);
    assert!((rvec.norm() - PI).abs() < TOL_LOOSE);
    // And rotating by it must actually flip a → b.
    let r = angle_axis_to_rotation_matrix(&rvec);
    assert!(vec_close(&(r * a), &b, TOL_LOOSE));
}

// ═══════════════════════════════════════════════════════════════════════════
// should_flip_quaternion
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn should_flip_detects_opposite_hemisphere() {
    let q = UnitQuaternion::from_euler_angles(0.3, -0.2, 0.5);
    let q_flipped = UnitQuaternion::new_unchecked(Quaternion::from(-q.coords));

    // q vs q itself: no flip needed.
    assert!(!should_flip_quaternion(&q, &q));
    // q vs -q: flip needed.
    assert!(should_flip_quaternion(&q, &q_flipped));
    // q vs a small perturbation: no flip.
    let q_near = UnitQuaternion::from_euler_angles(0.31, -0.21, 0.51);
    assert!(!should_flip_quaternion(&q, &q_near));
    // q vs (-q with small perturbation): flip.
    let q_near_neg = UnitQuaternion::new_unchecked(Quaternion::from(-q_near.coords));
    assert!(should_flip_quaternion(&q, &q_near_neg));
}
