//! Equivalence test for current dense ESKF math vs. (future) optimized
//! block-sparse implementation.
//!
//! # Why this exists
//!
//! The plan to optimize the ESKF (block-sparse F·P·Fᵀ in `propagate_state`,
//! block-sparse Joseph form in `update_*`) is meant to preserve observable
//! behavior to within float32 round-off. This test pins down the current
//! reference numerically using simulated IMU+pose data, so that any future
//! optimization can be validated bit-for-bit-equivalent (within tolerance)
//! against the snapshot it produces today.
//!
//! # How it works
//!
//! 1. A deterministic synthetic trajectory generator produces a sequence
//!    of `(accel, gyro, dt)` IMU tuples and intermittent `(pos, q)` pose
//!    measurements covering several flight regimes:
//!      - stationary hover (no motion, biases visible),
//!      - sustained climb (sign-correlated z-acceleration),
//!      - aggressive yaw (high gyro rates),
//!      - mixed maneuver (rotating velocity vector + climb),
//!      - extreme tilt (large attitude error → tests boxplus/renormalize).
//! 2. Two `Eskf` instances are built from identical config and initial
//!    state, then driven through identical command streams.
//! 3. After every operation, all 15 state components and all 225
//!    covariance entries are compared.
//!
//! Today both filters call the same dense paths, so the comparison is
//! trivially exact. When Section 1/2 of the optimization plan lands —
//! introducing a block-sparse path behind a feature flag or an alternate
//! method — this test should be modified to drive the *optimized* filter
//! through the new path, and the assertions will catch any algebra error.

use super::eskf::{Eskf, EskfConfig, UpdateOutcome};
use nalgebra::{SMatrix, UnitQuaternion, Vector3};

// ─── tolerances ─────────────────────────────────────────────────────────────

/// Per-component tolerance for state vector entries.
///
/// Float32 round-off accumulated over thousands of FMAs in the predict
/// and update paths bounds at roughly N·eps·magnitude ≈ 2000·1.2e-7·1.0
/// ≈ 2.4e-4. We allow 1e-4 absolute, which is conservative against the
/// theoretical bound and ~100× tighter than any threshold that would
/// represent an actual behavior change in the filter.
const TOL_STATE: f32 = 1e-4;

/// Per-entry tolerance for the 15×15 covariance matrix.
const TOL_COV: f32 = 1e-4;

// ─── deterministic PRNG (no `rand` dep, keeps `cybflight-core` lean) ───────

#[derive(Clone, Copy)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn gaussian(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f32 / ((1u64 << 53) as f32)).max(1e-9);
        let u2 = (self.next_u64() >> 11) as f32 / ((1u64 << 53) as f32);
        (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
    }
    fn vec3_gauss(&mut self, sigma: f32) -> Vector3<f32> {
        Vector3::new(self.gaussian(), self.gaussian(), self.gaussian()) * sigma
    }
}

// ─── synthetic command stream ──────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Op {
    Predict {
        accel: Vector3<f32>,
        gyro: Vector3<f32>,
        dt: f32,
    },
    UpdatePose {
        pos: Vector3<f32>,
        q: UnitQuaternion<f32>,
        pos_std: f32,
        att_std: f32,
    },
    UpdatePos {
        pos: Vector3<f32>,
        std: f32,
    },
    UpdateAtt {
        q: UnitQuaternion<f32>,
        std: f32,
    },
    UpdateVel {
        vel: Vector3<f32>,
        std: f32,
    },
    UpdateAltitude {
        alt: f32,
    },
    UpdateMag {
        mag_body: Vector3<f32>,
        mag_world: Vector3<f32>,
    },
}

fn build_command_stream() -> Vec<Op> {
    let mut rng = Rng::new(0xC0FFEE_BADC0FFE);
    let mut ops: Vec<Op> = Vec::with_capacity(2_000);

    let dt = 1e-3_f32;
    let g = Vector3::new(0.0, 0.0, -9.81);

    let mut q_true = UnitQuaternion::identity();
    let mut v_true = Vector3::zeros();
    let mut p_true = Vector3::zeros();

    // Phase 1: stationary, biased IMU.
    let accel_bias_true = Vector3::new(0.02, -0.03, 0.05);
    let gyro_bias_true = Vector3::new(0.001, -0.0015, 0.002);
    for i in 0..200 {
        let accel_meas = -g + accel_bias_true + rng.vec3_gauss(0.005);
        let gyro_meas = gyro_bias_true + rng.vec3_gauss(0.0005);
        ops.push(Op::Predict {
            accel: accel_meas,
            gyro: gyro_meas,
            dt,
        });
        if i % 10 == 9 {
            ops.push(Op::UpdatePose {
                pos: p_true + rng.vec3_gauss(0.002),
                q: q_true,
                pos_std: 0.01,
                att_std: 0.03,
            });
        }
    }

    // Phase 2: sustained climb.
    let climb_accel_world = Vector3::new(0.0, 0.0, 4.0);
    for i in 0..400 {
        let accel_meas = climb_accel_world - g + accel_bias_true + rng.vec3_gauss(0.005);
        let gyro_meas = gyro_bias_true + rng.vec3_gauss(0.0005);
        ops.push(Op::Predict {
            accel: accel_meas,
            gyro: gyro_meas,
            dt,
        });
        v_true += climb_accel_world * dt;
        p_true += v_true * dt;
        if i % 10 == 9 {
            ops.push(Op::UpdatePose {
                pos: p_true + rng.vec3_gauss(0.002),
                q: q_true,
                pos_std: 0.01,
                att_std: 0.03,
            });
        }
    }

    // Phase 3: aggressive yaw.
    let yaw_rate = 8.0;
    for i in 0..300 {
        let gyro_world = Vector3::new(0.0, 0.0, yaw_rate);
        let gyro_meas = gyro_world + gyro_bias_true + rng.vec3_gauss(0.001);
        let accel_meas = -g + accel_bias_true + rng.vec3_gauss(0.005);
        ops.push(Op::Predict {
            accel: accel_meas,
            gyro: gyro_meas,
            dt,
        });
        q_true *= UnitQuaternion::from_scaled_axis(gyro_world * dt);
        if i % 12 == 11 {
            ops.push(Op::UpdatePose {
                pos: p_true + rng.vec3_gauss(0.003),
                q: q_true,
                pos_std: 0.02,
                att_std: 0.05,
            });
        }
    }

    // Phase 4: tilted, moving.
    let tilt = UnitQuaternion::from_scaled_axis(Vector3::new(0.4, -0.3, 0.0));
    q_true = tilt * q_true;
    for i in 0..400 {
        let r_t = q_true.to_rotation_matrix().transpose();
        let a_world = Vector3::new(2.0 * (i as f32 * dt).sin(), 1.5, 0.5);
        let accel_meas = r_t * (a_world - g) + accel_bias_true + rng.vec3_gauss(0.01);
        let gyro_meas = Vector3::new(0.3, 0.2, 0.5) + gyro_bias_true + rng.vec3_gauss(0.001);
        ops.push(Op::Predict {
            accel: accel_meas,
            gyro: gyro_meas,
            dt,
        });
        v_true += a_world * dt;
        p_true += v_true * dt;
        q_true *= UnitQuaternion::from_scaled_axis(Vector3::new(0.3, 0.2, 0.5) * dt);
        if i % 8 == 7 {
            ops.push(Op::UpdatePose {
                pos: p_true + rng.vec3_gauss(0.005),
                q: q_true,
                pos_std: 0.03,
                att_std: 0.05,
            });
        }
    }

    // Phase 5: separate update_pos / update_att / update_vel /
    // update_altitude / update_mag, mixed in.
    for i in 0..200 {
        let accel_meas = -g + accel_bias_true + rng.vec3_gauss(0.003);
        let gyro_meas = rng.vec3_gauss(0.0005);
        ops.push(Op::Predict {
            accel: accel_meas,
            gyro: gyro_meas,
            dt,
        });

        match i % 5 {
            0 => ops.push(Op::UpdatePos {
                pos: p_true + rng.vec3_gauss(0.005),
                std: 0.02,
            }),
            1 => ops.push(Op::UpdateAtt {
                q: q_true,
                std: 0.04,
            }),
            2 => ops.push(Op::UpdateVel {
                vel: v_true + rng.vec3_gauss(0.02),
                std: 0.05,
            }),
            3 => ops.push(Op::UpdateAltitude {
                alt: p_true.z + rng.gaussian() * 0.1,
            }),
            _ => {
                let mag_world = Vector3::new(20.0, 5.0, -42.0);
                let mag_body = q_true.to_rotation_matrix().transpose() * mag_world
                    + rng.vec3_gauss(0.05);
                ops.push(Op::UpdateMag {
                    mag_body,
                    mag_world,
                });
            }
        }
    }

    // Phase 6: borderline-large innovation (exercises inflation path
    // of every update method without crossing the hard JumpRejected gate).
    for i in 0..100 {
        let accel_meas = -g + accel_bias_true + rng.vec3_gauss(0.003);
        ops.push(Op::Predict {
            accel: accel_meas,
            gyro: rng.vec3_gauss(0.0005),
            dt,
        });
        if i % 5 == 4 {
            ops.push(Op::UpdatePose {
                pos: p_true + Vector3::new(0.20, 0.0, 0.0),
                q: q_true,
                pos_std: 0.01,
                att_std: 0.03,
            });
        }
    }

    ops
}

/// Drive a filter using the dense (reference) implementations.
fn step(eskf: &mut Eskf, op: &Op) -> Option<UpdateOutcome> {
    match *op {
        Op::Predict { accel, gyro, dt } => {
            eskf.predict(accel, gyro, dt);
            None
        }
        Op::UpdatePose {
            pos,
            q,
            pos_std,
            att_std,
        } => Some(eskf.update_pose(pos, q, pos_std, att_std)),
        Op::UpdatePos { pos, std } => Some(eskf.update_pos(pos, std)),
        Op::UpdateAtt { q, std } => Some(eskf.update_att(q, std)),
        Op::UpdateVel { vel, std } => Some(eskf.update_vel(vel, std)),
        Op::UpdateAltitude { alt } => Some(eskf.update_altitude(alt)),
        Op::UpdateMag {
            mag_body,
            mag_world,
        } => Some(eskf.update_mag(mag_body, mag_world)),
    }
}

/// Drive a filter using the optimized (sparse) implementations where
/// available. Used as the `candidate` in equivalence tests so any algebra
/// drift is caught against the dense reference.
fn step_sparse(eskf: &mut Eskf, op: &Op) -> Option<UpdateOutcome> {
    match *op {
        Op::Predict { accel, gyro, dt } => {
            eskf.predict_sparse(accel, gyro, dt);
            None
        }
        Op::UpdatePose {
            pos,
            q,
            pos_std,
            att_std,
        } => Some(eskf.update_pose_sparse(pos, q, pos_std, att_std)),
        Op::UpdatePos { pos, std } => Some(eskf.update_pos(pos, std)),
        Op::UpdateAtt { q, std } => Some(eskf.update_att(q, std)),
        Op::UpdateVel { vel, std } => Some(eskf.update_vel(vel, std)),
        Op::UpdateAltitude { alt } => Some(eskf.update_altitude(alt)),
        Op::UpdateMag {
            mag_body,
            mag_world,
        } => Some(eskf.update_mag(mag_body, mag_world)),
    }
}

fn assert_filters_equal(reference: &Eskf, candidate: &Eskf, op_idx: usize, label: &str) {
    for i in 0..3 {
        let r = reference.position()[i];
        let c = candidate.position()[i];
        let diff = (r - c).abs();
        assert!(
            diff <= TOL_STATE,
            "{label}: position[{i}] differs at op {op_idx}: ref={r:e} cand={c:e} diff={diff:e}"
        );
    }
    for i in 0..3 {
        let r = reference.velocity()[i];
        let c = candidate.velocity()[i];
        let diff = (r - c).abs();
        assert!(
            diff <= TOL_STATE,
            "{label}: velocity[{i}] differs at op {op_idx}: ref={r:e} cand={c:e} diff={diff:e}"
        );
    }
    let q_r = reference.orientation();
    let q_c = candidate.orientation();
    let direct = (q_r.coords - q_c.coords).norm();
    let flipped = (q_r.coords + q_c.coords).norm();
    let diff = direct.min(flipped);
    assert!(
        diff <= TOL_STATE,
        "{label}: orientation differs at op {op_idx}: diff={diff:e}"
    );
    for i in 0..3 {
        let r = reference.gyro_bias()[i];
        let c = candidate.gyro_bias()[i];
        let diff = (r - c).abs();
        assert!(
            diff <= TOL_STATE,
            "{label}: gyro_bias[{i}] differs at op {op_idx}: ref={r:e} cand={c:e} diff={diff:e}"
        );
    }
    for i in 0..3 {
        let r = reference.accel_bias()[i];
        let c = candidate.accel_bias()[i];
        let diff = (r - c).abs();
        assert!(
            diff <= TOL_STATE,
            "{label}: accel_bias[{i}] differs at op {op_idx}: ref={r:e} cand={c:e} diff={diff:e}"
        );
    }
    let cov_r = reference.covariance();
    let cov_c = candidate.covariance();
    for i in 0..15 {
        for j in 0..15 {
            let r = cov_r[(i, j)];
            let c = cov_c[(i, j)];
            let diff = (r - c).abs();
            let scale = r.abs().max(c.abs()).max(1.0);
            let bound = TOL_COV * scale;
            assert!(
                diff <= bound,
                "{label}: cov[{i},{j}] differs at op {op_idx}: ref={r:e} cand={c:e} diff={diff:e} bound={bound:e}"
            );
        }
    }
}

fn fresh_pair() -> (Eskf, Eskf) {
    let mut a = Eskf::new(EskfConfig::default());
    let mut b = Eskf::new(EskfConfig::default());
    let p0 = Vector3::new(0.0, 0.0, 0.0);
    let q0 = UnitQuaternion::identity();
    a.init(p0, q0, Vector3::zeros(), Vector3::zeros());
    b.init(p0, q0, Vector3::zeros(), Vector3::zeros());
    (a, b)
}

#[test]
fn dense_vs_dense_step_by_step() {
    let ops = build_command_stream();
    let (mut reference, mut candidate) = fresh_pair();

    let mut accepted_inflated = 0usize;
    let mut accepted_normal = 0usize;
    let mut rejected = 0usize;

    for (idx, op) in ops.iter().enumerate() {
        let r_outcome = step(&mut reference, op);
        let c_outcome = step(&mut candidate, op);

        match (r_outcome, c_outcome) {
            (Some(r), Some(c)) => assert_eq!(
                r, c,
                "outcome mismatch at op {idx}: ref={r:?} cand={c:?}"
            ),
            (None, None) => {}
            _ => panic!("op-type mismatch at op {idx}"),
        }

        if let Some(o) = r_outcome {
            match o {
                UpdateOutcome::Accepted { inflated: true } => accepted_inflated += 1,
                UpdateOutcome::Accepted { inflated: false } => accepted_normal += 1,
                _ => rejected += 1,
            }
        }

        assert_filters_equal(&reference, &candidate, idx, "step-by-step");

        assert!(
            reference.is_initialized() && candidate.is_initialized(),
            "filter de-initialized at op {idx} — synthetic data drove it pathological"
        );
    }

    assert!(
        accepted_normal > 100,
        "expected many normal-accept updates, got {accepted_normal}"
    );
    assert!(
        accepted_inflated > 0,
        "expected at least one inflated update; phase 6 should produce them"
    );
    assert!(
        rejected < accepted_normal / 2,
        "unexpectedly high rejection rate: {rejected} / {accepted_normal} accepted"
    );
}

/// Drive the candidate through the sparse path (`predict_sparse`) and
/// the reference through the dense path. Asserts equivalence after every
/// op. This is the regression oracle for the optimization: any algebra
/// error in the sparse code surfaces here at the first divergent block.
#[test]
fn dense_vs_sparse_step_by_step() {
    let ops = build_command_stream();
    let (mut reference, mut candidate) = fresh_pair();

    for (idx, op) in ops.iter().enumerate() {
        let r_outcome = step(&mut reference, op);
        let c_outcome = step_sparse(&mut candidate, op);
        match (r_outcome, c_outcome) {
            (Some(r), Some(c)) => assert_eq!(
                r, c,
                "sparse outcome mismatch at op {idx}: ref={r:?} cand={c:?}"
            ),
            (None, None) => {}
            _ => panic!("sparse op-type mismatch at op {idx}"),
        }
        assert_filters_equal(&reference, &candidate, idx, "dense-vs-sparse");
        assert!(
            reference.is_initialized() && candidate.is_initialized(),
            "filter de-initialized at op {idx} during sparse equivalence test"
        );
    }
}

/// Sparse predict only — no measurement updates. Isolates the F·P·Fᵀ
/// computation so any algebra error in `propagate_state_sparse` is
/// reported without entanglement with the update path.
#[test]
fn dense_vs_sparse_predict_only() {
    let mut rng = Rng::new(0xDECAFBAD_DEADBEEF);
    let (mut reference, mut candidate) = fresh_pair();
    let dt = 1e-3_f32;
    let g = Vector3::new(0.0, 0.0, -9.81);

    for i in 0..3000 {
        let accel = -g
            + Vector3::new(
                (i as f32 * dt).sin() * 0.5,
                (i as f32 * dt * 0.3).cos() * 0.4,
                (i as f32 * dt * 0.7).sin() * 0.2,
            )
            + rng.vec3_gauss(0.01);
        let gyro = Vector3::new(
            0.5 * (i as f32 * dt).sin(),
            0.3 * (i as f32 * dt * 0.5).cos(),
            0.8 * (i as f32 * dt * 0.2).sin(),
        ) + rng.vec3_gauss(0.001);

        reference.predict(accel, gyro, dt);
        candidate.predict_sparse(accel, gyro, dt);
        if i % 50 == 49 {
            assert_filters_equal(&reference, &candidate, i, "sparse-predict-only");
        }
    }
    assert_filters_equal(&reference, &candidate, 2999, "sparse-predict-only-final");
}

#[test]
fn dense_vs_dense_predict_only() {
    let mut rng = Rng::new(0xDECAFBAD_DEADBEEF);
    let (mut reference, mut candidate) = fresh_pair();

    let dt = 1e-3_f32;
    let g = Vector3::new(0.0, 0.0, -9.81);

    for i in 0..3000 {
        let accel = -g
            + Vector3::new(
                (i as f32 * dt).sin() * 0.5,
                (i as f32 * dt * 0.3).cos() * 0.4,
                (i as f32 * dt * 0.7).sin() * 0.2,
            )
            + rng.vec3_gauss(0.01);
        let gyro = Vector3::new(
            0.5 * (i as f32 * dt).sin(),
            0.3 * (i as f32 * dt * 0.5).cos(),
            0.8 * (i as f32 * dt * 0.2).sin(),
        ) + rng.vec3_gauss(0.001);

        reference.predict(accel, gyro, dt);
        candidate.predict(accel, gyro, dt);
        if i % 50 == 49 {
            assert_filters_equal(&reference, &candidate, i, "predict-only");
        }
    }
    assert_filters_equal(&reference, &candidate, 2999, "predict-only-final");
}

#[test]
fn dense_vs_dense_updates_only() {
    let mut rng = Rng::new(0xFEED_FACE_CAFE_BABE);
    let (mut reference, mut candidate) = fresh_pair();

    let q_true = UnitQuaternion::identity();
    let p_true = Vector3::zeros();
    for i in 0..300 {
        let pos = p_true + rng.vec3_gauss(0.005);
        let q = q_true;
        let r_o = reference.update_pose(pos, q, 0.02, 0.04);
        let c_o = candidate.update_pose(pos, q, 0.02, 0.04);
        assert_eq!(r_o, c_o, "outcome mismatch at update {i}");
        assert_filters_equal(&reference, &candidate, i, "updates-only");
    }
}

#[test]
fn stream_is_deterministic() {
    let a = build_command_stream();
    let b = build_command_stream();
    assert_eq!(a.len(), b.len());
    for (i, (oa, ob)) in a.iter().zip(b.iter()).enumerate() {
        if let (
            Op::Predict {
                accel: a1,
                gyro: g1,
                dt: d1,
            },
            Op::Predict {
                accel: a2,
                gyro: g2,
                dt: d2,
            },
        ) = (oa, ob)
        {
            assert_eq!(a1, a2, "predict.accel diverged at op {i}");
            assert_eq!(g1, g2, "predict.gyro diverged at op {i}");
            assert_eq!(d1, d2, "predict.dt diverged at op {i}");
        }
    }
}

#[test]
fn final_state_is_sane() {
    let ops = build_command_stream();
    let (mut eskf, _) = fresh_pair();

    for op in &ops {
        step(&mut eskf, op);
    }

    let pos = eskf.position();
    let vel = eskf.velocity();
    let q = eskf.orientation();
    let gb = eskf.gyro_bias();
    let ab = eskf.accel_bias();
    let cov = eskf.covariance();

    assert!(pos.iter().all(|v| v.is_finite()), "non-finite final position");
    assert!(vel.iter().all(|v| v.is_finite()), "non-finite final velocity");
    assert!(
        q.coords.iter().all(|v| v.is_finite()),
        "non-finite final orientation"
    );
    assert!(gb.iter().all(|v| v.is_finite()), "non-finite final gyro_bias");
    assert!(ab.iter().all(|v| v.is_finite()), "non-finite final accel_bias");
    let cov_trace: f32 = (0..15).map(|i| cov[(i, i)]).sum();
    assert!(cov_trace.is_finite(), "non-finite covariance trace");
    assert!(cov_trace > 0.0, "covariance trace collapsed to zero");

    let gyro_bias_cov_trace = cov[(12, 12)] + cov[(13, 13)] + cov[(14, 14)];
    assert!(
        gyro_bias_cov_trace < 0.01,
        "gyro-bias cov did not converge: trace={gyro_bias_cov_trace:e}"
    );

    let mut max_asym = 0.0_f32;
    for i in 0..15 {
        for j in (i + 1)..15 {
            let asym = (cov[(i, j)] - cov[(j, i)]).abs();
            if asym > max_asym {
                max_asym = asym;
            }
        }
    }
    assert!(
        max_asym < 1e-6,
        "covariance lost symmetry: max |P-Pᵀ| = {max_asym:e}"
    );

    for i in 0..15 {
        assert!(
            cov[(i, i)] >= 1e-10,
            "P diagonal entry {i} below floor: {}",
            cov[(i, i)]
        );
    }
}

#[test]
fn covariance_self_equivalence() {
    let ops = build_command_stream();
    let (mut a, mut b) = fresh_pair();
    for op in &ops {
        step(&mut a, op);
        step(&mut b, op);
    }
    let ca = a.covariance();
    let cb = b.covariance();
    let diff: SMatrix<f32, 15, 15> = ca - cb;
    let max_abs = diff.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(max_abs < 1e-9, "self-equivalence broken: max |ΔP| = {max_abs}");
}
