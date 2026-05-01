//! MINCO min-snap trajectory solver (polynomial degree 7, s=4).
//!
//! Direct Rust port of `drolib::MincoSnap` from
//! `tmp/planner/include/drolib/solver/minco_snap.hpp`. Same banded
//! storage layout, same `set_boundary` / `solve` / `get_trajectory` /
//! `get_energy` / `add_energy_grad_by_*` / `propagate_grad` API as
//! [`super::minco_jerk::MincoJerk`], so callers can swap solvers with
//! no plumbing changes.
//!
//! For an n-piece trajectory the linear system is 8n × 8n with
//! half-bandwidth 8. Boundary conditions are PVAJ (position, velocity,
//! acceleration, jerk) at head and tail; intermediate constraints
//! enforce position match at the junction plus continuity through
//! derivative 6 (velocity, acceleration, jerk, snap, crackle, pop).
//!
//! See [`super::minco_jerk::MincoJerk`] for the s=3 variant used by the
//! BFGS planner; MincoSnap is heavier per solve but produces smoother
//! body-rate / body-acceleration profiles, which is desirable when the
//! offline schedule will be tracked open-loop.
//!
//! ## Stack-size warning
//!
//! `MincoSnap` carries ~42 KB of inline storage (banded buffer +
//! coefficient rows + seven time tables sized at [`MAX_PIECES`]).
//! `MincoSnap::new` returns by value with no NRVO guarantee, so a
//! stack-allocated `let mut s = MincoSnap::new(...)` may briefly
//! materialize the full struct on the caller's stack. On Embassy
//! task stacks (typically 4–16 KB) this overflows; firmware callers
//! must hold the solver in a `StaticCell` (BSS-resident), exactly as
//! `OFFLINE_MINCO` does in `cybflight::control::mission_planner`.
//! Host-side use (tests, sim, planning utilities) is unaffected.

use nalgebra::{UnitQuaternion, Vector3};

use super::banded_system::BandedSystem;
use super::piecewise_polynomial::PiecewisePolynomial;
use super::polynomial::Polynomial;
use super::types::{Vec3, ZERO3, PVAJ3D};
use super::MAX_PIECES;
use crate::rotation::quaternion_from_zb_and_yaw;

/// MINCO min-snap solver (polynomial degree 7, smoothness s=4).
///
/// Zero heap allocations — all buffers are inline fixed-size arrays
/// sized at [`MAX_PIECES`]. The active piece count is tracked in
/// `self.n`; only the leading `8·n` entries of every buffer are live
/// per solve.
pub struct MincoSnap {
    n: usize,
    head_pvaj: PVAJ3D,
    tail_pvaj: PVAJ3D,
    banded: BandedSystem,
    /// Coefficient matrix: 8N rows × 3 cols (one Vec3 per row).
    b: [Vector3<f32>; 8 * MAX_PIECES],
    t1: [f32; MAX_PIECES],
    t2: [f32; MAX_PIECES],
    t3: [f32; MAX_PIECES],
    t4: [f32; MAX_PIECES],
    t5: [f32; MAX_PIECES],
    t6: [f32; MAX_PIECES],
    t7: [f32; MAX_PIECES],
}

impl MincoSnap {
    /// Initialize the solver with placeholder boundary states. Reuse
    /// across solves with [`set_boundary`] — constructing a fresh
    /// `MincoSnap` zero-initializes ~35 KB of inline storage (the
    /// banded buffer alone is `8·MAX_PIECES·17` floats), so holding
    /// one in a `StaticCell` is significantly cheaper than building a
    /// new one per call.
    pub fn new(head_state: &PVAJ3D, tail_state: &PVAJ3D, piece_num: usize) -> Self {
        debug_assert!(piece_num >= 1 && piece_num <= MAX_PIECES);
        Self {
            n: piece_num,
            head_pvaj: *head_state,
            tail_pvaj: *tail_state,
            banded: BandedSystem::new(8 * piece_num, 8, 8),
            b: [Vector3::zeros(); 8 * MAX_PIECES],
            t1: [0.0; MAX_PIECES],
            t2: [0.0; MAX_PIECES],
            t3: [0.0; MAX_PIECES],
            t4: [0.0; MAX_PIECES],
            t5: [0.0; MAX_PIECES],
            t6: [0.0; MAX_PIECES],
            t7: [0.0; MAX_PIECES],
        }
    }

    /// Update the head/tail PVAJ in place. Mirror of
    /// [`super::minco_jerk::MincoJerk::set_boundary`].
    pub fn set_boundary(&mut self, head_state: &PVAJ3D, tail_state: &PVAJ3D) {
        self.head_pvaj = *head_state;
        self.tail_pvaj = *tail_state;
    }

    /// Reconfigure the active piece count in place. All inline buffers
    /// are sized to [`MAX_PIECES`] regardless, so this only updates the
    /// active extent and the underlying banded system's dimension.
    /// `solve()` must be called before any subsequent read; this leaves
    /// the buffers in an unspecified state.
    pub fn set_piece_count(&mut self, piece_num: usize) {
        debug_assert!(piece_num >= 1 && piece_num <= MAX_PIECES);
        self.n = piece_num;
        self.banded.set_dimension(8 * piece_num);
    }

    /// Currently configured piece count.
    #[inline]
    pub fn piece_count(&self) -> usize {
        self.n
    }

    /// Set waypoints + time allocation and solve for coefficients.
    ///
    /// `waypoints` contains the n−1 intermediate position targets in
    /// order; `times` contains the n segment durations. Mirrors the
    /// C++ `setParameters` exactly (same row indices, same sign
    /// convention on the −1 continuity entries).
    pub fn solve(&mut self, waypoints: &[Vec3], times: &[f32]) {
        debug_assert_eq!(times.len(), self.n);
        debug_assert_eq!(waypoints.len(), self.n - 1);

        for i in 0..self.n {
            let t = times[i];
            self.t1[i] = t;
            let t2 = t * t;
            self.t2[i] = t2;
            let t3 = t2 * t;
            self.t3[i] = t3;
            let t4 = t2 * t2;
            self.t4[i] = t4;
            self.t5[i] = t4 * t;
            self.t6[i] = t4 * t2;
            self.t7[i] = t4 * t3;
        }

        self.banded.reset();
        let sys_size = 8 * self.n;
        self.b[..sys_size].fill(Vector3::zeros());

        // Head boundary: c_0 = p_h, c_1 = v_h, 2·c_2 = a_h, 6·c_3 = j_h.
        self.banded.set(0, 0, 1.0);
        self.banded.set(1, 1, 1.0);
        self.banded.set(2, 2, 2.0);
        self.banded.set(3, 3, 6.0);

        self.b[0] = self.head_pvaj[0];
        self.b[1] = self.head_pvaj[1];
        self.b[2] = self.head_pvaj[2];
        self.b[3] = self.head_pvaj[3];

        // Intermediate junctions. Each junction emits 8 rows, indexed
        // 8i+4 … 8i+11. Layout follows MincoSnap.hpp verbatim.
        for i in 0..(self.n - 1) {
            let base = 8 * i;
            let t1 = self.t1[i];
            let t2 = self.t2[i];
            let t3 = self.t3[i];
            let t4 = self.t4[i];
            let t5 = self.t5[i];
            let t6 = self.t6[i];
            let t7 = self.t7[i];

            // row 8i+4: snap continuity → 24·c4 + 120·t·c5 + 360·t²·c6 + 840·t³·c7 = 24·c4'
            self.banded.set(base + 4, base + 4, 24.0);
            self.banded.set(base + 4, base + 5, 120.0 * t1);
            self.banded.set(base + 4, base + 6, 360.0 * t2);
            self.banded.set(base + 4, base + 7, 840.0 * t3);
            self.banded.set(base + 4, base + 12, -24.0);

            // row 8i+5: crackle continuity → 120·c5 + 720·t·c6 + 2520·t²·c7 = 120·c5'
            self.banded.set(base + 5, base + 5, 120.0);
            self.banded.set(base + 5, base + 6, 720.0 * t1);
            self.banded.set(base + 5, base + 7, 2520.0 * t2);
            self.banded.set(base + 5, base + 13, -120.0);

            // row 8i+6: pop continuity → 720·c6 + 5040·t·c7 = 720·c6'
            self.banded.set(base + 6, base + 6, 720.0);
            self.banded.set(base + 6, base + 7, 5040.0 * t1);
            self.banded.set(base + 6, base + 14, -720.0);

            // row 8i+7: position-at-waypoint (left side of junction).
            self.banded.set(base + 7, base + 0, 1.0);
            self.banded.set(base + 7, base + 1, t1);
            self.banded.set(base + 7, base + 2, t2);
            self.banded.set(base + 7, base + 3, t3);
            self.banded.set(base + 7, base + 4, t4);
            self.banded.set(base + 7, base + 5, t5);
            self.banded.set(base + 7, base + 6, t6);
            self.banded.set(base + 7, base + 7, t7);

            // row 8i+8: position continuity (right side − left side).
            self.banded.set(base + 8, base + 0, 1.0);
            self.banded.set(base + 8, base + 1, t1);
            self.banded.set(base + 8, base + 2, t2);
            self.banded.set(base + 8, base + 3, t3);
            self.banded.set(base + 8, base + 4, t4);
            self.banded.set(base + 8, base + 5, t5);
            self.banded.set(base + 8, base + 6, t6);
            self.banded.set(base + 8, base + 7, t7);
            self.banded.set(base + 8, base + 8, -1.0);

            // row 8i+9: velocity continuity.
            self.banded.set(base + 9, base + 1, 1.0);
            self.banded.set(base + 9, base + 2, 2.0 * t1);
            self.banded.set(base + 9, base + 3, 3.0 * t2);
            self.banded.set(base + 9, base + 4, 4.0 * t3);
            self.banded.set(base + 9, base + 5, 5.0 * t4);
            self.banded.set(base + 9, base + 6, 6.0 * t5);
            self.banded.set(base + 9, base + 7, 7.0 * t6);
            self.banded.set(base + 9, base + 9, -1.0);

            // row 8i+10: acceleration continuity.
            self.banded.set(base + 10, base + 2, 2.0);
            self.banded.set(base + 10, base + 3, 6.0 * t1);
            self.banded.set(base + 10, base + 4, 12.0 * t2);
            self.banded.set(base + 10, base + 5, 20.0 * t3);
            self.banded.set(base + 10, base + 6, 30.0 * t4);
            self.banded.set(base + 10, base + 7, 42.0 * t5);
            self.banded.set(base + 10, base + 10, -2.0);

            // row 8i+11: jerk continuity.
            self.banded.set(base + 11, base + 3, 6.0);
            self.banded.set(base + 11, base + 4, 24.0 * t1);
            self.banded.set(base + 11, base + 5, 60.0 * t2);
            self.banded.set(base + 11, base + 6, 120.0 * t3);
            self.banded.set(base + 11, base + 7, 210.0 * t4);
            self.banded.set(base + 11, base + 11, -6.0);

            // RHS row 8i+7 carries the waypoint position; rows 8i+8..8i+11
            // are zero (already cleared) since they encode continuity.
            self.b[base + 7] = waypoints[i];
        }

        // Tail boundary. Last 4 rows enforce p(T_n)=p_t, v(T_n)=v_t,
        // a(T_n)=a_t, j(T_n)=j_t at the end of segment n−1.
        let n8 = 8 * self.n;
        let last = self.n - 1;
        let t1 = self.t1[last];
        let t2 = self.t2[last];
        let t3 = self.t3[last];
        let t4 = self.t4[last];
        let t5 = self.t5[last];
        let t6 = self.t6[last];
        let t7 = self.t7[last];

        // p(T_last)
        self.banded.set(n8 - 4, n8 - 8, 1.0);
        self.banded.set(n8 - 4, n8 - 7, t1);
        self.banded.set(n8 - 4, n8 - 6, t2);
        self.banded.set(n8 - 4, n8 - 5, t3);
        self.banded.set(n8 - 4, n8 - 4, t4);
        self.banded.set(n8 - 4, n8 - 3, t5);
        self.banded.set(n8 - 4, n8 - 2, t6);
        self.banded.set(n8 - 4, n8 - 1, t7);

        // v(T_last)
        self.banded.set(n8 - 3, n8 - 7, 1.0);
        self.banded.set(n8 - 3, n8 - 6, 2.0 * t1);
        self.banded.set(n8 - 3, n8 - 5, 3.0 * t2);
        self.banded.set(n8 - 3, n8 - 4, 4.0 * t3);
        self.banded.set(n8 - 3, n8 - 3, 5.0 * t4);
        self.banded.set(n8 - 3, n8 - 2, 6.0 * t5);
        self.banded.set(n8 - 3, n8 - 1, 7.0 * t6);

        // a(T_last)
        self.banded.set(n8 - 2, n8 - 6, 2.0);
        self.banded.set(n8 - 2, n8 - 5, 6.0 * t1);
        self.banded.set(n8 - 2, n8 - 4, 12.0 * t2);
        self.banded.set(n8 - 2, n8 - 3, 20.0 * t3);
        self.banded.set(n8 - 2, n8 - 2, 30.0 * t4);
        self.banded.set(n8 - 2, n8 - 1, 42.0 * t5);

        // j(T_last)
        self.banded.set(n8 - 1, n8 - 5, 6.0);
        self.banded.set(n8 - 1, n8 - 4, 24.0 * t1);
        self.banded.set(n8 - 1, n8 - 3, 60.0 * t2);
        self.banded.set(n8 - 1, n8 - 2, 120.0 * t3);
        self.banded.set(n8 - 1, n8 - 1, 210.0 * t4);

        self.b[n8 - 4] = self.tail_pvaj[0];
        self.b[n8 - 3] = self.tail_pvaj[1];
        self.b[n8 - 2] = self.tail_pvaj[2];
        self.b[n8 - 1] = self.tail_pvaj[3];

        // Solve directly on `b` (in-place forward+back substitution).
        self.banded.factorize_lu();
        self.banded.solve3(&mut self.b[..sys_size]);
    }

    /// Extract the solved trajectory as a degree-7 PiecewisePolynomial.
    pub fn get_trajectory(&self) -> PiecewisePolynomial {
        let mut pieces = [Polynomial {
            degree: 0,
            duration: 0.0,
            coeffs: [ZERO3; super::polynomial::MAX_COEFFS],
        }; MAX_PIECES];

        for i in 0..self.n {
            let base = 8 * i;
            pieces[i] = Polynomial::new(7, self.t1[i], &self.b[base..base + 8]);
        }
        PiecewisePolynomial::from_pieces(&pieces[..self.n])
    }

    /// Closed-form ∫ ‖snap(t)‖² dt over all pieces. Mirrors the C++
    /// `MincoSnap::getEnergy` exactly.
    pub fn get_energy(&self) -> f32 {
        let mut energy = 0.0;
        for i in 0..self.n {
            let base = 8 * i;
            let b4 = self.b[base + 4];
            let b5 = self.b[base + 5];
            let b6 = self.b[base + 6];
            let b7 = self.b[base + 7];

            energy += 576.0 * b4.norm_squared() * self.t1[i]
                + 2880.0 * b4.dot(&b5) * self.t2[i]
                + 4800.0 * b5.norm_squared() * self.t3[i]
                + 5760.0 * b4.dot(&b6) * self.t3[i]
                + 21600.0 * b5.dot(&b6) * self.t4[i]
                + 10080.0 * b4.dot(&b7) * self.t4[i]
                + 25920.0 * b6.norm_squared() * self.t5[i]
                + 40320.0 * b5.dot(&b7) * self.t5[i]
                + 100800.0 * b6.dot(&b7) * self.t6[i]
                + 100800.0 * b7.norm_squared() * self.t7[i];
        }
        energy
    }

    /// Access the raw coefficient for piece `piece_idx`,
    /// coefficient index `coeff_idx`, dimension `dim`.
    #[inline]
    pub fn get_coeff(&self, piece_idx: usize, coeff_idx: usize, dim: usize) -> f32 {
        self.b[8 * piece_idx + coeff_idx][dim]
    }

    /// Borrow the 8 polynomial coefficients for `piece_idx` in
    /// ascending order.
    #[inline]
    pub fn piece_coeffs(&self, piece_idx: usize) -> &[Vector3<f32>] {
        let base = 8 * piece_idx;
        &self.b[base..base + 8]
    }

    /// Accumulate `scale · ∂E/∂coeffs` directly into `grad_c`
    /// (length 8·N). Mirrors `getEnergyPartialGradByCoeffs`. Rows
    /// 8i+0..8i+3 are unchanged (energy is independent of head-PVAJ
    /// rows); only rows 8i+4..8i+7 receive contributions.
    pub fn add_energy_grad_by_coeffs(&self, grad_c: &mut [Vector3<f32>], scale: f32) {
        for i in 0..self.n {
            let base = 8 * i;
            let b4 = self.b[base + 4];
            let b5 = self.b[base + 5];
            let b6 = self.b[base + 6];
            let b7 = self.b[base + 7];
            let t1 = self.t1[i];
            let t2 = self.t2[i];
            let t3 = self.t3[i];
            let t4 = self.t4[i];
            let t5 = self.t5[i];
            let t6 = self.t6[i];
            let t7 = self.t7[i];

            grad_c[base + 4] += (b4 * (1152.0 * t1)
                + b5 * (2880.0 * t2)
                + b6 * (5760.0 * t3)
                + b7 * (10080.0 * t4))
                * scale;
            grad_c[base + 5] += (b4 * (2880.0 * t2)
                + b5 * (9600.0 * t3)
                + b6 * (21600.0 * t4)
                + b7 * (40320.0 * t5))
                * scale;
            grad_c[base + 6] += (b4 * (5760.0 * t3)
                + b5 * (21600.0 * t4)
                + b6 * (51840.0 * t5)
                + b7 * (100800.0 * t6))
                * scale;
            grad_c[base + 7] += (b4 * (10080.0 * t4)
                + b5 * (40320.0 * t5)
                + b6 * (100800.0 * t6)
                + b7 * (201600.0 * t7))
                * scale;
            // rows 8i+0..8i+3 unchanged (energy is independent of them).
        }
    }

    /// Accumulate `scale · ∂E/∂times` directly into `grad_t`
    /// (length N). Mirrors `getEnergyPartialGradByTimes`.
    pub fn add_energy_grad_by_times(&self, grad_t: &mut [f32], scale: f32) {
        for i in 0..self.n {
            let base = 8 * i;
            let b4 = self.b[base + 4];
            let b5 = self.b[base + 5];
            let b6 = self.b[base + 6];
            let b7 = self.b[base + 7];
            let t1 = self.t1[i];
            let t2 = self.t2[i];
            let t3 = self.t3[i];
            let t4 = self.t4[i];
            let t5 = self.t5[i];
            let t6 = self.t6[i];
            grad_t[i] += scale
                * (576.0 * b4.norm_squared()
                    + 5760.0 * b4.dot(&b5) * t1
                    + 14400.0 * b5.norm_squared() * t2
                    + 17280.0 * b4.dot(&b6) * t2
                    + 86400.0 * b5.dot(&b6) * t3
                    + 40320.0 * b4.dot(&b7) * t3
                    + 129600.0 * b6.norm_squared() * t4
                    + 201600.0 * b5.dot(&b7) * t4
                    + 604800.0 * b6.dot(&b7) * t5
                    + 705600.0 * b7.norm_squared() * t6);
        }
    }

    /// Backpropagate gradients through the MINCO snap system.
    /// Mirrors `propagateGrad` from `minco_snap.hpp`.
    pub fn propagate_grad(
        &self,
        partial_grad_c: &[Vector3<f32>],
        partial_grad_t: &[f32],
        grad_points: &mut [Vector3<f32>],
        grad_times: &mut [f32],
    ) {
        let n = self.n;
        let sys_size = 8 * n;

        for gp in grad_points.iter_mut() {
            *gp = Vector3::zeros();
        }
        for gt in grad_times.iter_mut() {
            *gt = 0.0;
        }

        // Solve A^T · adj_grad = partial_grad_c.
        let mut adj_grad = [Vector3::<f32>::zeros(); 8 * MAX_PIECES];
        adj_grad[..sys_size].copy_from_slice(&partial_grad_c[..sys_size]);
        self.banded.solve3_adj(&mut adj_grad[..sys_size]);

        // Extract gradient w.r.t. waypoints from the position-at-waypoint
        // rows (8i+7).
        for i in 0..(n - 1) {
            grad_points[i] = adj_grad[8 * i + 7];
        }

        // Time gradient for interior segments via ∂A/∂T_i.
        // The block adj_grad[8i+4 .. 8i+11] (8 rows) pairs with B1
        // rows 0..7 in the C++ source, which respectively encode:
        //   row 0: -crackle  | row 4: -velocity (pos continuity row)
        //   row 1: -d_crackle| row 5: -acceleration
        //   row 2: -dd_crackle| row 6: -jerk
        //   row 3: -velocity (pos-at-waypoint row)
        //   row 7: -snap
        for i in 0..(n - 1) {
            let o = i * 8;
            let t1 = self.t1[i];
            let t2 = self.t2[i];
            let t3 = self.t3[i];
            let t4 = self.t4[i];
            let t5 = self.t5[i];
            let t6 = self.t6[i];

            // -velocity at end of segment i.
            let neg_vel = -(self.b[o + 1]
                + self.b[o + 2] * (2.0 * t1)
                + self.b[o + 3] * (3.0 * t2)
                + self.b[o + 4] * (4.0 * t3)
                + self.b[o + 5] * (5.0 * t4)
                + self.b[o + 6] * (6.0 * t5)
                + self.b[o + 7] * (7.0 * t6));

            // -acceleration.
            let neg_acc = -(self.b[o + 2] * 2.0
                + self.b[o + 3] * (6.0 * t1)
                + self.b[o + 4] * (12.0 * t2)
                + self.b[o + 5] * (20.0 * t3)
                + self.b[o + 6] * (30.0 * t4)
                + self.b[o + 7] * (42.0 * t5));

            // -jerk.
            let neg_jerk = -(self.b[o + 3] * 6.0
                + self.b[o + 4] * (24.0 * t1)
                + self.b[o + 5] * (60.0 * t2)
                + self.b[o + 6] * (120.0 * t3)
                + self.b[o + 7] * (210.0 * t4));

            // -snap.
            let neg_snap = -(self.b[o + 4] * 24.0
                + self.b[o + 5] * (120.0 * t1)
                + self.b[o + 6] * (360.0 * t2)
                + self.b[o + 7] * (840.0 * t3));

            // -crackle.
            let neg_crackle = -(self.b[o + 5] * 120.0
                + self.b[o + 6] * (720.0 * t1)
                + self.b[o + 7] * (2520.0 * t2));

            // -d_crackle.
            let neg_d_crackle = -(self.b[o + 6] * 720.0 + self.b[o + 7] * (5040.0 * t1));

            // -dd_crackle.
            let neg_dd_crackle = self.b[o + 7] * -5040.0;

            // C++ pairing:
            //   B1.row(0) (neg_crackle)     · adj_grad[8i+4]
            //   B1.row(1) (neg_d_crackle)   · adj_grad[8i+5]
            //   B1.row(2) (neg_dd_crackle)  · adj_grad[8i+6]
            //   B1.row(3) (neg_vel)         · adj_grad[8i+7]
            //   B1.row(4) (neg_vel)         · adj_grad[8i+8]
            //   B1.row(5) (neg_acc)         · adj_grad[8i+9]
            //   B1.row(6) (neg_jerk)        · adj_grad[8i+10]
            //   B1.row(7) (neg_snap)        · adj_grad[8i+11]
            let sum = neg_crackle.dot(&adj_grad[o + 4])
                + neg_d_crackle.dot(&adj_grad[o + 5])
                + neg_dd_crackle.dot(&adj_grad[o + 6])
                + neg_vel.dot(&adj_grad[o + 7])
                + neg_vel.dot(&adj_grad[o + 8])
                + neg_acc.dot(&adj_grad[o + 9])
                + neg_jerk.dot(&adj_grad[o + 10])
                + neg_snap.dot(&adj_grad[o + 11]);
            grad_times[i] = sum;
        }

        // Last segment (tail boundary) — only 4 rows pair (pos, vel,
        // acc, jerk at T_last).
        {
            let last = n - 1;
            let o = last * 8;
            let t1 = self.t1[last];
            let t2 = self.t2[last];
            let t3 = self.t3[last];
            let t4 = self.t4[last];
            let t5 = self.t5[last];
            let t6 = self.t6[last];

            let neg_vel = -(self.b[o + 1]
                + self.b[o + 2] * (2.0 * t1)
                + self.b[o + 3] * (3.0 * t2)
                + self.b[o + 4] * (4.0 * t3)
                + self.b[o + 5] * (5.0 * t4)
                + self.b[o + 6] * (6.0 * t5)
                + self.b[o + 7] * (7.0 * t6));

            let neg_acc = -(self.b[o + 2] * 2.0
                + self.b[o + 3] * (6.0 * t1)
                + self.b[o + 4] * (12.0 * t2)
                + self.b[o + 5] * (20.0 * t3)
                + self.b[o + 6] * (30.0 * t4)
                + self.b[o + 7] * (42.0 * t5));

            let neg_jerk = -(self.b[o + 3] * 6.0
                + self.b[o + 4] * (24.0 * t1)
                + self.b[o + 5] * (60.0 * t2)
                + self.b[o + 6] * (120.0 * t3)
                + self.b[o + 7] * (210.0 * t4));

            let neg_snap = -(self.b[o + 4] * 24.0
                + self.b[o + 5] * (120.0 * t1)
                + self.b[o + 6] * (360.0 * t2)
                + self.b[o + 7] * (840.0 * t3));

            // Tail block pairs B2.row(0..3) with adj_grad[8N-4..8N-1].
            let sum = neg_vel.dot(&adj_grad[8 * n - 4])
                + neg_acc.dot(&adj_grad[8 * n - 3])
                + neg_jerk.dot(&adj_grad[8 * n - 2])
                + neg_snap.dot(&adj_grad[8 * n - 1]);
            grad_times[last] = sum;
        }

        for i in 0..n {
            grad_times[i] += partial_grad_t[i];
        }
    }
}

// ── Quadrotor flatness map (port of `toStateWithTiltYaw`) ────────────
//
// Mirrors `drolib::QuadManifold::toStateWithTiltYaw` from
// `tmp/planner/src/system/quadrotor_manifold.cpp` (line 1695). Maps a
// (a, j, s) flat-output triple plus a yaw triple (ψ, ψ̇, ψ̈) and
// gravity into a quadrotor setpoint: collective thrust per unit mass,
// attitude quaternion, body rate, body angular acceleration. Position
// and velocity are kinematic flat outputs but unused by this map, so
// they are not part of the signature; callers multiply by mass to get
// thrust force.
//
// Yaw triple is `[ψ, ψ̇, ψ̈]` in rad / rad·s⁻¹ / rad·s⁻².

/// Output of [`flatness_to_state_tilt_yaw`].
///
/// All fields are world-frame except `omega` and `omega_dot`, which
/// are body-frame (the convention every quadrotor controller in this
/// codebase uses).
#[derive(Copy, Clone, Debug)]
pub struct FlatState {
    pub thrust_per_mass: f32,
    pub attitude: UnitQuaternion<f32>,
    pub omega: Vec3,
    pub omega_dot: Vec3,
}

/// Hard floor on `‖α‖²` (m²/s⁴). Below this we treat the input as
/// near-free-fall and bail; the closed-form inversion divides by
/// `‖α‖³` and `‖α‖⁵`, both of which would overflow `f32::MAX ≈ 3.4·10³⁸`
/// well before `‖α‖` reaches zero. The threshold is `(0.1·g)² ≈ 1`,
/// so any flight regime where the vehicle is still net-accelerating
/// upward stays well above the floor.
const ALPHA_NORM_SQR_FLOOR: f32 = 1.0;

/// Tilt singularity threshold on `zB.z + 1`. Below this the tilt-yaw
/// parameterization breaks down (`omg_den → 0`, the `dzb2²/omg_den²`
/// term in `ω̇` blows up). We refuse to evaluate rather than silently
/// emit garbage. ~5.7° below "fully inverted" — far enough from any
/// practical flight regime that hitting it indicates a planning bug
/// upstream, close enough that no legitimate maneuver hits it.
const TILT_DEN_FLOOR: f32 = 5e-3;

/// Did the flatness map evaluate cleanly, or did it hit a singularity?
///
/// `Singular` is returned by [`flatness_to_state_tilt_yaw`] without
/// consulting the offending input — caller is expected to short-circuit
/// (hold last setpoint, trip a fault, etc.). The MINCO trajectories
/// produced by this codebase should never hit it; if they do, the
/// trajectory is unflyable and the controller cannot rescue it from
/// downstream NaN.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FlatnessFault {
    /// `‖a + g·ẑ‖²` was below [`ALPHA_NORM_SQR_FLOOR`].
    NearFreeFall,
    /// `zB.z + 1` was below [`TILT_DEN_FLOOR`] (vehicle inverted or
    /// past the tilt-yaw parameterization's singularity).
    InvertedTilt,
}

/// Flat-output → state map (yaw-as-input convention).
///
/// Direct port of `toStateWithTiltYaw`. Returns `FlatState` whose
/// `thrust_per_mass = ‖a + g·ẑ‖` (collective thrust per unit mass),
/// `attitude` = tilt-then-yaw composition with `psi` as the yaw, and
/// `omega` / `omega_dot` from the closed-form differential-flatness
/// inversion.
///
/// `gravity` is the gravitational acceleration magnitude in m/s²
/// (positive). Pass `QuadPlanningConfig::grav` (default 9.81).
///
/// Returns `Err(FlatnessFault::*)` for the two singularities the
/// parameterization cannot represent: near-free-fall (`‖α‖ → 0`) and
/// inversion (`zB.z → -1`). Both should be unreachable on a valid
/// MINCO trajectory; if they fire, the upstream planner produced an
/// unflyable schedule and the caller must drop the sample.
///
/// ## Numerical layout (f32 / Cortex-M7 FPU)
///
/// The Cortex-M7 single-precision FPU pipelines `VMUL/VFMA` at one
/// per cycle but `VDIV/VSQRT` are 14-cycle blocking ops. The body
/// hoists every reciprocal once and reuses it via multiplies; only
/// **two** divisions and **one** sqrt are issued total.
pub fn flatness_to_state_tilt_yaw(
    acc: Vec3,
    jer: Vec3,
    sna: Vec3,
    yaw_triple: [f32; 3],
    gravity: f32,
) -> Result<FlatState, FlatnessFault> {
    let psi = yaw_triple[0];
    let dpsi = yaw_triple[1];
    let ddpsi = yaw_triple[2];

    // Single sin/cos pair via libm — no half-angle pair here; the
    // quaternion construction in `quaternion_from_zb_and_yaw`
    // computes its own ψ/2 sin/cos once internally.
    let c_psi = libm::cosf(psi);
    let s_psi = libm::sinf(psi);

    // α = a + g·ẑ_w.
    let alpha = Vec3::new(acc[0], acc[1], acc[2] + gravity);
    let alpha_norm_2 = alpha.norm_squared();
    if alpha_norm_2 < ALPHA_NORM_SQR_FLOOR {
        return Err(FlatnessFault::NearFreeFall);
    }
    // Single sqrt for the whole function — every other power of ‖α‖
    // is derived by multiplication via `inv_alpha_norm_*` below.
    let alpha_norm_1 = libm::sqrtf(alpha_norm_2);

    let alpha_dot_j = alpha.dot(&jer);
    let alpha_dot_j_sqr = alpha_dot_j * alpha_dot_j;
    let j_norm_2 = jer.norm_squared();

    // Hoisted reciprocals: one VDIV produces inv_alpha_norm_1, the
    // rest of the powers come from multiplies (free on Cortex-M7).
    let inv_alpha_norm_1 = 1.0 / alpha_norm_1;
    let inv_alpha_norm_2 = inv_alpha_norm_1 * inv_alpha_norm_1;
    let inv_a3 = inv_alpha_norm_2 * inv_alpha_norm_1;
    // inv_a5 derived via multiply, *not* `1.0 / alpha_norm_5` — saves
    // a 14-cycle VDIV.
    let inv_a5 = inv_a3 * inv_alpha_norm_2;

    // zB = α / ‖α‖
    let z_b = alpha * inv_alpha_norm_1;
    let zb0 = z_b[0];
    let zb1 = z_b[1];
    let zb2 = z_b[2];

    // Singularity guard: refuse rather than clamp. C++ clamps to
    // `±1e-6` and propagates the (now meaningless) result; we exit
    // cleanly so the controller cannot consume garbage. See
    // `FlatnessFault::InvertedTilt`.
    let zb2_1 = zb2 + 1.0;
    if zb2_1 < TILT_DEN_FLOOR {
        return Err(FlatnessFault::InvertedTilt);
    }

    // Collective thrust per unit mass = ‖α‖ (already computed). Avoids
    // 5 extra FLOPs and ~3 ulp of f32 accumulation that the longhand
    // `zB · α` would carry.
    let thrust_per_mass = alpha_norm_1;

    // dzB = N(α) · j = (j − α·(α·j)/‖α‖²) / ‖α‖.
    // Cleaner *and* better-conditioned than expanding the symmetric
    // `ng**` matrix by hand: when α is nearly axis-aligned, the
    // longhand form has `α_sqr_i + α_sqr_j` cancellations that this
    // form sidesteps. 3 mul + 3 sub + 3 mul = 9 FLOPs vs 9+6=15 in
    // the longhand `ng**` formulation.
    let proj_j = alpha_dot_j * inv_alpha_norm_2;
    let dzb0 = (jer[0] - alpha[0] * proj_j) * inv_alpha_norm_1;
    let dzb1 = (jer[1] - alpha[1] * proj_j) * inv_alpha_norm_1;
    let dzb2 = (jer[2] - alpha[2] * proj_j) * inv_alpha_norm_1;

    // N(α)·s by the same projection identity.
    let alpha_dot_s = alpha.dot(&sna);
    let proj_s = alpha_dot_s * inv_alpha_norm_2;
    let dn_alpha_s_0 = (sna[0] - alpha[0] * proj_s) * inv_alpha_norm_1;
    let dn_alpha_s_1 = (sna[1] - alpha[1] * proj_s) * inv_alpha_norm_1;
    let dn_alpha_s_2 = (sna[2] - alpha[2] * proj_s) * inv_alpha_norm_1;

    // ddzB = -2·(α·j)/‖α‖³ · j  +  α · (3·(α·j)² − ‖α‖²·‖j‖²)/‖α‖⁵
    //        +  N(α) · s
    //
    // The `common` term `(3·(α·j)² − ‖α‖²·‖j‖²) · inv_a5` folds a
    // catastrophic-cancellation-prone difference of two same-magnitude
    // terms into a single subtraction the optimizer can fuse. C++
    // (double) is unaffected; in f32 the original form measurably
    // increased the body-rate divergence vs the f64 reference.
    let common = (3.0 * alpha_dot_j_sqr - alpha_norm_2 * j_norm_2) * inv_a5;
    let neg_two_aj_inv_a3 = -2.0 * alpha_dot_j * inv_a3;
    let ddzb0 = neg_two_aj_inv_a3 * jer[0] + alpha[0] * common + dn_alpha_s_0;
    let ddzb1 = neg_two_aj_inv_a3 * jer[1] + alpha[1] * common + dn_alpha_s_1;
    let ddzb2 = neg_two_aj_inv_a3 * jer[2] + alpha[2] * common + dn_alpha_s_2;

    // Attitude: tilt(zB) ∘ yaw(ψ). `quaternion_from_zb_and_yaw` with
    // `use_tilt = true` is the same closed form as the C++ source's
    // tilt0/tilt1/tilt2 construction (see rotation.rs:268..283).
    let attitude = quaternion_from_zb_and_yaw(&z_b, psi, true);

    // Body rate. Hoist `1/omg_den` (one VDIV) and reuse it via
    // multiplies for both ω and ω̇.
    let inv_omg_den = 1.0 / zb2_1;
    let inv_omg_den_2 = inv_omg_den * inv_omg_den;

    let omg_term = dzb2 * inv_omg_den;
    let tmp_omg_1 = zb0 * s_psi - zb1 * c_psi;
    let tmp_omg_2 = zb0 * c_psi + zb1 * s_psi;
    let tmp_omg_3 = zb1 * dzb0 - zb0 * dzb1;
    // Hoisted: appear in both ω.x/.y *and* (as `tmp_omg_4/5` in C++)
    // in the ω̇.x/.y correction. Saves 4 mul + 2 sub.
    let dz_psi_a = dzb0 * s_psi - dzb1 * c_psi;
    let dz_psi_b = dzb0 * c_psi + dzb1 * s_psi;
    let omega = Vec3::new(
        dz_psi_a - tmp_omg_1 * omg_term,
        dz_psi_b - tmp_omg_2 * omg_term,
        tmp_omg_3 * inv_omg_den + dpsi,
    );

    // Body angular acceleration. Reuses dz_psi_{a,b} from above.
    let tmp_omg_6 = zb1 * ddzb0 - zb0 * ddzb1;
    let dzb2_sqr = dzb2 * dzb2;

    let omega_dot = Vec3::new(
        ddzb0 * s_psi - ddzb1 * c_psi - ddzb2 * tmp_omg_1 * inv_omg_den
            - dzb2 * dz_psi_a * inv_omg_den
            + dzb2_sqr * tmp_omg_1 * inv_omg_den_2,
        ddzb0 * c_psi + ddzb1 * s_psi - ddzb2 * tmp_omg_2 * inv_omg_den
            - dzb2 * dz_psi_b * inv_omg_den
            + dzb2_sqr * tmp_omg_2 * inv_omg_den_2,
        tmp_omg_6 * inv_omg_den - tmp_omg_3 * dzb2 * inv_omg_den_2 + ddpsi,
    );

    Ok(FlatState {
        thrust_per_mass,
        attitude,
        omega,
        omega_dot,
    })
}

/// Pole-safe flat-output → (thrust, attitude, body-rate) map for the MPC
/// outer-loop feedforward.
///
/// Companion to [`flatness_to_state_tilt_yaw`] tailored to a 4-channel
/// MPC whose control vector is `[T, ω_x, ω_y, ω_z]`. Returns:
///
/// - `thrust_per_mass = ‖a + g·ẑ‖`. Parameterization-independent —
///   has no dependence on the tilt-yaw decomposition, so it stays
///   well-defined arbitrarily close to the inverted pole.
/// - `attitude` from [`quaternion_from_zb_and_yaw`] with `use_tilt = true`.
///   The unique singularity at `z_b = -ẑ` is handled by a substituted
///   180° flip inside that function.
/// - `omega` in body frame, computed from the *minimum-norm* world
///   angular velocity
///
///       ω_world  =  z_b × dz_b  +  ψ̇ · ẑ_world
///
///   then rotated into body frame via the (pole-safe) attitude
///   quaternion. This is the parameterization-independent angular
///   velocity that produces the smooth attitude trajectory through
///   the pole; it is finite and bounded everywhere `‖a + g·ẑ‖` is
///   above the free-fall floor.
///
/// ## Why min-norm body rate (and not the C++ tilt-yaw closed form)
///
/// [`flatness_to_state_tilt_yaw`] computes ω in the *intrinsic-Euler
/// tilt-then-yaw* convention. That ω contains a `(zb1·dzb0 − zb0·dzb1)
/// / (zb.z + 1)` body-z term that *diverges* as `z_b.z → −1` — it is
/// the rate the body must spin around its z-axis to keep the intrinsic
/// Euler "yaw" angle constant while tilting through the pole. A real
/// drone cannot supply unbounded ω, so feeding this quantity into the
/// MPC's `u_ref` would push the input cost off a cliff near the pole.
///
/// The min-norm form picks instead the body rate that:
///
/// 1. Correctly evolves `z_b(t)` along the trajectory (the perpendicular
///    component is `z_b × dz_b`, identical to the limit of the closed
///    form).
/// 2. Adds yaw rate as a **world-z** rate (`ψ̇ · ẑ_world`) rather than as
///    an intrinsic-Euler rate. For our MINCO trajectories `ψ̇ = 0`, so
///    this distinction is invisible to the MPC.
///
/// At the pole the min-norm body rate matches the body-rate limit of the
/// substituted attitude in [`quaternion_from_zb_and_yaw`] — the pair is
/// kinematically consistent.
///
/// `omega_dot` is intentionally not returned: the firmware MPC's input
/// is `[T, ω_x, ω_y, ω_z]` (no `ω̇` channel), and the closed-form ω̇
/// formula contains `1/(zb.z + 1)²` which diverges quadratically faster
/// than ω. Skipping it removes the worst pole singularity from the
/// integration path entirely.
pub fn flatness_to_thrust_omega(
    acc: Vec3,
    jer: Vec3,
    yaw: f32,
    yaw_rate: f32,
    gravity: f32,
) -> Result<(f32, UnitQuaternion<f32>, Vec3), FlatnessFault> {
    let alpha = Vec3::new(acc[0], acc[1], acc[2] + gravity);
    let alpha_norm_2 = alpha.norm_squared();
    if alpha_norm_2 < ALPHA_NORM_SQR_FLOOR {
        return Err(FlatnessFault::NearFreeFall);
    }
    let alpha_norm_1 = libm::sqrtf(alpha_norm_2);
    let inv_alpha_norm_1 = 1.0 / alpha_norm_1;
    let inv_alpha_norm_2 = inv_alpha_norm_1 * inv_alpha_norm_1;

    // z_b = α / ‖α‖
    let z_b = alpha * inv_alpha_norm_1;

    // dz_b = N(α)·j = (j − α·(α·j)/‖α‖²) / ‖α‖. No division by `zb.z + 1`,
    // so this stays bounded across the pole.
    let proj_j = alpha.dot(&jer) * inv_alpha_norm_2;
    let dz_b = Vec3::new(
        (jer[0] - alpha[0] * proj_j) * inv_alpha_norm_1,
        (jer[1] - alpha[1] * proj_j) * inv_alpha_norm_1,
        (jer[2] - alpha[2] * proj_j) * inv_alpha_norm_1,
    );

    // World-frame angular velocity: perpendicular component rotates
    // z_b along the trajectory; world-z component carries the yaw rate.
    let perp = z_b.cross(&dz_b);
    let omega_world = Vec3::new(perp[0], perp[1], perp[2] + yaw_rate);

    // Attitude is pole-safe (substituted 180° flip at z_b = -ẑ).
    let attitude = quaternion_from_zb_and_yaw(&z_b, yaw, true);

    // ω_body = R^T · ω_world. UnitQuaternion's inverse_transform_vector
    // is `q^{-1} · v · q` — the standard body-from-world rotation.
    let omega_body = attitude.inverse_transform_vector(&omega_world);

    Ok((alpha_norm_1, attitude, omega_body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::ZERO3;

    /// Single-piece hover sanity: head = tail at the same point with
    /// zero PVAJ, the solution must be the zero polynomial.
    #[test]
    fn test_single_piece_zero_motion() {
        let head: PVAJ3D = [ZERO3, ZERO3, ZERO3, ZERO3];
        let tail: PVAJ3D = [ZERO3, ZERO3, ZERO3, ZERO3];
        let mut solver = MincoSnap::new(&head, &tail, 1);
        solver.solve(&[], &[1.0]);
        let traj = solver.get_trajectory();
        let p_mid = traj.get_pos(0.5);
        assert!(p_mid.norm() < 1e-4, "zero-motion solve nonzero: {p_mid:?}");
    }

    /// Single-piece p2p with zero VAJ at both endpoints.
    #[test]
    fn test_single_piece_p2p() {
        let head: PVAJ3D = [ZERO3, ZERO3, ZERO3, ZERO3];
        let tail: PVAJ3D = [Vec3::new(1.0, 0.0, 0.0), ZERO3, ZERO3, ZERO3];

        let mut solver = MincoSnap::new(&head, &tail, 1);
        solver.solve(&[], &[1.0]);
        let traj = solver.get_trajectory();

        let p0 = traj.get_pos(0.0);
        let p1 = traj.get_pos(1.0);
        assert!(p0[0].abs() < 1e-4, "p0 wrong: {p0:?}");
        assert!((p1[0] - 1.0).abs() < 1e-4, "p1 wrong: {p1:?}");

        let v0 = traj.get_vel(0.0);
        let v1 = traj.get_vel(1.0);
        assert!(v0.norm() < 1e-4, "v0 not zero: {v0:?}");
        assert!(v1.norm() < 1e-4, "v1 not zero: {v1:?}");

        let a0 = traj.get_acc(0.0);
        let a1 = traj.get_acc(1.0);
        assert!(a0.norm() < 1e-3, "a0 not zero: {a0:?}");
        assert!(a1.norm() < 1e-3, "a1 not zero: {a1:?}");

        let j0 = traj.get_jerk(0.0);
        let j1 = traj.get_jerk(1.0);
        assert!(j0.norm() < 1e-2, "j0 not zero: {j0:?}");
        assert!(j1.norm() < 1e-2, "j1 not zero: {j1:?}");
    }

    /// Hover flatness: zero a/j/s + zero yaw → identity attitude,
    /// zero body rate, thrust = g.
    #[test]
    fn test_flatness_hover() {
        let st = flatness_to_state_tilt_yaw(ZERO3, ZERO3, ZERO3, [0.0; 3], 9.81)
            .expect("hover should be a valid flat state");
        assert!((st.thrust_per_mass - 9.81).abs() < 1e-4);
        let q = st.attitude;
        assert!((q.w - 1.0).abs() < 1e-4);
        assert!(q.i.abs() < 1e-4);
        assert!(q.j.abs() < 1e-4);
        assert!(q.k.abs() < 1e-4);
        assert!(st.omega.norm() < 1e-4);
    }

    /// Free-fall: a = -g·ẑ → ‖α‖ ≈ 0; should fault, not NaN.
    #[test]
    fn test_flatness_free_fall_fault() {
        let acc = Vec3::new(0.0, 0.0, -9.81);
        let r = flatness_to_state_tilt_yaw(acc, ZERO3, ZERO3, [0.0; 3], 9.81);
        assert_eq!(r.unwrap_err(), FlatnessFault::NearFreeFall);
    }

    /// Inverted: a chosen so zB ≈ -ẑ; should fault, not produce
    /// blow-up ω̇.
    #[test]
    fn test_flatness_inverted_fault() {
        // α = (0, 0, -|α|) → zB = (0, 0, -1), zb2_1 = 0.
        let acc = Vec3::new(0.0, 0.0, -2.0 * 9.81);
        let r = flatness_to_state_tilt_yaw(acc, ZERO3, ZERO3, [0.0; 3], 9.81);
        assert_eq!(r.unwrap_err(), FlatnessFault::InvertedTilt);
    }

    // ── flatness_to_thrust_omega (pole-safe MPC u_ref feedforward) ──

    /// Hover: zero a/j → identity attitude, zero body rate, thrust=g.
    /// Same expectation as `test_flatness_hover` for the full map; this
    /// confirms the trimmed function returns the same hover values.
    #[test]
    fn test_thrust_omega_hover() {
        let (tpm, q, omega) =
            flatness_to_thrust_omega(ZERO3, ZERO3, 0.0, 0.0, 9.81).expect("hover ok");
        assert!((tpm - 9.81).abs() < 1e-4);
        assert!((q.w - 1.0).abs() < 1e-4);
        assert!(q.i.abs() < 1e-4 && q.j.abs() < 1e-4 && q.k.abs() < 1e-4);
        assert!(omega.norm() < 1e-4, "hover omega nonzero: {omega:?}");
    }

    /// Free-fall fault parity with the full map.
    #[test]
    fn test_thrust_omega_free_fall_fault() {
        let acc = Vec3::new(0.0, 0.0, -9.81);
        let r = flatness_to_thrust_omega(acc, ZERO3, 0.0, 0.0, 9.81);
        assert_eq!(r.unwrap_err(), FlatnessFault::NearFreeFall);
    }

    /// At the inverted pole the *full* map faults (`InvertedTilt`)
    /// because its closed-form ω diverges. The pole-safe map must NOT
    /// fault — that is the whole point — and the body rate it returns
    /// must be finite.
    #[test]
    fn test_thrust_omega_pole_no_fault_finite_omega() {
        // α = (0, 0, -2g) → z_b = (0, 0, -1): exact pole.
        let acc = Vec3::new(0.0, 0.0, -2.0 * 9.81);
        // Some nonzero jerk in xy so dz_b ≠ 0 and the perpendicular
        // angular-velocity component is non-trivial.
        let jer = Vec3::new(3.0, 1.5, 0.0);
        let (tpm, _q, omega) = flatness_to_thrust_omega(acc, jer, 0.3, 0.0, 9.81)
            .expect("pole-safe ok at z_b = -ẑ");
        // Thrust per mass = ‖α‖ = g (positive, finite).
        assert!((tpm - 9.81).abs() < 1e-3, "tpm wrong at pole: {tpm}");
        // Body rate must be finite and bounded — `‖dz_b‖` here is on
        // the order of `‖j‖/‖α‖` ≈ 3.4/9.81 ≈ 0.35 rad/s, so the body
        // rate magnitude should be of that order, not "infinite".
        assert!(
            omega.iter().all(|c| c.is_finite()),
            "non-finite omega at pole: {omega:?}"
        );
        assert!(
            omega.norm() < 5.0,
            "implausibly large omega at pole: {omega:?}"
        );
    }

    /// Off the pole, the pole-safe map's ω agrees with the geometric
    /// `z_b × dz_b` projected into body frame (which is its definition).
    /// This regression-locks the formula and catches any future axis-
    /// or sign-flipping mistake.
    #[test]
    fn test_thrust_omega_matches_min_norm_definition() {
        use nalgebra::Vector3;
        let acc = Vec3::new(2.0, -1.0, 1.5);
        let jer = Vec3::new(0.7, 0.4, -0.2);
        let yaw = 0.5;
        let yaw_rate = 0.0;
        let (tpm, q, omega_body) =
            flatness_to_thrust_omega(acc, jer, yaw, yaw_rate, 9.81).expect("nominal ok");

        // Thrust per mass = ‖a + g·ẑ‖
        let alpha = Vec3::new(acc[0], acc[1], acc[2] + 9.81);
        let alpha_norm = alpha.norm();
        assert!((tpm - alpha_norm).abs() < 1e-4);

        // Reconstruct ω_world from body-frame ω via the attitude.
        let omega_world_back = q * omega_body;

        // Geometric ω_world (yaw_rate = 0): z_b × dz_b
        let z_b = alpha / alpha_norm;
        let proj = alpha.dot(&jer) / (alpha_norm * alpha_norm);
        let dz_b = (jer - alpha * proj) / alpha_norm;
        let expected = z_b.cross(&dz_b);

        let diff: Vector3<f32> = omega_world_back - expected;
        assert!(
            diff.norm() < 1e-4,
            "omega_world reconstructed = {omega_world_back:?}, expected {expected:?}"
        );
    }

    /// Yaw rate of `ψ̇` rad/s in world-z, identity attitude (z_b = ẑ,
    /// yaw = 0): should produce body rate `(0, 0, ψ̇)` exactly. Confirms
    /// the world-z yaw-rate convention.
    #[test]
    fn test_thrust_omega_yaw_rate_at_hover() {
        let yaw_rate = 0.7;
        let (_tpm, _q, omega) =
            flatness_to_thrust_omega(ZERO3, ZERO3, 0.0, yaw_rate, 9.81).expect("ok");
        assert!(omega.x.abs() < 1e-4);
        assert!(omega.y.abs() < 1e-4);
        assert!((omega.z - yaw_rate).abs() < 1e-4);
    }

    /// Sweep z_b through the inverted pole along a continuous path and
    /// confirm thrust + body rate stay finite and bounded across the
    /// crossing. This is the regression test for the original bug:
    /// the full-map closed form blows up ω as `1/(zb.z + 1)`; the
    /// pole-safe map must not.
    #[test]
    fn test_thrust_omega_continuous_through_pole() {
        // Sweep φ ∈ [π/2 − δ, π/2 + δ] where the trajectory α =
        // ‖α‖ · (sin φ, 0, −cos φ) crosses the pole exactly at φ = π/2
        // (z_b = (1, 0, 0) → (0, 0, -1) → (-1, 0, 0)). dα/dφ supplies
        // the jerk via α̇ ≈ (dα/dφ) · φ̇ ; we use φ̇ = 1 rad/s for
        // simplicity, which makes ‖dz_b‖ = 1 rad/s by construction.
        let alpha_mag = 12.0; // > free-fall floor
        let phi_dot = 1.0;
        let mut max_norm = 0.0f32;
        let mut all_finite = true;
        for i in 0..201 {
            let phi = core::f32::consts::FRAC_PI_2 + (i as f32 - 100.0) * 1e-3;
            let s = libm::sinf(phi);
            let c = libm::cosf(phi);
            let alpha = Vec3::new(alpha_mag * s, 0.0, -alpha_mag * c);
            // d/dφ α = α_mag · (c, 0, s); jerk = α̇ − 0 = (dα/dφ)·φ̇.
            let alpha_dot = Vec3::new(alpha_mag * c * phi_dot, 0.0, alpha_mag * s * phi_dot);
            let acc = Vec3::new(alpha[0], alpha[1], alpha[2] - (-9.81)); // a = α − g·ẑ; here g·ẑ = (0,0,9.81), so a = α − (0,0,9.81)
            let jer = alpha_dot;
            let r = flatness_to_thrust_omega(acc, jer, 0.0, 0.0, 9.81);
            match r {
                Ok((tpm, _, omega)) => {
                    if !tpm.is_finite() || omega.iter().any(|c| !c.is_finite()) {
                        all_finite = false;
                    }
                    max_norm = max_norm.max(omega.norm());
                }
                Err(FlatnessFault::NearFreeFall) => {
                    // Possible at certain φ if α magnitude dips; should not happen here.
                    panic!("unexpected NearFreeFall at φ={phi}");
                }
                Err(FlatnessFault::InvertedTilt) => {
                    panic!("pole-safe map must not return InvertedTilt at φ={phi}");
                }
            }
        }
        assert!(all_finite, "non-finite ω somewhere in the pole sweep");
        // ‖dz_b‖ = 1 rad/s by construction, so ‖ω‖ should be ~1 rad/s
        // across the sweep — well under any "diverging" threshold.
        assert!(
            max_norm < 5.0,
            "max omega norm over pole sweep too large: {max_norm}"
        );
    }

    /// Sanity check on the closed-form energy gradient by finite
    /// difference: ∂E/∂c_4 should match the analytical value.
    #[test]
    fn test_energy_grad_by_coeffs_finite_diff() {
        let head: PVAJ3D = [ZERO3, ZERO3, ZERO3, ZERO3];
        let tail: PVAJ3D = [Vec3::new(1.0, 0.5, 0.0), ZERO3, ZERO3, ZERO3];
        let mut solver = MincoSnap::new(&head, &tail, 2);
        let waypoints = [Vec3::new(0.5, 0.25, 0.0)];
        let times = [0.5, 0.5];
        solver.solve(&waypoints, &times);

        let mut grad = vec![Vector3::<f32>::zeros(); 8 * 2];
        solver.add_energy_grad_by_coeffs(&mut grad, 1.0);

        // Energy gradient w.r.t. c_4 of piece 0, x-component, by FD.
        let e0 = solver.get_energy();
        let h = 1e-3f32;
        let saved = solver.b[4];
        solver.b[4] = saved + Vec3::new(h, 0.0, 0.0);
        let e_plus = solver.get_energy();
        solver.b[4] = saved - Vec3::new(h, 0.0, 0.0);
        let e_minus = solver.get_energy();

        let fd = (e_plus - e_minus) / (2.0 * h);
        let analytic = grad[4][0];
        let tol = 1e-1 * analytic.abs().max(1.0);
        assert!(
            (fd - analytic).abs() < tol,
            "FD {fd:.4} vs analytic {analytic:.4} (E0={e0})"
        );
    }
}
