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
//! [`MincoSnap`] (the instance at [`MAX_PIECES`]) carries ~84 KiB of
//! inline storage (banded buffer + coefficient rows + seven time
//! tables); size a [`MincoSnapN`] to the pieces you actually need.
//! `MincoSnap::new` returns by value with no NRVO guarantee, so a
//! stack-allocated `let mut s = MincoSnap::new(...)` may briefly
//! materialize the full struct on the caller's stack. On Embassy
//! task stacks (typically 4–16 KB) this overflows; firmware callers
//! must hold the solver in a `StaticCell` (BSS-resident), exactly as
//! `OFFLINE_MINCO` does in `cybflight::control::mission_planner`.
//! Host-side use (tests, sim, planning utilities) is unaffected.

use nalgebra::Vector3;

use super::banded_system::{snap_storage, BandedSystem};
use super::piecewise_polynomial::PiecewisePolynomial;
use super::polynomial::Polynomial;
use super::types::{Vec3, ZERO3, PVAJ3D};
use super::MAX_PIECES;

/// MINCO min-snap solver (polynomial degree 7, smoothness s=4), sized
/// for at most `P` pieces.
///
/// Zero heap allocations — all buffers are inline fixed-size arrays.
/// The active piece count is tracked in `self.n`; only the leading
/// `8·n` entries of every buffer are live per solve.
///
/// `C` must equal `8 * P` and `S` must be at least [`snap_storage`]`(P)`
/// — see [`super::minco_jerk::MincoJerkN`] for why the bound is spelled
/// three times. Use a type alias for each concrete instance; [`MincoSnap`]
/// is the one at the global cap.
pub struct MincoSnapN<const P: usize, const C: usize, const S: usize> {
    n: usize,
    head_pvaj: PVAJ3D,
    tail_pvaj: PVAJ3D,
    banded: BandedSystem<S>,
    /// Coefficient matrix: 8N rows × 3 cols (one Vec3 per row).
    b: [Vector3<f32>; C],
    t1: [f32; P],
    t2: [f32; P],
    t3: [f32; P],
    t4: [f32; P],
    t5: [f32; P],
    t6: [f32; P],
    t7: [f32; P],
}

/// [`MincoSnapN`] at the global trajectory cap [`MAX_PIECES`] (~84 KiB).
pub type MincoSnap = MincoSnapN<MAX_PIECES, { 8 * MAX_PIECES }, { snap_storage(MAX_PIECES) }>;

impl<const P: usize, const C: usize, const S: usize> MincoSnapN<P, C, S> {
    /// Compile-time check that `C` and `S` match `P`; evaluated by `new`.
    const LAYOUT_OK: () = assert!(
        P >= 1 && P <= MAX_PIECES && C == 8 * P && S >= snap_storage(P),
        "MincoSnapN<P, C, S>: C must be 8*P and S >= snap_storage(P)"
    );

    /// Initialize the solver with placeholder boundary states. Reuse
    /// across solves with [`set_boundary`](Self::set_boundary) —
    /// constructing a fresh solver zero-initializes all of its inline
    /// storage (the banded buffer alone is `8·P·17` floats), so holding
    /// one in a `StaticCell` is significantly cheaper than building a
    /// new one per call. Panics if `piece_num` exceeds `P`.
    pub fn new(head_state: &PVAJ3D, tail_state: &PVAJ3D, piece_num: usize) -> Self {
        let () = Self::LAYOUT_OK;
        assert!(
            piece_num >= 1 && piece_num <= P,
            "MincoSnapN: {piece_num} pieces exceeds the type's bound of {P}"
        );
        Self {
            n: piece_num,
            head_pvaj: *head_state,
            tail_pvaj: *tail_state,
            banded: BandedSystem::new(8 * piece_num, 8, 8),
            b: [Vector3::zeros(); C],
            t1: [0.0; P],
            t2: [0.0; P],
            t3: [0.0; P],
            t4: [0.0; P],
            t5: [0.0; P],
            t6: [0.0; P],
            t7: [0.0; P],
        }
    }

    /// Update the head/tail PVAJ in place. Mirror of
    /// [`super::minco_jerk::MincoJerk::set_boundary`].
    pub fn set_boundary(&mut self, head_state: &PVAJ3D, tail_state: &PVAJ3D) {
        self.head_pvaj = *head_state;
        self.tail_pvaj = *tail_state;
    }

    /// Reconfigure the active piece count in place. All inline buffers
    /// are sized to `P` regardless, so this only updates the active
    /// extent and the underlying banded system's dimension. `solve()`
    /// must be called before any subsequent read; this leaves the
    /// buffers in an unspecified state. Panics if `piece_num` exceeds `P`.
    pub fn set_piece_count(&mut self, piece_num: usize) {
        assert!(
            piece_num >= 1 && piece_num <= P,
            "MincoSnapN: {piece_num} pieces exceeds the type's bound of {P}"
        );
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
        }; P];

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
        let mut adj_grad = [Vector3::<f32>::zeros(); C];
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
