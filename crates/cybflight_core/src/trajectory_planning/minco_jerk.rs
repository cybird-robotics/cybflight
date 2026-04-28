use nalgebra::Vector3;

use super::banded_system::BandedSystem;
use super::piecewise_polynomial::PiecewisePolynomial;
use super::polynomial::Polynomial;
use super::types::{Vec3, ZERO3, PVA3D};
use super::MAX_PIECES;

/// MINCO min-jerk trajectory solver (polynomial degree 5, s=3).
/// Zero heap allocations — all buffers are inline fixed-size arrays.
pub struct MincoJerk {
    n: usize,
    head_pva: PVA3D,
    tail_pva: PVA3D,
    banded: BandedSystem,
    /// Coefficient matrix: 6N rows × 3 cols
    b: [Vector3<f32>; 6 * MAX_PIECES],
    t1: [f32; MAX_PIECES],
    t2: [f32; MAX_PIECES],
    t3: [f32; MAX_PIECES],
    t4: [f32; MAX_PIECES],
    t5: [f32; MAX_PIECES],
}

impl MincoJerk {
    /// Initialize the solver.
    pub fn new(head_state: &PVA3D, tail_state: &PVA3D, piece_num: usize) -> Self {
        debug_assert!(piece_num >= 1 && piece_num <= MAX_PIECES);
        Self {
            n: piece_num,
            head_pva: *head_state,
            tail_pva: *tail_state,
            banded: BandedSystem::new(6 * piece_num, 6, 6),
            b: [Vector3::zeros(); 6 * MAX_PIECES],
            t1: [0.0; MAX_PIECES],
            t2: [0.0; MAX_PIECES],
            t3: [0.0; MAX_PIECES],
            t4: [0.0; MAX_PIECES],
            t5: [0.0; MAX_PIECES],
        }
    }

    /// Update the boundary states (head/tail PVA) in place. Lets a
    /// caller reuse a single BSS-resident `MincoJerk` across solves
    /// with different start/end conditions without reconstructing
    /// (the construction allocates a 6N-row banded storage block, so
    /// holding one in `StaticCell` and just updating the boundary is
    /// significantly cheaper than building a fresh solver per call).
    pub fn set_boundary(&mut self, head_state: &PVA3D, tail_state: &PVA3D) {
        self.head_pva = *head_state;
        self.tail_pva = *tail_state;
    }

    /// Set waypoints and time allocation, then solve for coefficients.
    pub fn solve(&mut self, waypoints: &[Vec3], times: &[f32]) {
        debug_assert_eq!(times.len(), self.n);
        debug_assert_eq!(waypoints.len(), self.n - 1);

        for i in 0..self.n {
            let t = times[i];
            self.t1[i] = t;
            let t2 = t * t;
            self.t2[i] = t2;
            self.t3[i] = t2 * t;
            let t4 = t2 * t2;
            self.t4[i] = t4;
            self.t5[i] = t4 * t;
        }

        self.banded.reset();
        let sys_size = 6 * self.n;
        self.b[..sys_size].fill(Vector3::zeros());

        // Head boundary
        self.banded.set(0, 0, 1.0);
        self.banded.set(1, 1, 1.0);
        self.banded.set(2, 2, 2.0);

        self.b[0] = self.head_pva[0];
        self.b[1] = self.head_pva[1];
        self.b[2] = self.head_pva[2];

        // Intermediate junctions
        for i in 0..(self.n - 1) {
            let base = 6 * i;
            let t1 = self.t1[i];
            let t2 = self.t2[i];
            let t3 = self.t3[i];
            let t4 = self.t4[i];
            let t5 = self.t5[i];

            self.banded.set(base + 3, base + 3, 6.0);
            self.banded.set(base + 3, base + 4, 24.0 * t1);
            self.banded.set(base + 3, base + 5, 60.0 * t2);
            self.banded.set(base + 3, base + 9, -6.0);

            self.banded.set(base + 4, base + 4, 24.0);
            self.banded.set(base + 4, base + 5, 120.0 * t1);
            self.banded.set(base + 4, base + 10, -24.0);

            self.banded.set(base + 5, base + 0, 1.0);
            self.banded.set(base + 5, base + 1, t1);
            self.banded.set(base + 5, base + 2, t2);
            self.banded.set(base + 5, base + 3, t3);
            self.banded.set(base + 5, base + 4, t4);
            self.banded.set(base + 5, base + 5, t5);

            self.banded.set(base + 6, base + 0, 1.0);
            self.banded.set(base + 6, base + 1, t1);
            self.banded.set(base + 6, base + 2, t2);
            self.banded.set(base + 6, base + 3, t3);
            self.banded.set(base + 6, base + 4, t4);
            self.banded.set(base + 6, base + 5, t5);
            self.banded.set(base + 6, base + 6, -1.0);

            self.banded.set(base + 7, base + 1, 1.0);
            self.banded.set(base + 7, base + 2, 2.0 * t1);
            self.banded.set(base + 7, base + 3, 3.0 * t2);
            self.banded.set(base + 7, base + 4, 4.0 * t3);
            self.banded.set(base + 7, base + 5, 5.0 * t4);
            self.banded.set(base + 7, base + 7, -1.0);

            self.banded.set(base + 8, base + 2, 2.0);
            self.banded.set(base + 8, base + 3, 6.0 * t1);
            self.banded.set(base + 8, base + 4, 12.0 * t2);
            self.banded.set(base + 8, base + 5, 20.0 * t3);
            self.banded.set(base + 8, base + 8, -2.0);

            self.b[base + 5] = waypoints[i];
        }

        // Tail boundary
        let n6 = 6 * self.n;
        let last = self.n - 1;
        let t1 = self.t1[last];
        let t2 = self.t2[last];
        let t3 = self.t3[last];
        let t4 = self.t4[last];
        let t5 = self.t5[last];

        self.banded.set(n6 - 3, n6 - 6, 1.0);
        self.banded.set(n6 - 3, n6 - 5, t1);
        self.banded.set(n6 - 3, n6 - 4, t2);
        self.banded.set(n6 - 3, n6 - 3, t3);
        self.banded.set(n6 - 3, n6 - 2, t4);
        self.banded.set(n6 - 3, n6 - 1, t5);

        self.banded.set(n6 - 2, n6 - 5, 1.0);
        self.banded.set(n6 - 2, n6 - 4, 2.0 * t1);
        self.banded.set(n6 - 2, n6 - 3, 3.0 * t2);
        self.banded.set(n6 - 2, n6 - 2, 4.0 * t3);
        self.banded.set(n6 - 2, n6 - 1, 5.0 * t4);

        self.banded.set(n6 - 1, n6 - 4, 2.0);
        self.banded.set(n6 - 1, n6 - 3, 6.0 * t1);
        self.banded.set(n6 - 1, n6 - 2, 12.0 * t2);
        self.banded.set(n6 - 1, n6 - 1, 20.0 * t3);

        self.b[n6 - 3] = self.tail_pva[0];
        self.b[n6 - 2] = self.tail_pva[1];
        self.b[n6 - 1] = self.tail_pva[2];

        // === Solve directly on b ===
        self.banded.factorize_lu();
        self.banded.solve3(&mut self.b[..sys_size]);
    }

    /// Extract the solved trajectory as a PiecewisePolynomial (degree 5).
    pub fn get_trajectory(&self) -> PiecewisePolynomial {
        let mut pieces = [Polynomial {
            degree: 0,
            duration: 0.0,
            coeffs: [ZERO3; super::polynomial::MAX_COEFFS],
        }; MAX_PIECES];

        for i in 0..self.n {
            let base = 6 * i;
            pieces[i] = Polynomial::new(5, self.t1[i], &self.b[base..base + 6]);
        }
        PiecewisePolynomial::from_pieces(&pieces[..self.n])
    }

    /// Compute the jerk energy: ∫ ‖jerk(t)‖² dt over all pieces.
    pub fn get_energy(&self) -> f32 {
        let mut energy = 0.0;
        for i in 0..self.n {
            let base = 6 * i;
            let b3 = self.b[base + 3];
            let b4 = self.b[base + 4];
            let b5 = self.b[base + 5];

            energy += 36.0 * b3.norm_squared() * self.t1[i]
                + 144.0 * b3.dot(&b4) * self.t2[i]
                + (240.0 * b3.dot(&b5) + 192.0 * b4.norm_squared()) * self.t3[i]
                + 720.0 * b4.dot(&b5) * self.t4[i]
                + 720.0 * b5.norm_squared() * self.t5[i];
        }
        energy
    }

    /// Access the raw coefficient for piece `piece_idx`, coefficient index `coeff_idx`, dimension `dim`.
    #[inline]
    pub fn get_coeff(&self, piece_idx: usize, coeff_idx: usize, dim: usize) -> f32 {
        self.b[6 * piece_idx + coeff_idx][dim]
    }

    /// Borrow the 6 polynomial coefficients for `piece_idx` in ascending order.
    #[inline]
    pub fn piece_coeffs(&self, piece_idx: usize) -> &[Vector3<f32>] {
        let base = 6 * piece_idx;
        &self.b[base..base + 6]
    }

    /// Accumulate `scale · ∂E/∂coeffs` directly into `grad_c` (length 6·N).
    pub fn add_energy_grad_by_coeffs(&self, grad_c: &mut [Vector3<f32>], scale: f32) {
        for i in 0..self.n {
            let base = 6 * i;
            let b3 = self.b[base + 3];
            let b4 = self.b[base + 4];
            let b5 = self.b[base + 5];
            let t1 = self.t1[i];
            let t2 = self.t2[i];
            let t3 = self.t3[i];
            let t4 = self.t4[i];
            let t5 = self.t5[i];
            grad_c[base + 3] +=
                (b3 * (72.0 * t1) + b4 * (144.0 * t2) + b5 * (240.0 * t3)) * scale;
            grad_c[base + 4] +=
                (b3 * (144.0 * t2) + b4 * (384.0 * t3) + b5 * (720.0 * t4)) * scale;
            grad_c[base + 5] +=
                (b3 * (240.0 * t3) + b4 * (720.0 * t4) + b5 * (1440.0 * t5)) * scale;
            // rows 0,1,2 unchanged (energy is independent of them).
        }
    }

    /// Accumulate `scale · ∂E/∂times` directly into `grad_t` (length N).
    pub fn add_energy_grad_by_times(&self, grad_t: &mut [f32], scale: f32) {
        for i in 0..self.n {
            let base = 6 * i;
            let b3 = self.b[base + 3];
            let b4 = self.b[base + 4];
            let b5 = self.b[base + 5];
            let t1 = self.t1[i];
            let t2 = self.t2[i];
            let t3 = self.t3[i];
            let t4 = self.t4[i];
            grad_t[i] += scale
                * (36.0 * b3.norm_squared()
                    + 288.0 * b3.dot(&b4) * t1
                    + (720.0 * b3.dot(&b5) + 576.0 * b4.norm_squared()) * t2
                    + 2880.0 * b4.dot(&b5) * t3
                    + 3600.0 * b5.norm_squared() * t4);
        }
    }

    /// Backpropagate gradients through the MINCO jerk system.
    pub fn propagate_grad(
        &self,
        partial_grad_c: &[Vector3<f32>],
        partial_grad_t: &[f32],
        grad_points: &mut [Vector3<f32>],
        grad_times: &mut [f32],
    ) {
        let n = self.n;
        let sys_size = 6 * n;

        for gp in grad_points.iter_mut() { *gp = Vector3::zeros(); }
        for gt in grad_times.iter_mut() { *gt = 0.0; }

        // Solve A^T * adjGrad = partial_grad_c
        let mut adj_grad = [Vector3::<f32>::zeros(); 6 * MAX_PIECES];
        adj_grad[..sys_size].copy_from_slice(&partial_grad_c[..sys_size]);
        self.banded.solve3_adj(&mut adj_grad[..sys_size]);

        // Extract gradient w.r.t. waypoints from position constraint rows.
        for i in 0..(n - 1) {
            grad_points[i] = adj_grad[6 * i + 5];
        }

        // Compute gradient w.r.t. times via ∂A/∂T for interior segments.
        for i in 0..(n - 1) {
            let o = i * 6;
            let t1 = self.t1[i]; let t2 = self.t2[i]; let t3 = self.t3[i];
            let t4 = self.t4[i];

            // k=0: jerk continuity row → negative snap at end
            let b1_0 = -(self.b[o + 4] * 24.0 + self.b[o + 5] * (120.0 * t1));
            // k=1: snap continuity row → negative crackle at end
            let b1_1 = -(self.b[o + 5] * 120.0);
            // k=2, k=3: pos continuity → negative velocity at end
            let neg_vel = -(self.b[o + 1]
                + self.b[o + 2] * (2.0 * t1)
                + self.b[o + 3] * (3.0 * t2)
                + self.b[o + 4] * (4.0 * t3)
                + self.b[o + 5] * (5.0 * t4));
            let b1_2 = neg_vel;
            let b1_3 = neg_vel;
            // k=4: velocity continuity → negative acceleration at end
            let b1_4 = -(self.b[o + 2] * 2.0
                + self.b[o + 3] * (6.0 * t1)
                + self.b[o + 4] * (12.0 * t2)
                + self.b[o + 5] * (20.0 * t3));
            // k=5: acceleration continuity → negative jerk at end
            let b1_5 = -(self.b[o + 3] * 6.0
                + self.b[o + 4] * (24.0 * t1)
                + self.b[o + 5] * (60.0 * t2));

            let sum = b1_0.dot(&adj_grad[6 * i + 3])
                + b1_1.dot(&adj_grad[6 * i + 4])
                + b1_2.dot(&adj_grad[6 * i + 5])
                + b1_3.dot(&adj_grad[6 * i + 6])
                + b1_4.dot(&adj_grad[6 * i + 7])
                + b1_5.dot(&adj_grad[6 * i + 8]);
            grad_times[i] = sum;
        }

        // Last segment tail boundary
        {
            let last = n - 1;
            let o = last * 6;
            let t1 = self.t1[last]; let t2 = self.t2[last]; let t3 = self.t3[last];
            let t4 = self.t4[last];

            let neg_vel = -(self.b[o + 1]
                + self.b[o + 2] * (2.0 * t1)
                + self.b[o + 3] * (3.0 * t2)
                + self.b[o + 4] * (4.0 * t3)
                + self.b[o + 5] * (5.0 * t4));
            let neg_acc = -(self.b[o + 2] * 2.0
                + self.b[o + 3] * (6.0 * t1)
                + self.b[o + 4] * (12.0 * t2)
                + self.b[o + 5] * (20.0 * t3));
            let neg_jerk = -(self.b[o + 3] * 6.0
                + self.b[o + 4] * (24.0 * t1)
                + self.b[o + 5] * (60.0 * t2));

            let sum = neg_vel.dot(&adj_grad[6 * n - 3])
                + neg_acc.dot(&adj_grad[6 * n - 2])
                + neg_jerk.dot(&adj_grad[6 * n - 1]);
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

    #[test]
    fn test_single_piece_min_jerk() {
        let head: PVA3D = [ZERO3, ZERO3, ZERO3];
        let tail: PVA3D = [Vec3::new(1.0, 0.0, 0.0), ZERO3, ZERO3];

        let mut solver = MincoJerk::new(&head, &tail, 1);
        solver.solve(&[], &[1.0]);
        let traj = solver.get_trajectory();

        let p0 = traj.get_pos(0.0);
        let p1 = traj.get_pos(1.0);
        assert!((p0[0]).abs() < 1e-4);
        assert!((p1[0] - 1.0).abs() < 1e-4);
        let v0 = traj.get_vel(0.0);
        let v1 = traj.get_vel(1.0);
        assert!(v0[0].abs() + v0[1].abs() + v0[2].abs() < 1e-4);
        assert!(v1[0].abs() + v1[1].abs() + v1[2].abs() < 1e-4);
    }
}
