use super::banded_system::BandedSystem;
use super::piecewise_polynomial::PiecewisePolynomial;
use super::polynomial::Polynomial;
use super::types::{dot3, norm_sq3, Vec3, PVA3D};
use super::MAX_PIECES;

/// MINCO min-jerk trajectory solver (polynomial degree 5, s=3).
/// Zero heap allocations — all buffers are inline fixed-size arrays.
pub struct MincoJerk {
    n: usize,
    head_pva: PVA3D,
    tail_pva: PVA3D,
    banded: BandedSystem,
    /// Coefficient matrix: 6N rows × 3 cols
    b: [[f32; 3]; 6 * MAX_PIECES],
    t1: [f32; MAX_PIECES],
    t2: [f32; MAX_PIECES],
    t3: [f32; MAX_PIECES],
    t4: [f32; MAX_PIECES],
    t5: [f32; MAX_PIECES],
}

impl MincoJerk {
    /// Initialize the solver.
    /// - `head_state`: [pos, vel, acc] at start
    /// - `tail_state`: [pos, vel, acc] at end
    /// - `piece_num`: number of polynomial pieces (N), must be ≤ MAX_PIECES
    pub fn new(head_state: &PVA3D, tail_state: &PVA3D, piece_num: usize) -> Self {
        debug_assert!(piece_num >= 1 && piece_num <= MAX_PIECES);
        Self {
            n: piece_num,
            head_pva: *head_state,
            tail_pva: *tail_state,
            banded: BandedSystem::new(6 * piece_num, 6, 6),
            b: [[0.0; 3]; 6 * MAX_PIECES],
            t1: [0.0; MAX_PIECES],
            t2: [0.0; MAX_PIECES],
            t3: [0.0; MAX_PIECES],
            t4: [0.0; MAX_PIECES],
            t5: [0.0; MAX_PIECES],
        }
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
        self.b[..sys_size].fill([0.0; 3]);

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

            // Jerk continuity
            self.banded.set(base + 3, base + 3, 6.0);
            self.banded.set(base + 3, base + 4, 24.0 * t1);
            self.banded.set(base + 3, base + 5, 60.0 * t2);
            self.banded.set(base + 3, base + 9, -6.0);

            // Snap continuity
            self.banded.set(base + 4, base + 4, 24.0);
            self.banded.set(base + 4, base + 5, 120.0 * t1);
            self.banded.set(base + 4, base + 10, -24.0);

            // Position at end = waypoint
            self.banded.set(base + 5, base + 0, 1.0);
            self.banded.set(base + 5, base + 1, t1);
            self.banded.set(base + 5, base + 2, t2);
            self.banded.set(base + 5, base + 3, t3);
            self.banded.set(base + 5, base + 4, t4);
            self.banded.set(base + 5, base + 5, t5);

            // Position continuity
            self.banded.set(base + 6, base + 0, 1.0);
            self.banded.set(base + 6, base + 1, t1);
            self.banded.set(base + 6, base + 2, t2);
            self.banded.set(base + 6, base + 3, t3);
            self.banded.set(base + 6, base + 4, t4);
            self.banded.set(base + 6, base + 5, t5);
            self.banded.set(base + 6, base + 6, -1.0);

            // Velocity continuity
            self.banded.set(base + 7, base + 1, 1.0);
            self.banded.set(base + 7, base + 2, 2.0 * t1);
            self.banded.set(base + 7, base + 3, 3.0 * t2);
            self.banded.set(base + 7, base + 4, 4.0 * t3);
            self.banded.set(base + 7, base + 5, 5.0 * t4);
            self.banded.set(base + 7, base + 7, -1.0);

            // Acceleration continuity
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
            coeffs: [[0.0; 3]; super::polynomial::MAX_COEFFS],
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

            energy += 36.0 * norm_sq3(b3) * self.t1[i]
                + 144.0 * dot3(b3, b4) * self.t2[i]
                + (240.0 * dot3(b3, b5) + 192.0 * norm_sq3(b4)) * self.t3[i]
                + 720.0 * dot3(b4, b5) * self.t4[i]
                + 720.0 * norm_sq3(b5) * self.t5[i];
        }
        energy
    }

    /// Access the raw coefficient for piece `piece_idx`, coefficient index `coeff_idx`, dimension `dim`.
    #[inline]
    pub fn get_coeff(&self, piece_idx: usize, coeff_idx: usize, dim: usize) -> f32 {
        self.b[6 * piece_idx + coeff_idx][dim]
    }

    /// Compute partial gradients of jerk energy w.r.t. polynomial coefficients.
    /// `grad_c` must have length 6*N.
    pub fn get_energy_partial_grad_by_coeffs(&self, grad_c: &mut [[f32; 3]]) {
        for i in 0..self.n {
            let base = 6 * i;
            let b3 = self.b[base + 3];
            let b4 = self.b[base + 4];
            let b5 = self.b[base + 5];
            let t1 = self.t1[i]; let t2 = self.t2[i]; let t3 = self.t3[i];
            let t4 = self.t4[i]; let t5 = self.t5[i];
            for d in 0..3 {
                grad_c[base + 5][d] = 240.0*b3[d]*t3 + 720.0*b4[d]*t4 + 1440.0*b5[d]*t5;
                grad_c[base + 4][d] = 144.0*b3[d]*t2 + 384.0*b4[d]*t3 + 720.0*b5[d]*t4;
                grad_c[base + 3][d] = 72.0*b3[d]*t1 + 144.0*b4[d]*t2 + 240.0*b5[d]*t3;
                grad_c[base + 0][d] = 0.0;
                grad_c[base + 1][d] = 0.0;
                grad_c[base + 2][d] = 0.0;
            }
        }
    }

    /// Compute partial gradients of jerk energy w.r.t. times.
    /// `grad_t` must have length N.
    pub fn get_energy_partial_grad_by_times(&self, grad_t: &mut [f32]) {
        for i in 0..self.n {
            let base = 6 * i;
            let b3 = self.b[base + 3];
            let b4 = self.b[base + 4];
            let b5 = self.b[base + 5];
            let t1 = self.t1[i]; let t2 = self.t2[i]; let t3 = self.t3[i];
            let t4 = self.t4[i];
            grad_t[i] = 36.0 * norm_sq3(b3)
                + 288.0 * dot3(b3, b4) * t1
                + (720.0 * dot3(b3, b5) + 576.0 * norm_sq3(b4)) * t2
                + 2880.0 * dot3(b4, b5) * t3
                + 3600.0 * norm_sq3(b5) * t4;
        }
    }

    /// Backpropagate gradients through the MINCO jerk system.
    ///
    /// - `partial_grad_c`: 6N×3, partial gradient w.r.t. polynomial coefficients
    /// - `partial_grad_t`: N, partial gradient w.r.t. times
    /// - `grad_points`: output, (N-1)×3 gradient w.r.t. waypoint positions
    /// - `grad_times`: output, N gradient w.r.t. times
    pub fn propagate_grad(
        &self,
        partial_grad_c: &[[f32; 3]],
        partial_grad_t: &[f32],
        grad_points: &mut [[f32; 3]],
        grad_times: &mut [f32],
    ) {
        let n = self.n;
        let sys_size = 6 * n;

        for gp in grad_points.iter_mut() { *gp = [0.0; 3]; }
        for gt in grad_times.iter_mut() { *gt = 0.0; }

        // Solve A^T * adjGrad = partial_grad_c
        let mut adj_grad = [[0.0f32; 3]; 6 * MAX_PIECES];
        adj_grad[..sys_size].copy_from_slice(&partial_grad_c[..sys_size]);
        self.banded.solve3_adj(&mut adj_grad[..sys_size]);

        // Extract gradient w.r.t. waypoints from position constraint rows
        // In MincoJerk, the position constraint row for waypoint i is at row 6*i + 5
        for i in 0..(n - 1) {
            grad_points[i] = adj_grad[6 * i + 5];
        }

        // Compute gradient w.r.t. times via ∂A/∂T for interior segments.
        //
        // A(T)·b = boundary. The T-dependent rows for segment i's end (rows
        // 6i+3 .. 6i+8) evaluate derivatives of the piece polynomial at
        // t = T_i. ∂A/∂T applied to b raises each derivative order by one:
        //
        //   row 6i+3 (jerk continuity, p''')  →  ∂/∂T = p'''' = snap at end
        //   row 6i+4 (snap continuity, p'''') →  ∂/∂T = p''''' = crackle at end
        //   row 6i+5 (pos = waypoint,   p)    →  ∂/∂T = p'     = velocity at end
        //   row 6i+6 (pos continuity,   p)    →  ∂/∂T = p'     = velocity at end
        //   row 6i+7 (vel continuity,   p')   →  ∂/∂T = p''    = acceleration at end
        //   row 6i+8 (acc continuity,   p'')  →  ∂/∂T = p'''   = jerk at end
        //
        // gradT_i = -adjGrad · (∂A/∂T · b), so we store negatives in b1.
        for i in 0..(n - 1) {
            let o = i * 6;
            let t1 = self.t1[i]; let t2 = self.t2[i]; let t3 = self.t3[i];
            let t4 = self.t4[i];

            let mut b1 = [[0.0f32; 3]; 6];

            // k=0: jerk continuity row → negative snap at end (24 b4 + 120 T b5)
            b1[0] = neg3(add_scaled_rows(&self.b, o, &[
                (4, 24.0), (5, 120.0 * t1)
            ]));

            // k=1: snap continuity row → negative crackle at end (120 b5)
            b1[1] = neg3(add_scaled_rows(&self.b, o, &[
                (5, 120.0)
            ]));

            // k=2, k=3: pos=waypoint and pos continuity → negative velocity at end
            let neg_vel = neg3(add_scaled_rows(&self.b, o, &[
                (1, 1.0), (2, 2.0*t1), (3, 3.0*t2), (4, 4.0*t3), (5, 5.0*t4)
            ]));
            b1[2] = neg_vel;
            b1[3] = neg_vel;

            // k=4: velocity continuity row → negative acceleration at end
            b1[4] = neg3(add_scaled_rows(&self.b, o, &[
                (2, 2.0), (3, 6.0*t1), (4, 12.0*t2), (5, 20.0*t3)
            ]));

            // k=5: acceleration continuity row → negative jerk at end
            b1[5] = neg3(add_scaled_rows(&self.b, o, &[
                (3, 6.0), (4, 24.0*t1), (5, 60.0*t2)
            ]));

            // gradByTimes(i) = B1 . adjGrad[6i+3 .. 6i+9]
            let mut sum = 0.0;
            for k in 0..6 {
                sum += dot3(b1[k], adj_grad[6 * i + 3 + k]);
            }
            grad_times[i] = sum;
        }

        // Last segment tail boundary
        {
            let last = n - 1;
            let o = last * 6;
            let t1 = self.t1[last]; let t2 = self.t2[last]; let t3 = self.t3[last];
            let t4 = self.t4[last];

            let mut b2 = [[0.0f32; 3]; 3];

            // negative velocity
            b2[0] = neg3(add_scaled_rows(&self.b, o, &[
                (1, 1.0), (2, 2.0*t1), (3, 3.0*t2), (4, 4.0*t3), (5, 5.0*t4)
            ]));

            // negative acceleration
            b2[1] = neg3(add_scaled_rows(&self.b, o, &[
                (2, 2.0), (3, 6.0*t1), (4, 12.0*t2), (5, 20.0*t3)
            ]));

            // negative jerk
            b2[2] = neg3(add_scaled_rows(&self.b, o, &[
                (3, 6.0), (4, 24.0*t1), (5, 60.0*t2)
            ]));

            let mut sum = 0.0;
            for k in 0..3 {
                sum += dot3(b2[k], adj_grad[6 * n - 3 + k]);
            }
            grad_times[last] = sum;
        }

        // Add partial_grad_t
        for i in 0..n {
            grad_times[i] += partial_grad_t[i];
        }
    }
}

/// Helper: compute sum of scaled coefficient rows.
#[inline]
fn add_scaled_rows(b: &[[f32; 3]], base: usize, terms: &[(usize, f32)]) -> [f32; 3] {
    let mut r = [0.0f32; 3];
    for &(offset, scale) in terms {
        let row = b[base + offset];
        r[0] += scale * row[0];
        r[1] += scale * row[1];
        r[2] += scale * row[2];
    }
    r
}

#[inline]
fn neg3(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_piece_min_jerk() {
        let head: PVA3D = [[0.0; 3], [0.0; 3], [0.0; 3]];
        let tail: PVA3D = [[1.0, 0.0, 0.0], [0.0; 3], [0.0; 3]];

        let mut solver = MincoJerk::new(&head, &tail, 1);
        solver.solve(&[], &[1.0]);
        let traj = solver.get_trajectory();

        let p0 = traj.get_pos(0.0);
        let p1 = traj.get_pos(1.0);
        assert!((p0[0]).abs() < 1e-4);
        assert!((p1[0] - 1.0).abs() < 1e-4);
        assert!(norm_sq3(traj.get_vel(0.0)) < 1e-10);
        assert!(norm_sq3(traj.get_vel(1.0)) < 1e-10);
    }
}
