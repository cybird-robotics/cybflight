/// Maximum banded storage size.
///
/// Sized for the MINCO min-jerk solver (only consumer): system size = 6·N,
/// upper bandwidth = lower bandwidth = 6, band width = 13 rows in compact
/// storage. Storage = 6 · MAX_PIECES · 13.
const MAX_STORAGE: usize = 6 * super::MAX_PIECES * (6 + 6 + 1);

/// A banded matrix with compact band storage (Golub & Van Loan convention).
///
/// Zero heap allocations. All storage is inline in the struct.
///
/// Element (i,j) is stored at `data[(i - j + q) * N + j]`
/// where q = upper bandwidth.
pub struct BandedSystem {
    n: usize,
    lower_bw: usize,
    upper_bw: usize,
    data: [f32; MAX_STORAGE],
}

impl BandedSystem {
    /// Create an N×N banded system with lower bandwidth `p` and upper bandwidth `q`.
    /// Panics if the required storage exceeds MAX_STORAGE.
    pub fn new(n: usize, p: usize, q: usize) -> Self {
        debug_assert!(
            n * (p + q + 1) <= MAX_STORAGE,
            "BandedSystem: required storage {} exceeds MAX_STORAGE {}",
            n * (p + q + 1),
            MAX_STORAGE
        );
        Self {
            n,
            lower_bw: p,
            upper_bw: q,
            data: [0.0; MAX_STORAGE],
        }
    }

    /// Reset all entries to zero.
    #[inline]
    pub fn reset(&mut self) {
        // Only zero the portion we actually use
        let used = self.n * (self.lower_bw + self.upper_bw + 1);
        self.data[..used].fill(0.0);
    }

    #[inline(always)]
    fn idx(&self, i: usize, j: usize) -> usize {
        // Wrapping arithmetic avoids branch for the unsigned underflow case.
        // This is safe because (i - j + upper_bw) is always non-negative
        // for valid banded indices.
        ((i + self.upper_bw) - j) * self.n + j
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        unsafe { *self.data.get_unchecked(self.idx(i, j)) }
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, val: f32) {
        let idx = self.idx(i, j);
        unsafe {
            *self.data.get_unchecked_mut(idx) = val;
        }
    }

    #[inline(always)]
    fn get_mut_ref(&mut self, i: usize, j: usize) -> &mut f32 {
        let idx = self.idx(i, j);
        unsafe { self.data.get_unchecked_mut(idx) }
    }

    /// In-place banded LU factorization without pivoting.
    ///
    /// Tiny pivots are clamped to ±1e-6 to prevent NaN/Inf propagation
    /// from near-singular systems.
    ///
    /// Inner loops are unconditional multiply-adds: for the MINCO-jerk
    /// structure, the band is structurally dense, so adding `!= 0.0`
    /// guards costs more in branch mispredicts than it saves in skipped FMAs.
    pub fn factorize_lu(&mut self) {
        let n = self.n;
        for k in 0..n - 1 {
            let i_max = (k + self.lower_bw).min(n - 1);
            let raw_pivot = self.get(k, k);
            let pivot = if raw_pivot.abs() < 1e-6 {
                if raw_pivot >= 0.0 {
                    1e-6
                } else {
                    -1e-6
                }
            } else {
                raw_pivot
            };
            let inv_pivot = 1.0 / pivot;
            for i in (k + 1)..=i_max {
                *self.get_mut_ref(i, k) *= inv_pivot;
            }
            let j_max = (k + self.upper_bw).min(n - 1);
            for j in (k + 1)..=j_max {
                let c = self.get(k, j);
                for i in (k + 1)..=i_max {
                    let lik = self.get(i, k);
                    *self.get_mut_ref(i, j) -= lik * c;
                }
            }
        }
    }

    /// Solve Ax = b in-place where b has 3 columns (x/y/z).
    ///
    /// `b` is a slice of `[f32; 3]` with length N.
    /// After solve, `b` contains the solution x.
    ///
    /// This operates directly on the [f32; 3] rows, avoiding
    /// the need to flatten/unflatten between the solver and the caller.
    pub fn solve3(&self, b: &mut [[f32; 3]]) {
        let n = self.n;
        // Forward substitution (L). Inner FMA is unconditional — see the
        // comment on `factorize_lu` for why we drop the zero-skip.
        for j in 0..n {
            let i_max = (j + self.lower_bw).min(n - 1);
            for i in (j + 1)..=i_max {
                let lij = self.get(i, j);
                let bj = b[j];
                let bi = &mut b[i];
                bi[0] -= lij * bj[0];
                bi[1] -= lij * bj[1];
                bi[2] -= lij * bj[2];
            }
        }
        // Backward substitution (U).
        for j in (0..n).rev() {
            let inv_diag = 1.0 / self.get(j, j);
            b[j][0] *= inv_diag;
            b[j][1] *= inv_diag;
            b[j][2] *= inv_diag;
            let i_min = j.saturating_sub(self.upper_bw);
            for i in i_min..j {
                let uij = self.get(i, j);
                let bj = b[j];
                let bi = &mut b[i];
                bi[0] -= uij * bj[0];
                bi[1] -= uij * bj[1];
                bi[2] -= uij * bj[2];
            }
        }
    }

    /// Solve A^T x = b in-place (adjoint/transpose solve).
    /// Used for gradient backpropagation through the MINCO system.
    pub fn solve3_adj(&self, b: &mut [[f32; 3]]) {
        let n = self.n;
        // Forward pass: solve U^T part.
        for j in 0..n {
            let inv_diag = 1.0 / self.get(j, j);
            b[j][0] *= inv_diag;
            b[j][1] *= inv_diag;
            b[j][2] *= inv_diag;
            let i_max = (j + self.upper_bw).min(n - 1);
            for i in (j + 1)..=i_max {
                let aji = self.get(j, i);
                let bj = b[j];
                let bi = &mut b[i];
                bi[0] -= aji * bj[0];
                bi[1] -= aji * bj[1];
                bi[2] -= aji * bj[2];
            }
        }
        // Backward pass: solve L^T part.
        for j in (0..n).rev() {
            let i_min = j.saturating_sub(self.lower_bw);
            for i in i_min..j {
                let aji = self.get(j, i);
                let bj = b[j];
                let bi = &mut b[i];
                bi[0] -= aji * bj[0];
                bi[1] -= aji * bj[1];
                bi[2] -= aji * bj[2];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tridiagonal() {
        let mut a = BandedSystem::new(3, 1, 1);
        a.set(0, 0, 2.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 3.0);
        a.set(1, 2, 1.0);
        a.set(2, 1, 1.0);
        a.set(2, 2, 2.0);

        a.factorize_lu();

        let mut b: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        a.solve3(&mut b);

        let a_dense = [
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ];
        let rhs = [1.0, 2.0, 3.0];
        for i in 0..3 {
            let mut sum = 0.0;
            for j in 0..3 {
                sum += a_dense[i][j] * b[j][0];
            }
            assert!((sum - rhs[i]).abs() < 1e-5, "Row {i}: {sum} != {}", rhs[i]);
        }
    }
}
