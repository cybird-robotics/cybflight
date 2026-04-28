use nalgebra::Vector3;

/// Maximum banded storage size.
///
/// Sized for the most demanding consumer:
///   - MINCO min-jerk: system size = 6·N, half-bandwidth 6 → 6·N·13 floats.
///   - MINCO min-snap: system size = 8·N, half-bandwidth 8 → 8·N·17 floats.
/// MincoSnap dominates, so we allocate `8 · MAX_PIECES · 17` floats.
const MAX_STORAGE: usize = 8 * super::MAX_PIECES * (8 + 8 + 1);

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
        let used = self.n * (self.lower_bw + self.upper_bw + 1);
        self.data[..used].fill(0.0);
    }

    #[inline(always)]
    fn idx(&self, i: usize, j: usize) -> usize {
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
    pub fn solve3(&self, b: &mut [Vector3<f32>]) {
        let n = self.n;
        // Forward substitution (L).
        for j in 0..n {
            let i_max = (j + self.lower_bw).min(n - 1);
            for i in (j + 1)..=i_max {
                let lij = self.get(i, j);
                let bj = b[j];
                b[i] -= bj * lij;
            }
        }
        // Backward substitution (U).
        for j in (0..n).rev() {
            let inv_diag = 1.0 / self.get(j, j);
            b[j] *= inv_diag;
            let i_min = j.saturating_sub(self.upper_bw);
            for i in i_min..j {
                let uij = self.get(i, j);
                let bj = b[j];
                b[i] -= bj * uij;
            }
        }
    }

    /// Solve A^T x = b in-place (adjoint/transpose solve).
    pub fn solve3_adj(&self, b: &mut [Vector3<f32>]) {
        let n = self.n;
        for j in 0..n {
            let inv_diag = 1.0 / self.get(j, j);
            b[j] *= inv_diag;
            let i_max = (j + self.upper_bw).min(n - 1);
            for i in (j + 1)..=i_max {
                let aji = self.get(j, i);
                let bj = b[j];
                b[i] -= bj * aji;
            }
        }
        for j in (0..n).rev() {
            let i_min = j.saturating_sub(self.lower_bw);
            for i in i_min..j {
                let aji = self.get(j, i);
                let bj = b[j];
                b[i] -= bj * aji;
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

        let mut b = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
        ];
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
                sum += a_dense[i][j] * b[j].x;
            }
            assert!((sum - rhs[i]).abs() < 1e-5, "Row {i}: {sum} != {}", rhs[i]);
        }
    }
}
