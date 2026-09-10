use nalgebra::Vector3;

/// Band storage (in floats) for a MINCO system of order `s` over `pieces`
/// polynomial pieces.
///
/// A MINCO of order `s` (s=2 acc, s=3 jerk, s=4 snap) has `2s` coefficients
/// per piece, so the linear system is `2s·N` square. Each row couples a
/// piece's coefficients to its neighbour's, giving half-bandwidth `2s` on
/// both sides; compact band storage therefore needs `2·(2s)+1 = 4s+1`
/// floats per row. The `9`, `13` and `17` that used to be written inline
/// are exactly `4s+1` for s = 2, 3, 4.
pub const fn minco_storage(s: usize, pieces: usize) -> usize {
    (2 * s) * pieces * (4 * s + 1)
}
/// Storage for the min-acceleration (s=2) `4N × 4N` system at `pieces`.
pub const fn acc_storage(pieces: usize) -> usize {
    minco_storage(2, pieces)
}
/// Storage for the min-jerk (s=3) `6N × 6N` system at `pieces`.
pub const fn jerk_storage(pieces: usize) -> usize {
    minco_storage(3, pieces)
}
/// Storage for the min-snap (s=4) `8N × 8N` system at `pieces`.
pub const fn snap_storage(pieces: usize) -> usize {
    minco_storage(4, pieces)
}

/// Largest system any solver builds at the global piece cap: min-snap at
/// [`super::MAX_PIECES`]. The default `S` for [`BandedSystem`]; solvers
/// sized for a smaller bound should pass their own `S`.
pub const MAX_STORAGE: usize = snap_storage(super::MAX_PIECES);
/// Min-acceleration storage at the global piece cap.
pub const ACC_STORAGE: usize = acc_storage(super::MAX_PIECES);
/// Min-jerk storage at the global piece cap.
pub const JERK_STORAGE: usize = jerk_storage(super::MAX_PIECES);

/// Smallest magnitude a U-diagonal entry is allowed to take.
const PIVOT_FLOOR: f32 = 1e-6;

/// Push a pivot away from zero, preserving sign (non-finite → +floor).
#[inline]
fn clamp_pivot(raw: f32) -> f32 {
    if !raw.is_finite() || raw.abs() < PIVOT_FLOOR {
        if raw < 0.0 { -PIVOT_FLOOR } else { PIVOT_FLOOR }
    } else {
        raw
    }
}

/// A banded matrix with compact band storage (Golub & Van Loan convention).
///
/// Zero heap allocations. All storage is inline in the struct.
///
/// Element (i,j) is stored at `data[(i - j + q) * N + j]`
/// where q = upper bandwidth.
pub struct BandedSystem<const S: usize = MAX_STORAGE> {
    n: usize,
    lower_bw: usize,
    upper_bw: usize,
    data: [f32; S],
}

impl<const S: usize> BandedSystem<S> {
    /// Create an N×N banded system with lower bandwidth `p` and upper bandwidth `q`.
    ///
    /// Panics if the required storage exceeds `S`. This is a real assertion,
    /// not a `debug_assert`: [`get`](Self::get)/[`set`](Self::set) index the
    /// storage unchecked for speed, so this once-per-solve check is what
    /// keeps an oversized `n` from writing past the array in release builds.
    pub fn new(n: usize, p: usize, q: usize) -> Self {
        assert!(
            n >= 1 && n * (p + q + 1) <= S,
            "BandedSystem: n={} needs {} floats of storage, have {}",
            n,
            n * (p + q + 1),
            S
        );
        Self {
            n,
            lower_bw: p,
            upper_bw: q,
            data: [0.0; S],
        }
    }

    /// Reset all entries to zero.
    #[inline]
    pub fn reset(&mut self) {
        let used = self.n * (self.lower_bw + self.upper_bw + 1);
        self.data[..used].fill(0.0);
    }

    /// Reconfigure the active matrix dimension in place. The inline
    /// storage is already sized to `MAX_STORAGE`, so resizing only
    /// updates `self.n` (which determines the index stride and the
    /// `reset()` clear range). Subsequent reads of pre-existing entries
    /// are invalidated — callers must re-populate via `set()` before
    /// next use, which `MincoSnap::solve()` does unconditionally.
    #[inline]
    pub fn set_dimension(&mut self, n: usize) {
        // Same real check as `new()` — see the note there.
        assert!(
            n >= 1 && n * (self.lower_bw + self.upper_bw + 1) <= S,
            "BandedSystem::set_dimension: n={} needs {} floats of storage, have {}",
            n,
            n * (self.lower_bw + self.upper_bw + 1),
            S
        );
        self.n = n;
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
    /// Tiny pivots are clamped to ±[`PIVOT_FLOOR`] and **written back** into
    /// the U diagonal, so the substitution passes divide by the same clamped
    /// value the multipliers were built from. The last diagonal, which the
    /// elimination loop never visits, is clamped after the loop. Without
    /// both, a singular system factorized fine but `solve3` divided by the
    /// raw zero and produced inf/NaN.
    pub fn factorize_lu(&mut self) {
        let n = self.n;
        for k in 0..n - 1 {
            let i_max = (k + self.lower_bw).min(n - 1);
            let pivot = clamp_pivot(self.get(k, k));
            self.set(k, k, pivot);
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
        let last = clamp_pivot(self.get(n - 1, n - 1));
        self.set(n - 1, n - 1, last);
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

    /// Solve Ax = b in-place where b is a single scalar column.
    /// Same substitution as [`solve3`](Self::solve3) with `f32` rows;
    /// used by the 1D (yaw) MINCO solver.
    pub fn solve1(&self, b: &mut [f32]) {
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
        let mut a = BandedSystem::<MAX_STORAGE>::new(3, 1, 1);
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

    /// The pivot floor must reach the substitution passes: an exactly
    /// singular system used to factorize "fine" and then divide by the
    /// raw zero diagonal in `solve3`.
    #[test]
    fn singular_system_stays_finite() {
        let mut a = BandedSystem::<MAX_STORAGE>::new(3, 1, 1);
        a.set(0, 0, 1.0);
        a.set(1, 1, 0.0); // zero pivot in the middle
        a.set(2, 2, 0.0); // ... and on the never-eliminated last diagonal
        a.factorize_lu();
        let mut b = [Vector3::new(1.0, 2.0, 3.0); 3];
        a.solve3(&mut b);
        let mut b1 = [1.0f32, 2.0, 3.0];
        a.solve1(&mut b1);
        let mut badj = [Vector3::new(1.0, 2.0, 3.0); 3];
        a.solve3_adj(&mut badj);
        for i in 0..3 {
            assert!(b[i].iter().all(|v| v.is_finite()), "solve3 row {i}: {:?}", b[i]);
            assert!(b1[i].is_finite(), "solve1 row {i}: {}", b1[i]);
            assert!(badj[i].iter().all(|v| v.is_finite()), "solve3_adj row {i}: {:?}", badj[i]);
        }
    }

    #[test]
    #[should_panic(expected = "BandedSystem: n=")]
    fn oversized_system_is_rejected_in_release_too() {
        // 4 floats of storage cannot hold a 2×2 tridiagonal (2·3 = 6).
        let _ = BandedSystem::<4>::new(2, 1, 1);
    }
}
