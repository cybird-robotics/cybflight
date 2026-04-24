use super::polynomial::Polynomial;
use super::types::{Vec3, ZERO3};
use super::MAX_PIECES;

/// A piecewise polynomial trajectory in 3D with zero heap allocations.
///
/// Stores up to MAX_PIECES polynomial pieces and precomputed cumulative
/// durations for O(log N) piece lookup via binary search.
#[derive(Clone)]
pub struct PiecewisePolynomial {
    pub pieces: [Polynomial; MAX_PIECES],
    pub n: usize,
    /// Cumulative durations: cum_dur[i] = sum of durations[0..=i].
    /// Used for O(log N) piece lookup.
    cum_dur: [f32; MAX_PIECES],
}

impl PiecewisePolynomial {
    pub fn new() -> Self {
        Self {
            pieces: [Polynomial {
                degree: 0,
                duration: 0.0,
                coeffs: [ZERO3; super::polynomial::MAX_COEFFS],
            }; MAX_PIECES],
            n: 0,
            cum_dur: [0.0; MAX_PIECES],
        }
    }

    /// Build from a slice of polynomial pieces.
    pub fn from_pieces(pieces: &[Polynomial]) -> Self {
        debug_assert!(pieces.len() <= MAX_PIECES);
        let mut pp = Self::new();
        pp.n = pieces.len();
        pp.pieces[..pp.n].copy_from_slice(pieces);
        pp.recompute_cumulative();
        pp
    }

    /// Recompute the cumulative duration table. Call after modifying pieces.
    fn recompute_cumulative(&mut self) {
        let mut acc = 0.0;
        for i in 0..self.n {
            acc += self.pieces[i].duration;
            self.cum_dur[i] = acc;
        }
    }

    #[inline]
    pub fn num_pieces(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Access a polynomial piece by index.
    #[inline]
    pub fn piece(&self, idx: usize) -> &Polynomial {
        debug_assert!(idx < self.n);
        &self.pieces[idx]
    }

    /// Total duration of the trajectory.
    #[inline]
    pub fn total_duration(&self) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            self.cum_dur[self.n - 1]
        }
    }

    /// Find the piece index and local time for a given global time.
    /// Uses binary search on precomputed cumulative durations for O(log N).
    /// Clamps to the last piece if t exceeds total duration.
    #[inline]
    pub fn locate_piece(&self, t: f32) -> (usize, f32) {
        debug_assert!(self.n > 0);
        // Binary search: find first i where cum_dur[i] >= t
        let cum = &self.cum_dur[..self.n];
        let mut lo = 0usize;
        let mut hi = self.n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if cum[mid] < t {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo >= self.n {
            lo = self.n - 1;
        }
        let prev_cum = if lo > 0 { cum[lo - 1] } else { 0.0 };
        (lo, t - prev_cum)
    }

    #[inline(always)]
    pub fn get_pos(&self, t: f32) -> Vec3 {
        let (idx, lt) = self.locate_piece(t);
        self.pieces[idx].get_pos(lt)
    }

    #[inline(always)]
    pub fn get_vel(&self, t: f32) -> Vec3 {
        let (idx, lt) = self.locate_piece(t);
        self.pieces[idx].get_vel(lt)
    }

    #[inline(always)]
    pub fn get_acc(&self, t: f32) -> Vec3 {
        let (idx, lt) = self.locate_piece(t);
        self.pieces[idx].get_acc(lt)
    }

    #[inline(always)]
    pub fn get_jerk(&self, t: f32) -> Vec3 {
        let (idx, lt) = self.locate_piece(t);
        self.pieces[idx].get_jerk(lt)
    }

    #[inline(always)]
    pub fn get_snap(&self, t: f32) -> Vec3 {
        let (idx, lt) = self.locate_piece(t);
        self.pieces[idx].get_snap(lt)
    }

    /// Get all boundary points (start of each piece + end of last piece).
    /// Returns the number of points written and the buffer.
    pub fn get_points(&self, out: &mut [Vec3]) -> usize {
        let count = self.n + 1;
        for i in 0..self.n {
            out[i] = self.pieces[i].get_pos(0.0);
        }
        if self.n > 0 {
            let last = &self.pieces[self.n - 1];
            out[self.n] = last.get_pos(last.duration);
        }
        count
    }

    /// Get all boundary points as a Vec (std feature only).
    #[cfg(feature = "std")]
    pub fn get_points_vec(&self) -> Vec<Vec3> {
        let mut pts = Vec::with_capacity(self.n + 1);
        for i in 0..self.n {
            pts.push(self.pieces[i].get_pos(0.0));
        }
        if self.n > 0 {
            let last = &self.pieces[self.n - 1];
            pts.push(last.get_pos(last.duration));
        }
        pts
    }

    /// Get durations of all pieces as a fixed array.
    pub fn durations(&self, out: &mut [f32]) -> usize {
        for i in 0..self.n {
            out[i] = self.pieces[i].duration;
        }
        self.n
    }
}

impl Default for PiecewisePolynomial {
    fn default() -> Self {
        Self::new()
    }
}
