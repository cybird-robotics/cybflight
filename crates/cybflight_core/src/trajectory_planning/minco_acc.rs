//! MINCO min-acceleration trajectory solver (polynomial degree 3, s=2),
//! specialized to one dimension.
//!
//! Port of `minco::MINCO_S2NU` from GCOPTER's `minco.hpp`, reduced from
//! 3D to scalar: its sole consumer is the yaw flat output ψ(t) (the
//! fourth component of the quadrotor's differentially flat output
//! alongside 3D position). Planning yaw with the `Vector3`-row layout
//! of [`super::minco_jerk::MincoJerk`] would waste two of three
//! columns in every buffer and every FLOP, so the right-hand side here
//! is a scalar column solved via [`BandedSystem::solve1`].
//!
//! For an n-piece trajectory the linear system is 4n × 4n with
//! half-bandwidth 4. Boundary conditions are (ψ, ψ̇) at head and tail;
//! intermediate constraints enforce the yaw waypoint at each junction
//! plus continuity through the second derivative.
//!
//! ## Angle unwrapping
//!
//! The solver interpolates plain scalars — it knows nothing about
//! S¹ topology. The caller must unwrap the yaw waypoint sequence to a
//! continuous branch first (consecutive values differing by the true
//! intended rotation, not the wrapped principal value), otherwise a
//! −179° → +179° step plans a near-full spin instead of a 2° nudge.
//!
//! ## Omitted API (vs the C++ `MINCO_S2NU`)
//!
//! `getEnergy` / `getEnergyPartialGradByCoeffs` / `propogateGrad` are
//! deliberately not ported. They are only needed when the yaw pieces
//! participate in a spatio-temporal optimization loop; today the yaw
//! waypoints are fixed mission inputs and the segment times are
//! inherited from the already-solved position trajectory, so the
//! solve here is a one-shot linear system. Port them from
//! `MINCO_S2NU` if yaw ever joins the BFGS objective.
//!
//! ## Stack-size warning
//!
//! Like [`super::minco_snap::MincoSnap`], the inline buffers are large
//! (~22 KiB for [`MincoAcc`] at [`MAX_PIECES`]), so firmware must hold
//! the solver in a `StaticCell` — never construct one on an Embassy task
//! stack — and should size a [`MincoAccN`] to the pieces it needs.

use super::banded_system::{acc_storage, BandedSystem};
use super::MAX_PIECES;

/// Scalar boundary state: [ψ, ψ̇] in rad / rad·s⁻¹.
pub type PV1D = [f32; 2];

/// Shift `target` by the multiple of 2π that lands it nearest `prev`.
///
/// Chain over a yaw-waypoint sequence to unwrap it onto one continuous
/// branch before feeding [`MincoAcc::solve`] (see the module-level
/// "Angle unwrapping" note). The result may leave [−π, π] — that is the
/// point; downstream consumers only see ψ through sin/cos.
pub fn unwrap_nearest(prev: f32, target: f32) -> f32 {
    use core::f32::consts::{PI, TAU};
    let mut d = (target - prev) % TAU;
    if d > PI {
        d -= TAU;
    } else if d < -PI {
        d += TAU;
    }
    prev + d
}

/// MINCO min-acceleration solver (degree 3, smoothness s=2), 1D, sized
/// for at most `P` pieces.
///
/// Zero heap allocations — all buffers are inline fixed-size arrays.
/// The active piece count is tracked in `self.n`; only the leading
/// `4·n` entries of every buffer are live per solve.
///
/// `C` must equal `4 * P` and `S` must be at least [`acc_storage`]`(P)`
/// — see [`super::minco_jerk::MincoJerkN`]. [`MincoAcc`] is the instance
/// at the global cap; the yaw output [`YawTrajectory`] is always sized
/// at [`MAX_PIECES`] like the position trajectory container.
pub struct MincoAccN<const P: usize, const C: usize, const S: usize> {
    n: usize,
    head_pv: PV1D,
    tail_pv: PV1D,
    banded: BandedSystem<S>,
    /// Coefficient column: 4N rows, ascending power order per piece.
    b: [f32; C],
    t1: [f32; P],
    t2: [f32; P],
    t3: [f32; P],
}

/// [`MincoAccN`] at the global trajectory cap [`MAX_PIECES`] (~22 KiB).
pub type MincoAcc = MincoAccN<MAX_PIECES, { 4 * MAX_PIECES }, { acc_storage(MAX_PIECES) }>;

impl<const P: usize, const C: usize, const S: usize> MincoAccN<P, C, S> {
    /// Compile-time check that `C` and `S` match `P`; evaluated by `new`.
    const LAYOUT_OK: () = assert!(
        P >= 1 && P <= MAX_PIECES && C == 4 * P && S >= acc_storage(P),
        "MincoAccN<P, C, S>: C must be 4*P and S >= acc_storage(P)"
    );

    /// Initialize the solver. Panics if `piece_num` exceeds `P`.
    pub fn new(head_state: &PV1D, tail_state: &PV1D, piece_num: usize) -> Self {
        let () = Self::LAYOUT_OK;
        assert!(
            piece_num >= 1 && piece_num <= P,
            "MincoAccN: {piece_num} pieces exceeds the type's bound of {P}"
        );
        Self {
            n: piece_num,
            head_pv: *head_state,
            tail_pv: *tail_state,
            banded: BandedSystem::new(4 * piece_num, 4, 4),
            b: [0.0; C],
            t1: [0.0; P],
            t2: [0.0; P],
            t3: [0.0; P],
        }
    }

    /// Update the head/tail (ψ, ψ̇) in place. Mirror of
    /// [`super::minco_jerk::MincoJerk::set_boundary`] — lets a
    /// BSS-resident solver be reused across solves.
    pub fn set_boundary(&mut self, head_state: &PV1D, tail_state: &PV1D) {
        self.head_pv = *head_state;
        self.tail_pv = *tail_state;
    }

    /// Reconfigure the active piece count in place (mission lengths
    /// vary; the yaw piece count must track the position trajectory's).
    /// `solve()` must be called before any subsequent read.
    pub fn set_piece_count(&mut self, piece_num: usize) {
        assert!(
            piece_num >= 1 && piece_num <= P,
            "MincoAccN: {piece_num} pieces exceeds the type's bound of {P}"
        );
        self.n = piece_num;
        self.banded.set_dimension(4 * piece_num);
    }

    /// Currently configured piece count.
    #[inline]
    pub fn piece_count(&self) -> usize {
        self.n
    }

    /// Set yaw waypoints + time allocation and solve for coefficients.
    ///
    /// `waypoints` contains the n−1 intermediate (unwrapped) yaw
    /// targets in order; `times` contains the n segment durations —
    /// pass the same durations the position trajectory was solved
    /// with so the two stay synchronized. Mirrors the C++
    /// `setParameters` exactly (same row indices, same sign convention
    /// on the −1 continuity entries).
    pub fn solve(&mut self, waypoints: &[f32], times: &[f32]) {
        debug_assert_eq!(times.len(), self.n);
        debug_assert_eq!(waypoints.len(), self.n - 1);

        for i in 0..self.n {
            let t = times[i];
            self.t1[i] = t;
            let t2 = t * t;
            self.t2[i] = t2;
            self.t3[i] = t2 * t;
        }

        self.banded.reset();
        let sys_size = 4 * self.n;
        self.b[..sys_size].fill(0.0);

        // Head boundary: c_0 = ψ_h, c_1 = ψ̇_h.
        self.banded.set(0, 0, 1.0);
        self.banded.set(1, 1, 1.0);
        self.b[0] = self.head_pv[0];
        self.b[1] = self.head_pv[1];

        // Intermediate junctions. Each junction emits 4 rows, indexed
        // 4i+2 … 4i+5. Layout follows MINCO_S2NU verbatim.
        for i in 0..(self.n - 1) {
            let base = 4 * i;
            let t1 = self.t1[i];
            let t2 = self.t2[i];
            let t3 = self.t3[i];

            // row 4i+2: acceleration continuity → 2·c2 + 6·t·c3 = 2·c2'.
            self.banded.set(base + 2, base + 2, 2.0);
            self.banded.set(base + 2, base + 3, 6.0 * t1);
            self.banded.set(base + 2, base + 6, -2.0);

            // row 4i+3: yaw-at-waypoint (left side of junction).
            self.banded.set(base + 3, base + 0, 1.0);
            self.banded.set(base + 3, base + 1, t1);
            self.banded.set(base + 3, base + 2, t2);
            self.banded.set(base + 3, base + 3, t3);

            // row 4i+4: position continuity.
            self.banded.set(base + 4, base + 0, 1.0);
            self.banded.set(base + 4, base + 1, t1);
            self.banded.set(base + 4, base + 2, t2);
            self.banded.set(base + 4, base + 3, t3);
            self.banded.set(base + 4, base + 4, -1.0);

            // row 4i+5: velocity continuity.
            self.banded.set(base + 5, base + 1, 1.0);
            self.banded.set(base + 5, base + 2, 2.0 * t1);
            self.banded.set(base + 5, base + 3, 3.0 * t2);
            self.banded.set(base + 5, base + 5, -1.0);

            self.b[base + 3] = waypoints[i];
        }

        // Tail boundary: ψ(T_last) = ψ_t, ψ̇(T_last) = ψ̇_t.
        let n4 = 4 * self.n;
        let last = self.n - 1;
        let t1 = self.t1[last];
        let t2 = self.t2[last];
        let t3 = self.t3[last];

        self.banded.set(n4 - 2, n4 - 4, 1.0);
        self.banded.set(n4 - 2, n4 - 3, t1);
        self.banded.set(n4 - 2, n4 - 2, t2);
        self.banded.set(n4 - 2, n4 - 1, t3);

        self.banded.set(n4 - 1, n4 - 3, 1.0);
        self.banded.set(n4 - 1, n4 - 2, 2.0 * t1);
        self.banded.set(n4 - 1, n4 - 1, 3.0 * t2);

        self.b[n4 - 2] = self.tail_pv[0];
        self.b[n4 - 1] = self.tail_pv[1];

        // Solve directly on b.
        self.banded.factorize_lu();
        self.banded.solve1(&mut self.b[..sys_size]);
    }

    /// Extract the solved trajectory as a piecewise cubic.
    pub fn get_trajectory(&self) -> YawTrajectory {
        let mut traj = YawTrajectory {
            n: self.n,
            coeffs: [[0.0; 4]; MAX_PIECES],
            cum_dur: [0.0; MAX_PIECES],
        };
        let mut acc = 0.0;
        for i in 0..self.n {
            let base = 4 * i;
            traj.coeffs[i].copy_from_slice(&self.b[base..base + 4]);
            acc += self.t1[i];
            traj.cum_dur[i] = acc;
        }
        traj
    }
}

/// Piecewise-cubic yaw trajectory solved by [`MincoAcc`].
///
/// Scalar counterpart of [`super::piecewise_polynomial::PiecewisePolynomial`]:
/// same cumulative-duration binary search, same clamp-to-last-piece
/// behavior past the end (samples extrapolate the final cubic, exactly
/// as the position trajectory does — the caller clamps `t` for both).
#[derive(Clone)]
pub struct YawTrajectory {
    n: usize,
    /// Per-piece cubic coefficients in ascending power order.
    coeffs: [[f32; 4]; MAX_PIECES],
    /// Cumulative durations: cum_dur[i] = sum of durations[0..=i].
    cum_dur: [f32; MAX_PIECES],
}

impl YawTrajectory {
    #[inline]
    pub fn num_pieces(&self) -> usize {
        self.n
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

    /// Sample the yaw triple `[ψ, ψ̇, ψ̈]` at global time `t` — the
    /// exact shape [`super::flatness::flatness_to_state_tilt_yaw`]
    /// takes as its `yaw_triple` argument.
    pub fn sample(&self, t: f32) -> [f32; 3] {
        debug_assert!(self.n > 0);
        // Binary search: first i where cum_dur[i] >= t.
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
        let lt = t - prev_cum;

        let [c0, c1, c2, c3] = self.coeffs[lo];
        [
            c0 + lt * (c1 + lt * (c2 + lt * c3)),
            c1 + lt * (2.0 * c2 + lt * (3.0 * c3)),
            2.0 * c2 + lt * (6.0 * c3),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single piece, rest-to-rest: the min-acc cubic over T=1 from 0 to
    /// ψ_f is ψ(t) = ψ_f·(3t² − 2t³); check endpoints, midpoint, and
    /// zero boundary rates.
    #[test]
    fn test_single_piece_rest_to_rest() {
        let psi_f = core::f32::consts::FRAC_PI_2;
        let mut solver = MincoAcc::new(&[0.0, 0.0], &[psi_f, 0.0], 1);
        solver.solve(&[], &[1.0]);
        let traj = solver.get_trajectory();

        let [p0, v0, _] = traj.sample(0.0);
        let [pm, _, _] = traj.sample(0.5);
        let [p1, v1, _] = traj.sample(1.0);
        assert!(p0.abs() < 1e-5);
        assert!(v0.abs() < 1e-5);
        assert!((pm - 0.5 * psi_f).abs() < 1e-4, "midpoint {pm}");
        assert!((p1 - psi_f).abs() < 1e-4);
        assert!(v1.abs() < 1e-4);
    }

    /// Nonzero boundary rates are honored exactly (they are hard
    /// constraint rows, not soft costs).
    #[test]
    fn test_boundary_rates() {
        let mut solver = MincoAcc::new(&[0.2, 0.5], &[1.0, -0.3], 1);
        solver.solve(&[], &[2.0]);
        let traj = solver.get_trajectory();

        let [p0, v0, _] = traj.sample(0.0);
        let [p1, v1, _] = traj.sample(2.0);
        assert!((p0 - 0.2).abs() < 1e-4);
        assert!((v0 - 0.5).abs() < 1e-4);
        assert!((p1 - 1.0).abs() < 1e-4);
        assert!((v1 - (-0.3)).abs() < 1e-4);
    }

    #[test]
    fn test_unwrap_nearest() {
        use core::f32::consts::PI;
        let deg = |d: f32| d * PI / 180.0;
        // −179° → +179° goes the short way (2° backwards, to −181°).
        let u = unwrap_nearest(deg(-179.0), deg(179.0));
        assert!((u - deg(-181.0)).abs() < 1e-5, "{u}");
        // Already within π: unchanged.
        assert!((unwrap_nearest(0.3, 1.0) - 1.0).abs() < 1e-6);
        // Multiple turns collapse to the nearest branch: 3π from 0 → π.
        let u = unwrap_nearest(0.0, 3.0 * PI);
        assert!((u - PI).abs() < 1e-5, "{u}");
    }

    /// A heading sequence crossing ±π, unwrapped and solved: the spline
    /// passes through the unwrapped waypoint and hits the unwrapped tail
    /// (which lies outside [−π, π] — by design).
    #[test]
    fn test_yaw_spline_through_pi_crossing() {
        use core::f32::consts::PI;
        let start = 0.9 * PI;
        // Raw headings: −0.9π, −0.7π — the short way from +0.9π is
        // *forward* through +π onto the next branch.
        let w0 = unwrap_nearest(start, -0.9 * PI);
        let tail = unwrap_nearest(w0, -0.7 * PI);
        assert!((w0 - 1.1 * PI).abs() < 1e-5, "{w0}");
        assert!((tail - 1.3 * PI).abs() < 1e-5, "{tail}");

        let mut solver = MincoAcc::new(&[start, 0.0], &[tail, 0.0], 2);
        solver.solve(&[w0], &[1.0, 1.0]);
        let traj = solver.get_trajectory();
        let [p_wp, _, _] = traj.sample(1.0);
        assert!((p_wp - w0).abs() < 1e-4, "waypoint missed: {p_wp}");
        let [p_end, v_end, _] = traj.sample(2.0);
        assert!((p_end - tail).abs() < 1e-4);
        assert!(v_end.abs() < 1e-4);
    }

    /// Two pieces: the waypoint is hit at the junction and ψ, ψ̇, ψ̈
    /// are all continuous across it (S2NU enforces C² at junctions).
    #[test]
    fn test_two_piece_waypoint_and_continuity() {
        let mut solver = MincoAcc::new(&[0.0, 0.0], &[1.5, 0.0], 2);
        solver.solve(&[0.9], &[0.8, 1.2]);
        let traj = solver.get_trajectory();

        assert!((traj.total_duration() - 2.0).abs() < 1e-6);
        let [p_wp, _, _] = traj.sample(0.8);
        assert!((p_wp - 0.9).abs() < 1e-4, "waypoint missed: {p_wp}");

        let eps = 1e-3;
        let left = traj.sample(0.8 - eps);
        let right = traj.sample(0.8 + eps);
        assert!((left[0] - right[0]).abs() < 1e-2, "ψ jump at junction");
        assert!((left[1] - right[1]).abs() < 1e-2, "ψ̇ jump at junction");
        assert!((left[2] - right[2]).abs() < 5e-2, "ψ̈ jump at junction");
    }
}
