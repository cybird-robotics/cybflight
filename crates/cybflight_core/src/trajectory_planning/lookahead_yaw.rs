//! Desired yaw for `lookahead: true` missions: point the nose at the
//! reference position `dt_s` ahead along the trajectory, followed at a
//! bounded rate.
//!
//! Two pieces, split so the firmware keeps doing the trajectory
//! evaluations it already has in hand (the sampler's per-node pos/vel):
//!
//! - [`chord_heading`] — the raw look-at heading ψ = atan2(Δy, Δx) of the
//!   chord d = p(τ+dt) − p(τ), plus its **analytic** rate
//!   ψ̇ = (d × ḋ)_z / ‖d_xy‖², ḋ = v(τ+dt) − v(τ). The analytic rate is
//!   what makes the body-rate feedforward agree with the per-node
//!   attitude sweep; a finite difference would be noisy exactly where
//!   the atan2 is ill-conditioned.
//! - [`slew_heading`] — one step of a rate-limited follower. The raw
//!   heading is discontinuous: it flips ~180° wherever the xy path
//!   doubles back (the chord shrinks through zero and reverses), and at
//!   mission entry it generally differs from the yaw the vehicle is
//!   holding. The follower turns both into a sweep at `max_rate`, always
//!   the short way round, and reports the rate it actually commands so
//!   the feedforward stays consistent with the attitude reference.

use super::minco_acc::unwrap_nearest;
use super::types::Vec3;

/// Below this look-at displacement [m] the chord's direction is noise;
/// the caller holds the previous heading instead.
pub const MIN_LOOKAHEAD_DIST_M: f32 = 0.05;

/// Wrap an angle into [−π, π].
#[inline]
pub fn wrap_pi(x: f32) -> f32 {
    unwrap_nearest(0.0, x)
}

/// Raw look-at heading and its analytic rate from the chord between the
/// reference point `(pos, vel)` at τ and the point `(pos_ahead,
/// vel_ahead)` at τ + dt. When τ + dt is clamped to the trajectory end the
/// ahead point is stationary: pass `vel_ahead = Vec3::zeros()`.
///
/// Returns `None` when the xy chord is shorter than
/// [`MIN_LOOKAHEAD_DIST_M`] or any input is non-finite — the heading is
/// undefined there and the caller should hold.
pub fn chord_heading(
    pos: &Vec3,
    vel: &Vec3,
    pos_ahead: &Vec3,
    vel_ahead: &Vec3,
) -> Option<(f32, f32)> {
    let d = pos_ahead - pos;
    let (dx, dy) = (d.x, d.y);
    let d2 = dx * dx + dy * dy;
    // `!(a > b)` rather than `a <= b` so a NaN chord also holds.
    if !(d2 > MIN_LOOKAHEAD_DIST_M * MIN_LOOKAHEAD_DIST_M) {
        return None;
    }
    let dd = vel_ahead - vel;
    let (ddx, ddy) = (dd.x, dd.y);
    let psi = libm::atan2f(dy, dx);
    let dpsi = (dx * ddy - dy * ddx) / d2;
    if psi.is_finite() && dpsi.is_finite() {
        Some((psi, dpsi))
    } else {
        None
    }
}

/// One step of the bounded-rate heading follower: advance from `prev`
/// toward `target` (raw heading, raw rate) by at most `max_rate · dt`,
/// the short way round.
///
/// Returns `(ψ, ψ̇)` with ψ wrapped to [−π, π]:
/// - `target = None` (degenerate chord): hold `prev`, ψ̇ = 0.
/// - within reach: land on the raw heading with its analytic rate
///   (clamped to ±`max_rate`) — the steady, well-tracked case, where the
///   reference is exactly the look-at heading.
/// - out of reach: step by ±`max_rate · dt` and report ψ̇ = ±`max_rate`,
///   the rate the reference actually sweeps at.
///
/// A non-positive or non-finite `max_rate` / `dt` freezes the heading at
/// `prev` (degrade to constant yaw rather than emit a NaN).
pub fn slew_heading(prev: f32, target: Option<(f32, f32)>, max_rate: f32, dt: f32) -> (f32, f32) {
    let Some((raw, raw_rate)) = target else {
        return (wrap_pi(prev), 0.0);
    };
    let max_step = max_rate * dt;
    if !(max_step > 0.0 && max_step.is_finite()) {
        return (wrap_pi(prev), 0.0);
    }
    let err = wrap_pi(raw - prev);
    if err.abs() <= max_step {
        (wrap_pi(raw), raw_rate.clamp(-max_rate, max_rate))
    } else {
        let dir = if err > 0.0 { 1.0 } else { -1.0 };
        (wrap_pi(prev + dir * max_step), dir * max_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, PI};

    fn close(a: f32, b: f32, tol: f32) -> bool {
        wrap_pi(a - b).abs() <= tol
    }

    #[test]
    fn straight_line_heading_has_zero_rate() {
        // Constant velocity along +y.
        let v = Vec3::new(0.0, 2.0, 0.0);
        let h = chord_heading(&Vec3::new(0.0, 0.0, 1.0), &v, &Vec3::new(0.0, 1.0, 1.0), &v);
        let (psi, dpsi) = h.unwrap();
        assert!(close(psi, FRAC_PI_2, 1e-6));
        assert!(dpsi.abs() < 1e-6);
    }

    #[test]
    fn circle_rate_matches_finite_difference() {
        // p(t) = R·(cos ωt, sin ωt); chord to t + dt.
        let (r, w, dt) = (1.5f32, 2.0f32, 0.5f32);
        let p = |t: f32| Vec3::new(r * libm::cosf(w * t), r * libm::sinf(w * t), 1.0);
        let v = |t: f32| Vec3::new(-r * w * libm::sinf(w * t), r * w * libm::cosf(w * t), 0.0);
        let head = |t: f32| chord_heading(&p(t), &v(t), &p(t + dt), &v(t + dt)).unwrap();
        for i in 0..20 {
            let t = 0.1 * i as f32;
            let h = 1e-3;
            let fd = wrap_pi(head(t + h).0 - head(t - h).0) / (2.0 * h);
            let (_, dpsi) = head(t);
            // On a circle the look-at heading rotates at ω.
            assert!((dpsi - w).abs() < 1e-3, "t={t}: analytic {dpsi} vs ω {w}");
            assert!((dpsi - fd).abs() < 1e-2, "t={t}: analytic {dpsi} vs FD {fd}");
        }
    }

    #[test]
    fn short_or_nan_chord_is_none() {
        let z = Vec3::zeros();
        // 2.2 cm xy chord (the 5 m vertical component does not count).
        assert!(chord_heading(&z, &z, &Vec3::new(0.01, 0.02, 5.0), &z).is_none());
        assert!(chord_heading(&z, &z, &Vec3::new(f32::NAN, 1.0, 0.0), &z).is_none());
    }

    #[test]
    fn slew_holds_on_degenerate_chord() {
        assert_eq!(slew_heading(0.3, None, 4.0, 0.02), (0.3, 0.0));
    }

    #[test]
    fn slew_lands_on_reachable_target_with_clamped_rate() {
        let (psi, dpsi) = slew_heading(0.0, Some((0.05, 9.0)), 4.0, 0.02);
        assert_eq!(psi, 0.05);
        assert_eq!(dpsi, 4.0);
    }

    #[test]
    fn entry_step_becomes_bounded_sweep() {
        // Vehicle holds 0 rad, path heading is 160° away: the reference
        // must sweep there at max_rate, never step.
        let target = Some((160f32.to_radians(), 0.0));
        let (max_rate, dt) = (4.0f32, 0.02f32);
        let mut psi = 0.0f32;
        let mut steps = 0;
        loop {
            let (next, dpsi) = slew_heading(psi, target, max_rate, dt);
            assert!(wrap_pi(next - psi).abs() <= max_rate * dt + 1e-6);
            psi = next;
            steps += 1;
            if dpsi == 0.0 {
                break;
            }
            assert_eq!(dpsi, max_rate);
            assert!(steps < 100, "never converged");
        }
        assert!(close(psi, 160f32.to_radians(), 1e-6));
        // 160° at 4 rad/s ≈ 0.70 s = 35 ticks of 20 ms.
        assert!((34..=36).contains(&steps), "{steps}");
    }

    #[test]
    fn reversal_flip_is_rate_limited_and_takes_short_way() {
        // Raw heading flips from +170° to −170°: the short way is +20°
        // through ±180°, not −340°.
        let (psi, dpsi) = slew_heading(170f32.to_radians(), Some((-170f32.to_radians(), 0.0)), 4.0, 0.02);
        assert!(dpsi > 0.0);
        assert!(close(psi, 170f32.to_radians() + 0.08, 1e-5));
        // And the output stays wrapped.
        assert!((-PI..=PI).contains(&psi));
        let (psi, _) = slew_heading(3.1, Some((-3.1, 0.0)), 4.0, 0.02);
        assert!((-PI..=PI).contains(&psi));
    }

    #[test]
    fn bad_rate_or_dt_freezes_heading() {
        let t = Some((1.0, 0.0));
        assert_eq!(slew_heading(0.2, t, 0.0, 0.02), (0.2, 0.0));
        assert_eq!(slew_heading(0.2, t, f32::NAN, 0.02), (0.2, 0.0));
        assert_eq!(slew_heading(0.2, t, 4.0, -1.0), (0.2, 0.0));
    }
}
