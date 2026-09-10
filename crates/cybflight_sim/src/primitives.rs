//! Procedural maneuver primitives for recovery-episode training
//! (docs/learned_mpc_cost.md, PLAN B).
//!
//! Each primitive is a short MINCO trajectory built from a template
//! (sprint, hairpin, chicane, climb/dive, split-S, Immelmann, loop, banked
//! circle, hover-to-sprint), optionally chained with a second one, at a
//! random speed, heading and position. Durations are stretched until the
//! flatness-derived thrust and body-rate demands fit inside a *sampled*
//! fraction of the vehicle envelope, so the family sweeps the actuator
//! margin from "at the limit" to "easy" instead of clustering at either.
//!
//! The templates are geometric *types*, not any flown mission: the
//! evaluation trajectories (`missions/indoor_splits_timeopt.yaml` in
//! particular) are never generated here.

use cybflight_core::trajectory_planning::flatness::flatness_to_thrust_omega;
use cybflight_core::trajectory_planning::minco_snap::MincoSnap;
use cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};
use nalgebra::Vector3;
use rand::Rng;

/// Vehicle envelope the generator plans against.
#[derive(Clone, Copy, Debug)]
pub struct Envelope {
    pub mass_kg: f32,
    pub grav: f32,
    /// Collective thrust ceiling the MPC is allowed [N].
    pub thrust_max_n: f32,
    /// Per-axis body-rate limit [rad/s].
    pub rate_max: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Sprint,
    Hairpin,
    Chicane,
    ClimbDive,
    SplitS,
    Immelmann,
    Loop,
    Circle,
    HoverSprint,
    /// Dense random waypoints in a box — the shape class of the planned
    /// time-optimal missions (curvy, thrust-limited everywhere).
    RandomWaypoints,
}

pub const KINDS: [Kind; 10] = [
    Kind::Sprint,
    Kind::Hairpin,
    Kind::Chicane,
    Kind::ClimbDive,
    Kind::SplitS,
    Kind::Immelmann,
    Kind::Loop,
    Kind::Circle,
    Kind::HoverSprint,
    Kind::RandomWaypoints,
];

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Sprint => "sprint",
            Kind::Hairpin => "hairpin",
            Kind::Chicane => "chicane",
            Kind::ClimbDive => "climb_dive",
            Kind::SplitS => "split_s",
            Kind::Immelmann => "immelmann",
            Kind::Loop => "loop",
            Kind::Circle => "circle",
            Kind::HoverSprint => "hover_sprint",
            Kind::RandomWaypoints => "random_waypoints",
        }
    }
}

/// A generated primitive and its provenance.
pub struct Primitive {
    pub traj: PiecewisePolynomial,
    pub kind: Kind,
    /// Commanded path speed [m/s] before feasibility stretching.
    pub speed: f32,
    /// Fraction of the thrust ceiling the trajectory was allowed to use.
    pub thrust_budget: f32,
    /// Peak flatness thrust demand / `thrust_max_n` after stretching.
    pub peak_thrust_frac: f32,
    /// Peak flatness rate demand / limit after stretching.
    pub peak_rate_frac: f32,
    /// Feasibility stretch iterations applied.
    pub stretches: u32,
}

/// Template points in the local frame (start at the origin heading +x),
/// plus the heading (unit xy vector) at the end.
fn template(kind: Kind, rng: &mut impl Rng) -> (Vec<Vec3>, Vec3) {
    let mut pts = Vec::new();
    let mut heading = Vec3::x();
    match kind {
        Kind::Sprint | Kind::HoverSprint => {
            let len = rng.random_range(3.0..10.0f32);
            for i in 1..=4 {
                pts.push(Vec3::new(len * i as f32 / 4.0, 0.0, 0.0));
            }
        }
        Kind::Hairpin => {
            let r = rng.random_range(0.5..2.5f32);
            let sgn = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            for i in 1..=8 {
                let th = core::f32::consts::PI * i as f32 / 8.0;
                pts.push(Vec3::new(r * th.sin(), sgn * r * (1.0 - th.cos()), 0.0));
            }
            let out = rng.random_range(1.0..4.0f32);
            pts.push(Vec3::new(-out, sgn * 2.0 * r, 0.0));
            heading = -Vec3::x();
        }
        Kind::Chicane => {
            let amp = rng.random_range(0.5..2.0f32);
            let lam = rng.random_range(2.0..5.0f32);
            let n = 16;
            for i in 1..=n {
                let x = 2.0 * lam * i as f32 / n as f32;
                pts.push(Vec3::new(x, amp * (2.0 * core::f32::consts::PI * x / lam).sin(), 0.0));
            }
        }
        Kind::ClimbDive => {
            let amp = rng.random_range(0.3..1.5f32);
            let lam = rng.random_range(2.0..5.0f32);
            let n = 16;
            for i in 1..=n {
                let x = 2.0 * lam * i as f32 / n as f32;
                pts.push(Vec3::new(x, 0.0, amp * (2.0 * core::f32::consts::PI * x / lam).sin()));
            }
        }
        Kind::SplitS | Kind::Immelmann => {
            let r = rng.random_range(0.4..1.6f32);
            let up = if kind == Kind::Immelmann { 1.0 } else { -1.0 };
            for i in 1..=8 {
                let th = core::f32::consts::PI * i as f32 / 8.0;
                pts.push(Vec3::new(r * th.sin(), 0.0, up * r * (1.0 - th.cos())));
            }
            let out = rng.random_range(1.0..4.0f32);
            pts.push(Vec3::new(-out, 0.0, up * 2.0 * r));
            heading = -Vec3::x();
        }
        Kind::Loop => {
            let r = rng.random_range(0.5..1.5f32);
            for i in 1..=16 {
                let th = 2.0 * core::f32::consts::PI * i as f32 / 16.0;
                pts.push(Vec3::new(r * th.sin(), 0.0, r * (1.0 - th.cos())));
            }
            let out = rng.random_range(1.0..3.0f32);
            pts.push(Vec3::new(out, 0.0, 0.0));
        }
        Kind::RandomWaypoints => {
            let n = rng.random_range(6..12);
            let mut p = Vec3::new(0.0, 0.0, 0.0);
            let mut dir = Vec3::x();
            for _ in 0..n {
                // Turn by up to ±100° in yaw and ±35° in pitch per hop,
                // hop length 0.8–2.5 m, keeping the path inside ~6 m.
                let dyaw = rng.random_range(-1.75..1.75f32);
                let dpitch = rng.random_range(-0.6..0.6f32);
                let yaw = dir.y.atan2(dir.x) + dyaw;
                dir = Vec3::new(yaw.cos() * dpitch.cos(), yaw.sin() * dpitch.cos(), dpitch.sin());
                let len = rng.random_range(0.8..2.5f32);
                p += dir * len;
                p.x = p.x.clamp(-4.0, 6.0);
                p.y = p.y.clamp(-4.0, 4.0);
                p.z = p.z.clamp(-1.0, 2.0);
                pts.push(p);
            }
            heading = Vec3::new(dir.x, dir.y, 0.0);
            if heading.norm() < 1e-3 {
                heading = Vec3::x();
            }
            heading /= heading.norm();
        }
        Kind::Circle => {
            let r = rng.random_range(0.8..3.0f32);
            let sgn = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            for i in 1..=16 {
                let th = 2.0 * core::f32::consts::PI * i as f32 / 16.0;
                pts.push(Vec3::new(r * th.sin(), sgn * r * (1.0 - th.cos()), 0.0));
            }
        }
    }
    (pts, heading)
}

fn rot_z(v: Vec3, yaw: f32) -> Vec3 {
    let (s, c) = yaw.sin_cos();
    Vec3::new(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
}

/// Peak flatness demand along the trajectory as fractions of the envelope.
pub fn peak_demand(traj: &PiecewisePolynomial, env: &Envelope) -> (f32, f32) {
    let dur = traj.total_duration();
    let mut t = 0.0f32;
    let (mut pt, mut pr) = (0.0f32, 0.0f32);
    while t <= dur {
        let a = traj.get_acc(t);
        let j = traj.get_jerk(t);
        let (thrust, omega) = match flatness_to_thrust_omega(a, j, 0.0, 0.0, env.grav) {
            Ok((alpha, _, w)) => (env.mass_kg * alpha, w),
            Err(_) => (env.mass_kg * Vector3::new(a.x, a.y, a.z + env.grav).norm(), ZERO3),
        };
        pt = pt.max(thrust / env.thrust_max_n);
        for i in 0..3 {
            pr = pr.max(omega[i].abs() / env.rate_max[i]);
        }
        t += 0.01;
    }
    (pt, pr)
}

/// Draw one primitive. `start` is the world position of its first point
/// (the generator only shapes and orients; the caller places it).
pub fn random_primitive(rng: &mut impl Rng, env: &Envelope, start: Vec3) -> Primitive {
    let kind = KINDS[rng.random_range(0..KINDS.len())];
    // Log-uniform speed over the whole flight envelope (indoor racing to
    // outdoor 60 m/s). Template geometry is scaled with `(v/12)²` below so
    // the centripetal demand of a template stays speed-independent and the
    // feasibility loop shapes it from there rather than from a fixed box.
    let speed = (rng.random_range(1.5f32.ln()..60.0f32.ln())).exp();
    let geom = (speed / 12.0).max(1.0).powi(2);
    // Budgets deliberately reach past the envelope: time-optimal missions
    // (the evaluation split-S included) demand rates above the limit and
    // thrust at the ceiling, and the policy must have seen that regime.
    let thrust_budget = rng.random_range(0.45..1.05f32);
    let rate_budget = rng.random_range(1.0..3.0f32);
    let yaw = rng.random_range(-core::f32::consts::PI..core::f32::consts::PI);

    let (body, mut heading) = template(kind, rng);
    // Lead-in straight: the head enters with `v_in` and zero
    // acceleration, so the maneuver's curvature must build up over a
    // straight run (as every planned mission does) or the min-snap fit
    // rings at the first piece. Hover starts need no lead-in.
    let mut pts: Vec<Vec3> = Vec::new();
    let lead_in = if kind == Kind::HoverSprint { 0.0 } else { rng.random_range(1.0..3.0f32) };
    if lead_in > 0.0 {
        pts.push(Vec3::new(lead_in * 0.5, 0.0, 0.0));
        pts.push(Vec3::new(lead_in, 0.0, 0.0));
    }
    let offset = Vec3::new(lead_in, 0.0, 0.0);
    for p in &body {
        pts.push(offset + *p);
    }
    // Chain a second primitive half the time so "what comes next" varies.
    if rng.random_bool(0.5) && kind != Kind::HoverSprint {
        let k2 = KINDS[rng.random_range(0..KINDS.len())];
        let k2 = if k2 == Kind::HoverSprint { Kind::Sprint } else { k2 };
        let (p2, h2) = template(k2, rng);
        let anchor = *pts.last().unwrap();
        let yaw2 = heading.y.atan2(heading.x);
        let bridge = rng.random_range(0.8..2.0f32);
        pts.push(anchor + heading * bridge);
        for p in p2 {
            pts.push(anchor + heading * bridge + rot_z(p, yaw2));
        }
        heading = rot_z(h2, yaw2);
    }
    // Lead-out straight along the exit heading.
    let anchor = *pts.last().unwrap();
    let lead_out = rng.random_range(1.0..3.0f32);
    pts.push(anchor + heading * (lead_out * 0.5));
    pts.push(anchor + heading * lead_out);
    // Orient and place, then lift the whole plan so the reference never
    // comes within 1.2 m of the ground: a split-S descends two radii from
    // its entry, and a vehicle recovering onto it overshoots below it.
    let mut pts: Vec<Vec3> = pts.into_iter().map(|p| start + rot_z(p, yaw)).collect();
    let z_min = pts.iter().map(|p| p.z).fold(start.z, f32::min);
    let lift = (1.2 - z_min).max(0.0);
    let start = start + Vec3::new(0.0, 0.0, lift);
    for p in pts.iter_mut() {
        p.z += lift;
    }
    let heading = rot_z(heading, yaw);
    let v_in = if kind == Kind::HoverSprint { ZERO3 } else { rot_z(Vec3::x(), yaw) * speed };
    // End at rest 30 % of the time (brake), otherwise carry the speed out.
    let v_out = if rng.random_bool(0.3) { ZERO3 } else { heading * speed };

    // Durations from segment length at the commanded speed, with a floor
    // that keeps the MINCO pieces well conditioned.
    let n = pts.len();
    let mut durations = Vec::with_capacity(n);
    let mut prev = start;
    for p in &pts {
        let d = (p - prev).norm();
        durations.push((d / speed).max(0.08));
        prev = *p;
    }
    if v_in == ZERO3 {
        durations[0] *= 1.8;
    }
    if v_out == ZERO3 {
        durations[n - 1] *= 1.8;
    }

    // Time scaling: slow the whole plan down uniformly (durations and
    // boundary velocities together, or the head would have to brake from
    // the commanded speed to the stretched pace) until the flatness
    // demand fits the sampled budget — and speed it *up* while it sits
    // well inside the budget, so the family is thrust-limited the way
    // time-optimal missions are rather than stuck at whatever the
    // commanded speed happened to demand.
    let intermediate: Vec<Vec3> = pts[..n - 1].to_vec();
    let mut minco = Box::new(MincoSnap::new(&[start, v_in, ZERO3, ZERO3], &[pts[n - 1], v_out, ZERO3, ZERO3], n));
    let mut stretches = 0u32;
    let mut scale = 1.0f32;
    let mut last_dir = 0i8;
    let (traj, pt, pr) = loop {
        let head = [start, v_in / scale, ZERO3, ZERO3];
        let tail = [pts[n - 1], v_out / scale, ZERO3, ZERO3];
        *minco = MincoSnap::new(&head, &tail, n);
        let d: Vec<f32> = durations.iter().map(|d| d * scale).collect();
        minco.solve(&intermediate, &d);
        let traj = minco.get_trajectory();
        let (pt, pr) = peak_demand(&traj, env);
        let too_hard = pt > thrust_budget || pr > rate_budget;
        // Speed-up is capped: a straight sprint never reaches the thrust
        // budget, and 15 m/s is the top of the band the missions fly.
        // Speed-up stops at the envelope top, at a 10× time compression, and
        // at a 1.2 s minimum episode (a straight sprint never reaches the
        // thrust budget and would otherwise shrink to nothing).
        let dur_total: f32 = durations.iter().sum::<f32>() * scale;
        let too_easy = pt < 0.85 * thrust_budget && scale > 0.1 && speed / scale < 65.0 && dur_total * 0.92 > 1.2;
        let dir: i8 = if too_hard { 1 } else if too_easy { -1 } else { 0 };
        // Stop when inside the band, when the direction flips (the band
        // is between two grid points), or at the iteration cap.
        if dir == 0 || (last_dir != 0 && dir != last_dir) || stretches >= 30 {
            break (traj, pt, pr);
        }
        scale *= if dir > 0 { 1.1 } else { 0.92 };
        last_dir = dir;
        stretches += 1;
    };
    Primitive {
        traj,
        kind,
        speed,
        thrust_budget,
        peak_thrust_frac: pt,
        peak_rate_frac: pr,
        stretches,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Every kind must plan and stay inside its sampled envelope budget
    /// (up to the stretch cap), and the family must actually sweep the
    /// thrust margin rather than sit at one end of it.
    #[test]
    fn primitives_cover_the_envelope() {
        let env = Envelope {
            mass_kg: 0.6,
            grav: 9.81,
            thrust_max_n: 30.0,
            rate_max: Vec3::new(10.0, 10.0, 6.0),
        };
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let mut peak = Vec::new();
        let mut by_kind: std::collections::BTreeMap<&str, Vec<f32>> = Default::default();
        for _ in 0..200 {
            let p = random_primitive(&mut rng, &env, Vec3::new(0.0, 0.0, 1.5));
            by_kind.entry(p.kind.name()).or_default().push(p.peak_thrust_frac);
            assert!(p.traj.total_duration() > 0.3, "{:?}", p.kind);
            assert!(p.traj.get_pos(0.0).iter().all(|v| v.is_finite()));
            assert!(p.peak_thrust_frac < 1.3 && p.peak_thrust_frac.is_finite(), "{:?} never fit: pt={} pr={}", p.kind, p.peak_thrust_frac, p.peak_rate_frac);
            peak.push(p.peak_thrust_frac);
        }
        let lo = peak.iter().filter(|&&f| f < 0.5).count();
        let hi = peak.iter().filter(|&&f| f > 0.7).count();
        for (k, v) in &by_kind {
            eprintln!("{k:<18} n={:3} mean pt={:.2} max={:.2}", v.len(), v.iter().sum::<f32>() / v.len() as f32, v.iter().cloned().fold(0.0, f32::max));
        }
        let mut sorted = peak.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("peak thrust frac quartiles: {:.2} {:.2} {:.2} {:.2} {:.2}",
            sorted[0], sorted[50], sorted[100], sorted[150], sorted[199]);
        assert!(lo > 10 && hi >= 8, "margin sweep too narrow: lo={lo} hi={hi}");
    }
}
