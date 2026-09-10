//! Mission YAML → validated schedule loader — the parser behind the
//! firmware's `missions/*.yaml` bake (`crates/cybflight/build.rs`).
//!
//! Accepts the offline planner's native output shape (`start`,
//! `waypoints`, per-segment `durations`) as well as absolute
//! `timestamps`, and performs mechanically what used to be done by hand
//! when transcribing planner results into Rust arrays:
//! - `durations` → cumulative absolute timestamps;
//! - dropping a leading waypoint that duplicates `start` (some planner
//!   exports include the head pose as `waypoints[0]`, but the firmware's
//!   MINCO contract takes the head from the live setpoint);
//! - structural validation (finite, matching lengths, strictly
//!   increasing positive timing) so a malformed mission fails the build,
//!   not the flight.

/// Tolerance for the "waypoints[0] duplicates start" planner quirk [m].
const HEAD_DUP_EPS: f32 = 1e-3;

/// Which differential-flatness map turns the mission's flat outputs
/// (acceleration/jerk + yaw) into attitude and body-rate references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlatnessMapKind {
    /// Intrinsic tilt-then-yaw convention (`flatness_to_state_tilt_yaw`
    /// family). The default.
    #[default]
    TiltYaw,
    /// Compass-heading ("true yaw") convention
    /// (`flatness_to_state_true_yaw`, the Faessler/RPG nominal
    /// reference inputs): body-x heading tracks ψ regardless of tilt.
    TrueYaw,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MissionYaml {
    /// Selection axis: environment (`indoor` | `outdoor`).
    env: String,
    /// Selection axis: trajectory family (e.g. `splits`, `drag`).
    variant: String,
    /// Selection axis: speed tier (e.g. `slow`, `mid`, `fast`).
    speed: String,
    /// Recorded mission start position [m].
    start: [f32; 3],
    /// Waypoints [m]. May include the head pose as the first entry
    /// (planner-native export); it is dropped when it duplicates `start`
    /// and the lengths only work out without it.
    waypoints: Vec<[f32; 3]>,
    /// Absolute time of arrival at each waypoint [s]. Exactly one of
    /// `timestamps` / `durations` must be present.
    #[serde(default)]
    timestamps: Option<Vec<f32>>,
    /// Per-segment durations [s] (planner-native). Converted to
    /// cumulative timestamps at load.
    #[serde(default)]
    durations: Option<Vec<f32>>,
    /// Desired yaw at each waypoint [deg], one per waypoint (pre-head-
    /// drop count). Absent → the firmware holds the yaw setpoint in
    /// force at mission entry (constant-yaw fallback).
    #[serde(default)]
    headings: Option<Vec<f32>>,
    /// Look-forward yaw mode: yaw points at the reference position
    /// `yaw_lookahead_dt_s` ahead along the trajectory. Overrides
    /// `headings` when both are given.
    lookahead: bool,
    /// Lookahead horizon [s] for `lookahead: true`. Required in every
    /// mission file (must lie in [`YAW_LOOKAHEAD_DT_RANGE_S`]) so the
    /// value travels with the mission rather than a global runtime param.
    yaw_lookahead_dt_s: f32,
    /// Slew-rate cap [rad/s] on the lookahead yaw reference: the raw
    /// look-at heading flips ~180° on path reversals and differs from the
    /// entry yaw at mission start, and the firmware sweeps through both at
    /// this rate (further capped by the vehicle's yaw-rate bound).
    /// Required in every mission file (finite, > 0).
    yaw_lookahead_max_rate_rad_s: f32,
    /// Flatness-map convention for this mission's references
    /// (`tilt_yaw` | `true_yaw`). Defaults to `tilt_yaw`.
    #[serde(default)]
    flatness_map: FlatnessMapKind,
}

/// A validated mission schedule (firmware-convention: `waypoints[i]`
/// reached at absolute `timestamps[i]`, head pose NOT included).
#[derive(Debug, Clone, PartialEq)]
pub struct Mission {
    pub env: String,
    pub variant: String,
    pub speed: String,
    pub start: [f32; 3],
    pub waypoints: Vec<[f32; 3]>,
    pub timestamps: Vec<f32>,
    /// Desired yaw at each waypoint [rad] (converted from YAML degrees),
    /// or `None` for the constant-entry-yaw fallback.
    pub headings: Option<Vec<f32>>,
    pub lookahead: bool,
    pub yaw_lookahead_dt_s: f32,
    pub yaw_lookahead_max_rate_rad_s: f32,
    pub flatness_map: FlatnessMapKind,
    /// Non-fatal findings for the caller to surface (the firmware bake
    /// prints them as `cargo:warning`).
    pub warnings: Vec<String>,
}

/// Accepted `yaw_lookahead_dt_s` range [s]. Below ~0.2 s the chord only
/// clears the 5 cm look-at threshold above ~0.25 m/s, so yaw holds the
/// entry value through the whole take-off acceleration and then jumps;
/// above ~1 s the heading cuts corners badly (it looks across the
/// curve rather than along it).
pub const YAW_LOOKAHEAD_DT_RANGE_S: (f32, f32) = (0.2, 1.0);

/// Parse and validate one mission file. `label` is used in error messages
/// (typically the file stem, which becomes the mission name).
pub fn load_mission(label: &str, yaml: &str) -> Result<Mission, String> {
    let my: MissionYaml = serde_yaml::from_str(yaml).map_err(|e| format!("{label}: {e}"))?;

    match my.env.as_str() {
        "indoor" | "outdoor" => {}
        other => {
            return Err(format!("{label}: env must be indoor|outdoor, got {other:?}"));
        }
    }
    if !my.start.iter().all(|v| v.is_finite()) {
        return Err(format!("{label}: start must be finite"));
    }
    if my.waypoints.is_empty() {
        return Err(format!("{label}: waypoints must not be empty"));
    }
    if !my.waypoints.iter().flatten().all(|v| v.is_finite()) {
        return Err(format!("{label}: waypoints must be finite"));
    }
    let (dt_lo, dt_hi) = YAW_LOOKAHEAD_DT_RANGE_S;
    if !(my.yaw_lookahead_dt_s >= dt_lo && my.yaw_lookahead_dt_s <= dt_hi) {
        return Err(format!(
            "{label}: yaw_lookahead_dt_s must lie in [{dt_lo}, {dt_hi}] s, got {}",
            my.yaw_lookahead_dt_s
        ));
    }
    if !(my.yaw_lookahead_max_rate_rad_s.is_finite() && my.yaw_lookahead_max_rate_rad_s > 0.0) {
        return Err(format!(
            "{label}: yaw_lookahead_max_rate_rad_s must be finite and > 0, got {}",
            my.yaw_lookahead_max_rate_rad_s
        ));
    }
    let mut warnings = Vec::new();

    // Headings are validated against the pre-head-drop waypoint count
    // (they pair 1:1 with the waypoints as written), even when
    // `lookahead: true` supersedes them below — a malformed list is a
    // typo worth failing the build for either way.
    let mut headings = my.headings;
    if let Some(h) = &headings {
        if h.len() != my.waypoints.len() {
            return Err(format!(
                "{label}: {} headings vs {} waypoints",
                h.len(),
                my.waypoints.len()
            ));
        }
        if !h.iter().all(|v| v.is_finite()) {
            return Err(format!("{label}: headings must be finite"));
        }
    }
    if my.lookahead {
        // Lookahead wins: the explicit list is superseded.
        if headings.take().is_some() {
            warnings.push(format!(
                "{label}: `headings` ignored — `lookahead: true` supersedes them"
            ));
        }
    }

    let timestamps: Vec<f32> = match (&my.timestamps, &my.durations) {
        (Some(ts), None) => ts.clone(),
        (None, Some(ds)) => {
            if !ds.iter().all(|d| d.is_finite() && *d > 0.0) {
                return Err(format!("{label}: durations must be finite and > 0"));
            }
            ds.iter()
                .scan(0.0f32, |acc, d| {
                    *acc += d;
                    Some(*acc)
                })
                .collect()
        }
        (Some(_), Some(_)) => {
            return Err(format!(
                "{label}: give exactly one of timestamps / durations, not both"
            ));
        }
        (None, None) => {
            return Err(format!("{label}: one of timestamps / durations is required"));
        }
    };

    // Planner-native exports sometimes include the head pose as
    // waypoints[0]. Drop it iff the lengths only work out without it AND
    // it actually duplicates `start` — anything else is a real mismatch.
    let mut waypoints = my.waypoints;
    if waypoints.len() == timestamps.len() + 1 {
        let d2: f32 = waypoints[0]
            .iter()
            .zip(my.start.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if d2.sqrt() <= HEAD_DUP_EPS {
            waypoints.remove(0);
            // Headings pair 1:1 with waypoints as written — drop in sync.
            if let Some(h) = headings.as_mut() {
                h.remove(0);
            }
        } else {
            return Err(format!(
                "{label}: {} waypoints vs {} timing entries, and waypoints[0] \
                 does not duplicate start (dist {} m) — cannot reconcile",
                waypoints.len(),
                timestamps.len(),
                d2.sqrt(),
            ));
        }
    }
    if waypoints.len() != timestamps.len() {
        return Err(format!(
            "{label}: {} waypoints vs {} timing entries",
            waypoints.len(),
            timestamps.len(),
        ));
    }
    // Re-check AFTER the head-dup drop: `waypoints: [start]` with an
    // empty timing list would otherwise slip through as a zero-piece
    // mission and only fail later as an opaque const assert.
    if waypoints.is_empty() {
        return Err(format!(
            "{label}: mission has no pieces after dropping the duplicate head waypoint"
        ));
    }

    let mut prev = 0.0f32;
    for (i, t) in timestamps.iter().enumerate() {
        if !(t.is_finite() && *t > prev) {
            return Err(format!(
                "{label}: timestamps must be finite and strictly increasing from > 0 \
                 (entry {i} = {t}, previous {prev})"
            ));
        }
        prev = *t;
    }

    Ok(Mission {
        env: my.env,
        variant: my.variant,
        speed: my.speed,
        start: my.start,
        waypoints,
        timestamps,
        headings: headings.map(|h| h.iter().map(|d| d.to_radians()).collect()),
        lookahead: my.lookahead,
        yaw_lookahead_dt_s: my.yaw_lookahead_dt_s,
        yaw_lookahead_max_rate_rad_s: my.yaw_lookahead_max_rate_rad_s,
        flatness_map: my.flatness_map,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
env: outdoor
variant: drag
speed: slow
start: [0.0, 0.0, 3.0]
waypoints:
  - [0.0, 1.0, 3.0]
  - [0.0, 2.0, 3.0]
timestamps: [0.5, 1.0]
lookahead: false
yaw_lookahead_dt_s: 0.5
yaw_lookahead_max_rate_rad_s: 4.0
"#;

    #[test]
    fn loads_timestamps_form() {
        let m = load_mission("t", MINIMAL).unwrap();
        assert_eq!(m.waypoints.len(), 2);
        assert_eq!(m.timestamps, vec![0.5, 1.0]);
    }

    #[test]
    fn durations_convert_to_cumulative_timestamps() {
        let yaml = MINIMAL.replace("timestamps: [0.5, 1.0]", "durations: [0.5, 0.5]");
        let m = load_mission("t", &yaml).unwrap();
        assert!((m.timestamps[0] - 0.5).abs() < 1e-7);
        assert!((m.timestamps[1] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn both_or_neither_timing_is_an_error() {
        let both = MINIMAL.replace(
            "timestamps: [0.5, 1.0]",
            "timestamps: [0.5, 1.0]\ndurations: [0.5, 0.5]",
        );
        assert!(load_mission("t", &both).unwrap_err().contains("exactly one"));
        let neither = MINIMAL.replace("timestamps: [0.5, 1.0]", "");
        assert!(load_mission("t", &neither).unwrap_err().contains("required"));
    }

    #[test]
    fn duplicate_head_waypoint_is_dropped() {
        let yaml = MINIMAL.replace(
            "waypoints:",
            "waypoints:\n  - [0.0, 0.0, 3.0]",
        );
        // 3 waypoints, 2 timestamps, wp[0] == start → dropped.
        let m = load_mission("t", &yaml).unwrap();
        assert_eq!(m.waypoints.len(), 2);
        assert_eq!(m.waypoints[0], [0.0, 1.0, 3.0]);
    }

    #[test]
    fn length_mismatch_without_head_dup_is_an_error() {
        let yaml = MINIMAL.replace("waypoints:", "waypoints:\n  - [9.0, 9.0, 9.0]");
        assert!(load_mission("t", &yaml).unwrap_err().contains("cannot reconcile"));
    }

    #[test]
    fn non_monotonic_timestamps_are_an_error() {
        let yaml = MINIMAL.replace("timestamps: [0.5, 1.0]", "timestamps: [1.0, 0.5]");
        assert!(load_mission("t", &yaml).unwrap_err().contains("strictly increasing"));
    }

    #[test]
    fn bad_env_is_an_error() {
        let yaml = MINIMAL.replace("env: outdoor", "env: moonbase");
        assert!(load_mission("t", &yaml).unwrap_err().contains("indoor|outdoor"));
    }

    #[test]
    fn unknown_field_is_an_error() {
        let yaml = format!("{MINIMAL}\nvelocity_limit: 3.0\n");
        assert!(load_mission("t", &yaml).is_err());
    }

    #[test]
    fn headings_parse_in_degrees_to_radians() {
        let yaml = format!("{MINIMAL}headings: [90.0, -180.0]\n");
        let m = load_mission("t", &yaml).unwrap();
        let h = m.headings.unwrap();
        assert!((h[0] - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((h[1] + core::f32::consts::PI).abs() < 1e-6);
        assert!(!m.lookahead);
    }

    #[test]
    fn missing_headings_is_none() {
        let m = load_mission("t", MINIMAL).unwrap();
        assert_eq!(m.headings, None);
    }

    #[test]
    fn headings_length_mismatch_is_an_error() {
        let yaml = format!("{MINIMAL}headings: [0.0]\n");
        let err = load_mission("t", &yaml).unwrap_err();
        assert!(err.contains("headings"), "{err}");
    }

    #[test]
    fn non_finite_heading_is_an_error() {
        let yaml = format!("{MINIMAL}headings: [0.0, .nan]\n");
        let err = load_mission("t", &yaml).unwrap_err();
        assert!(err.contains("finite"), "{err}");
    }

    #[test]
    fn headings_dropped_in_sync_with_duplicate_head() {
        let yaml = MINIMAL
            .replace("waypoints:", "waypoints:\n  - [0.0, 0.0, 3.0]")
            + "headings: [45.0, 90.0, 135.0]\n";
        // 3 waypoints (dup head) + 3 headings → both drop their first entry.
        let m = load_mission("t", &yaml).unwrap();
        assert_eq!(m.waypoints.len(), 2);
        let h = m.headings.unwrap();
        assert_eq!(h.len(), 2);
        assert!((h[0] - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn lookahead_wins_over_headings() {
        let yaml = MINIMAL.replace("lookahead: false", "lookahead: true")
            + "headings: [0.0, 90.0]\n";
        let m = load_mission("t", &yaml).unwrap();
        assert!(m.lookahead);
        assert_eq!(m.headings, None);
        assert_eq!(m.warnings.len(), 1);
        assert!(m.warnings[0].contains("headings"), "{:?}", m.warnings);
    }

    #[test]
    fn plain_mission_has_no_warnings() {
        assert!(load_mission("t", MINIMAL).unwrap().warnings.is_empty());
    }

    #[test]
    fn missing_lookahead_keys_are_an_error() {
        let no_lookahead = MINIMAL.replace("lookahead: false\n", "");
        assert!(load_mission("t", &no_lookahead).is_err());
        let no_dt = MINIMAL.replace("yaw_lookahead_dt_s: 0.5\n", "");
        assert!(load_mission("t", &no_dt).is_err());
        let no_rate = MINIMAL.replace("yaw_lookahead_max_rate_rad_s: 4.0\n", "");
        assert!(load_mission("t", &no_rate).is_err());
    }

    #[test]
    fn out_of_range_yaw_lookahead_dt_is_an_error() {
        for bad in ["0.05", "2.0", ".nan"] {
            let yaml = MINIMAL.replace("yaw_lookahead_dt_s: 0.5", &format!("yaw_lookahead_dt_s: {bad}"));
            let err = load_mission("t", &yaml).unwrap_err();
            assert!(err.contains("yaw_lookahead_dt_s"), "{bad}: {err}");
        }
    }

    #[test]
    fn non_positive_yaw_lookahead_max_rate_is_an_error() {
        for bad in ["0.0", "-1.0", ".inf"] {
            let yaml = MINIMAL.replace(
                "yaw_lookahead_max_rate_rad_s: 4.0",
                &format!("yaw_lookahead_max_rate_rad_s: {bad}"),
            );
            let err = load_mission("t", &yaml).unwrap_err();
            assert!(err.contains("yaw_lookahead_max_rate_rad_s"), "{bad}: {err}");
        }
    }

    #[test]
    fn flatness_map_defaults_to_tilt_yaw() {
        let m = load_mission("t", MINIMAL).unwrap();
        assert_eq!(m.flatness_map, FlatnessMapKind::TiltYaw);
    }

    #[test]
    fn flatness_map_true_yaw_parses() {
        let yaml = format!("{MINIMAL}flatness_map: true_yaw\n");
        let m = load_mission("t", &yaml).unwrap();
        assert_eq!(m.flatness_map, FlatnessMapKind::TrueYaw);
    }

    #[test]
    fn unknown_flatness_map_is_an_error() {
        let yaml = format!("{MINIMAL}flatness_map: hopf_fibration\n");
        assert!(load_mission("t", &yaml).is_err());
    }

    #[test]
    fn non_positive_yaw_lookahead_dt_is_an_error() {
        let yaml = MINIMAL.replace("yaw_lookahead_dt_s: 0.5", "yaw_lookahead_dt_s: 0.0");
        let err = load_mission("t", &yaml).unwrap_err();
        assert!(err.contains("yaw_lookahead_dt_s"), "{err}");
    }
}
