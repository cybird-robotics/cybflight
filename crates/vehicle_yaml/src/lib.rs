//! Vehicle YAML → [`FirmwareConfig`] loader — the single parser behind
//! both the firmware bake (`crates/cybflight/build.rs`) and the host
//! simulation baseline (`crates/cybflight_sim/src/plant.rs`). One schema,
//! one loader; a firmware/sim divergence is a *file diff* between
//! `vehicles/*.yaml`, never a code drift.
//!
//! Format (see `vehicles/sakura_bench.yaml` for a commented example):
//! - `airframe:` — REQUIRED physical identity (mass, inertia, max rates,
//!   exactly 4 motors). No defaults; a missing field is an error — this is
//!   the "no default mass" rule of docs/param_redesign_plan.md.
//! - `origin:` — optional fixed geodetic ENU anchor (`lat_deg`,
//!   `lon_deg`, `alt_msl_m`). Its own section rather than `tuning:` keys
//!   because that map is `f32` and an f32 latitude quantises to ~0.6 m.
//! - `tuning:` — optional flat map of registry parameter names to values,
//!   validated against the live registry (unknown name or out-of-range
//!   value is an error) and applied over the schema defaults.

pub mod mission;

use std::collections::{BTreeMap, BTreeSet};

use cybflight_core::mixer::SpinDir;
use cybflight_core::param_registry::ParamGroup;
use cybflight_core::params::FirmwareConfig;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct VehicleYaml {
    #[serde(default)]
    build: Option<BuildYaml>,
    /// Default offline mission for this vehicle, by name
    /// (= `missions/<name>.yaml` file stem). Resolved and env-validated
    /// by the firmware bake; the sim ignores it.
    #[serde(default)]
    default_mission: Option<String>,
    airframe: AirframeYaml,
    /// Optional fixed ENU origin. Absent → the firmware anchors on the
    /// first RTK-fixed PVT (the historical behaviour).
    #[serde(default)]
    origin: Option<OriginYaml>,
    /// Optional host-simulation plant physics. Ignored by the firmware
    /// bake — these describe effects the *plant* must model but the
    /// controller has no use for (aerodynamic drag, per-motor thrust
    /// scatter). See [`SimYaml`].
    #[serde(default)]
    sim: Option<SimYaml>,
    /// Optional learned-cost policy to bake, by name
    /// (= `crates/cybflight/data/cost_policies/<name>.bin` file stem).
    /// Absent → no policy baked, no flash cost. Consumed by the firmware
    /// bake; the sim loads policies from paths at runtime instead.
    #[serde(default)]
    mpc_cost_policy: Option<String>,
    #[serde(default)]
    tuning: BTreeMap<String, f32>,
}

/// Host-simulation plant physics — the deliberately *unmodelled* part of
/// the vehicle, from the controller's point of view.
///
/// Everything the controller also needs (rotor time constant, max rotor
/// speed, G2 yaw coefficient, throttle-curve `k`) already lives in the
/// parameter registry and is pinned via `tuning:`; duplicating it here
/// would create exactly the two-sources-of-truth drift the YAML bake
/// exists to prevent. What remains is plant-only, and that is what this
/// section carries.
///
/// Defaults are all-zero / neutral, i.e. the drag-free, perfectly-matched
/// plant that predates rotor-state simulation. A vehicle that declares no
/// `sim:` section therefore behaves exactly as before.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimYaml {
    /// Rotor-speed-proportional aerodynamic drag `[x, y, z]` in the body
    /// (FLU) frame, in N·s²/(m·rad):
    ///
    /// ```text
    /// F_drag,axis = −c_axis · (Σᵢ ωᵢ) · v_body,axis
    /// ```
    ///
    /// This is the standard rotor-drag / H-force form: linear in body
    /// velocity, scaled by total rotor speed. To convert a specific-force
    /// coefficient `k` identified against accelerometer data (as in
    /// `a_x = −k_x · v_bx · Σω`), multiply by mass: `c_x = m · k_x`.
    #[serde(default)]
    pub aero_drag: [f32; 3],
    /// Quadratic (parasitic) body drag `[x, y, z]` in the body frame,
    /// `½·ρ·C_d·A` per axis in N·s²/m²: `F_axis = −k_axis·|v_axis|·v_axis`.
    /// Plant-only; the controllers never model it. Default 0 (the frozen
    /// baseline has none), so it is a deliberate unmodelled-uncertainty
    /// knob for robustness studies. A 5-inch quad is ≈ 0.006–0.025.
    #[serde(default)]
    pub body_drag: [f32; 3],
    /// Rotor speed [rad/s] at zero throttle command — the ESC idle floor
    /// in the steady-state throttle map. Non-zero values shift the
    /// effective curvature of the command→thrust curve away from the
    /// analytic model the controller inverts, which is a real and
    /// deliberate source of plant/model mismatch.
    #[serde(default)]
    pub rotor_omega_min_rad_s: f32,
    /// Polar moment of inertia of one rotor + prop [kg·m²]. Drives the
    /// yaw reaction torque `τ_z += Σᵢ sᵢ·J_r·ω̇ᵢ` and the rotor
    /// gyroscopic term. Relates to a specific-torque identification
    /// `ṙ += k_rd·Σ±ω̇ᵢ` by `J_r = I_zz · k_rd`.
    #[serde(default)]
    pub rotor_inertia_kg_m2: f32,
    /// Plant-side throttle-curve curvature `k ∈ [0, 1]` in
    /// `ω_c = (ω_max − ω_min)·√(k·d² + (1−k)·d) + ω_min`.
    ///
    /// Deliberately separate from the controller's `indi_nonlin_m*`: the
    /// controller's value is clamped to the range over which its analytic
    /// *inverse* is well conditioned, while the plant is free to be as
    /// curved as the hardware actually is. Zero selects a linear map.
    #[serde(default)]
    pub rotor_throttle_curve_k: f32,
    /// Per-motor thrust-coefficient multiplier, applied to `c_T`. Models
    /// build scatter (a tired motor, a chipped prop) and, at extreme
    /// values, motor faults. Scales thrust AND every moment that motor
    /// contributes, which is the physically coupled way to inject
    /// asymmetry. Empty or absent → all 1.0.
    #[serde(default)]
    pub motor_thrust_scale: Option<[f32; 4]>,
}

impl Default for SimYaml {
    fn default() -> Self {
        Self {
            aero_drag: [0.0; 3],
            body_drag: [0.0; 3],
            rotor_omega_min_rad_s: 0.0,
            rotor_inertia_kg_m2: 0.0,
            rotor_throttle_curve_k: 0.0,
            motor_thrust_scale: None,
        }
    }
}

impl SimYaml {
    /// Per-motor thrust scale with the `None` case resolved to unity.
    pub fn thrust_scale(&self) -> [f32; 4] {
        self.motor_thrust_scale.unwrap_or([1.0; 4])
    }

    fn validate(&self, label: &str) -> Result<(), String> {
        if !self.aero_drag.iter().all(|v| v.is_finite() && *v >= 0.0) {
            return Err(format!(
                "{label}: sim.aero_drag must be finite and >= 0, got {:?}",
                self.aero_drag
            ));
        }
        if !(self.rotor_omega_min_rad_s.is_finite() && self.rotor_omega_min_rad_s >= 0.0) {
            return Err(format!(
                "{label}: sim.rotor_omega_min_rad_s must be finite and >= 0"
            ));
        }
        if !(self.rotor_inertia_kg_m2.is_finite() && self.rotor_inertia_kg_m2 >= 0.0) {
            return Err(format!(
                "{label}: sim.rotor_inertia_kg_m2 must be finite and >= 0"
            ));
        }
        if !(self.rotor_throttle_curve_k.is_finite()
            && (0.0..=1.0).contains(&self.rotor_throttle_curve_k))
        {
            return Err(format!(
                "{label}: sim.rotor_throttle_curve_k must be in [0, 1], got {}",
                self.rotor_throttle_curve_k
            ));
        }
        if let Some(s) = &self.motor_thrust_scale {
            if !s.iter().all(|v| v.is_finite() && *v > 0.0) {
                return Err(format!(
                    "{label}: sim.motor_thrust_scale entries must be finite and > 0, got {s:?}"
                ));
            }
        }
        Ok(())
    }
}

/// Fixed geodetic anchor for the ENU frame.
///
/// **Not a `tuning:` key.** That map is `f32`, and an f32 latitude has a
/// ULP of ~0.6 m at mid-latitudes — it would silently throw away the
/// centimetre accuracy the RTK-fixed anchor exists to provide. Latitude
/// and longitude are therefore `f64` in their own section, matching
/// `LlhOrigin` and the receiver's own 1e-7-degree integers.
///
/// Pin this when ENU coordinates must mean the same physical place on
/// every flight — mission waypoints are expressed in ENU, so a
/// per-flight origin puts the same mission in a different spot each
/// time it is flown.
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginYaml {
    /// Geodetic latitude [deg], WGS84.
    pub lat_deg: f64,
    /// Geodetic longitude [deg], WGS84.
    pub lon_deg: f64,
    /// Altitude above mean sea level [m], matching the receiver's
    /// `alt_msl` (not ellipsoidal height).
    pub alt_msl_m: f32,
}

/// Optional `build:` section — compile-time selections that are per-vehicle
/// hardware facts (which PCB, receiver wiring, GNSS unit, …). Consumed by
/// the firmware build tooling (`tools/vehicle_features.py`, `build.rs`
/// cross-check); the sim ignores it. Every knob is optional so partial
/// declarations validate; the build tooling decides what absence means.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildYaml {
    /// Flight-controller PCB: `sakurah743` | `foxeerh743` | `micoair743v2`.
    #[serde(default)]
    pub board: Option<String>,
    /// RC receiver protocol (implies wiring): `crsf` (2-wire full-duplex)
    /// | `ghst` (1-wire half-duplex).
    #[serde(default)]
    pub rc_protocol: Option<String>,
    /// Outer-loop controller: `mpc` | `cascade` | `rate`.
    #[serde(default)]
    pub outer_loop: Option<String>,
    /// ESKF position source: `mocap` | `gps`.
    #[serde(default)]
    pub pos_source: Option<String>,
    /// GNSS receiver driver: `ublox` | `unicore` (UM982, heading-capable).
    #[serde(default)]
    pub gps_model: Option<String>,
    /// Is a second GNSS antenna (ANT2) populated on this install?
    /// `yes` | `no` (default). Independent of `gps_model`: a UM982 is
    /// heading-*capable*, but can still be wired with one antenna — which
    /// is exactly how `sakura_bench_hunter_outdoor` flies.
    ///
    /// A `build:` knob rather than a tuning param because it is an
    /// immutable fact about the airframe's wiring, and because only here
    /// can it be cross-checked against `gps_model`: `yes` on a receiver
    /// that cannot report a heading is a build error rather than a boot
    /// warning nobody reads. `gps_base_*` (where ANT2 sits) stays a
    /// runtime param — the same capability-vs-calibration split as
    /// `gps_model` vs `gps_ant_*`.
    #[serde(default)]
    pub gps_dual_antenna: Option<String>,
    /// Gimbal role: `leader` | `chaser`; omit for no role.
    #[serde(default)]
    pub role: Option<String>,
    /// Primary-IMU output data rate / inner-loop rate: `8khz` (low-latency,
    /// the default) | `1khz` (ICM426xx low-noise mode). ICM boards only.
    #[serde(default)]
    pub imu_rate: Option<String>,
    /// Run the INDI inner loop? `yes` (default) | `no`.
    ///
    /// `no` drops the incremental terms and leaves a proportional rate
    /// controller allocating through G1 — indiflight's `useIncrement =
    /// false`. A `build:` knob rather than a tuning param for the same
    /// reason as `outer_loop`: which control law flies the airframe is a
    /// vehicle fact, it decides how `indi_rate_*` must be tuned, and it
    /// must not be reachable by a stray `param set` on an armed vehicle.
    #[serde(default)]
    pub indi: Option<String>,
    /// Compile the online BFGS trajectory optimizer (`yes`) or fly only
    /// the missions baked from `missions/*.yaml` (`no`, the default).
    ///
    /// A build knob rather than a parameter for the reason in
    /// optimization rule 7: the online path owns ~55 KiB of solver
    /// `.bss` plus a similar amount of task-future state held across
    /// awaits, and neither is removed by const-folding a dead branch. A
    /// runtime toggle would pay the memory on every vehicle to give the
    /// option to one.
    #[serde(default)]
    pub plan_online: Option<String>,
}

impl BuildYaml {
    /// Validate every present knob against its allowed values. `label` is
    /// used in error messages. Shared by the firmware bake and host
    /// tooling so there is exactly one authority on legal values.
    pub fn validate(&self, label: &str) -> Result<(), String> {
        let check = |knob: &str, v: &Option<String>, allowed: &[&str]| -> Result<(), String> {
            match v {
                Some(s) if !allowed.contains(&s.as_str()) => Err(format!(
                    "{label}: build.{knob} must be one of {allowed:?}, got {s:?}"
                )),
                _ => Ok(()),
            }
        };
        check("board", &self.board, &["sakurah743", "foxeerh743", "micoair743v2"])?;
        check("rc_protocol", &self.rc_protocol, &["crsf", "ghst"])?;
        check(
            "outer_loop",
            &self.outer_loop,
            &["mpc", "mpc_full", "cascade", "rate"],
        )?;
        check("pos_source", &self.pos_source, &["mocap", "gps"])?;
        check("gps_model", &self.gps_model, &["ublox", "unicore"])?;
        check("gps_dual_antenna", &self.gps_dual_antenna, &["yes", "no"])?;
        check("role", &self.role, &["leader", "chaser"])?;
        check("imu_rate", &self.imu_rate, &["8khz", "1khz"])?;
        check("indi", &self.indi, &["yes", "no"])?;
        check("plan_online", &self.plan_online, &["yes", "no"])?;

        // Cross-knob: ANT2 is only reachable through a receiver that
        // reports a heading. Declaring the antenna on a u-blox build used
        // to be a runtime warning on a link nobody watches; it is the
        // whole reason this moved out of `tuning:`.
        if self.gps_dual_antenna.as_deref() == Some("yes")
            && self.gps_model.as_deref() != Some("unicore")
        {
            return Err(format!(
                "{label}: build.gps_dual_antenna: yes requires build.gps_model: unicore \
                 (got {:?}) — only the UM982 driver reports a dual-antenna heading, so \
                 ANT2 would be wired but never fused",
                self.gps_model.as_deref().unwrap_or("unset"),
            ));
        }
        Ok(())
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AirframeYaml {
    /// Stable identity of the **physical airframe** — not of this file.
    ///
    /// One drone has one name, and the same drone flown in two
    /// environments is two vehicle files sharing one name. That makes the
    /// invariant checkable: any two vehicle YAMLs with the same
    /// `airframe.name` must declare an identical `airframe:` block, since
    /// mass, inertia and motor geometry are facts about the hardware, not
    /// about where it flies. `same_name_airframes_agree` enforces it.
    ///
    /// It is also the join key to the rest of the toolchain: the mocap
    /// rigid-body / Vicon subject, and cybgcs's `fleet.toml` roster entry
    /// that binds a drone to an ESP32 IP. Keeping those spellings equal is
    /// what lets a mis-flashed board be detected on the ground instead of
    /// discovered as a jump cascade in the air.
    ///
    /// Optional at the parse level so existing vehicles keep building; the
    /// bake warns when it is absent.
    #[serde(default)]
    name: Option<String>,
    mass_kg: f32,
    inertia_kg_m2: [f32; 9],
    max_rate_rad_s: [f32; 3],
    motors: Vec<MotorYaml>,
    /// Optional at the PARSE level so the sim baseline (which never uses
    /// a thrust model from YAML) stays valid; the firmware bake makes it
    /// REQUIRED — a missing model is a build error, same philosophy as
    /// "no default mass".
    #[serde(default)]
    thrust_model: Option<ThrustModelYaml>,
}

/// Thrust-model declaration. The actual `ThrustModel::Table` pointer is a
/// `&'static ThrustTable` living in the firmware crate's generated code,
/// so this crate only *names* the model — `crates/cybflight/build.rs`
/// resolves it against the baked CSV set.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "type")]
pub enum ThrustModelYaml {
    /// Bench-measured 2D `(thrust, voltage) → command` table; `table` is a
    /// CSV stem under `crates/cybflight/data/thrust_tables/`.
    Table { table: String },
    /// `u = k·d² + (1−k)·d` (indiflight port).
    Quadratic { k: f32 },
    /// `u = (k·d + (1−k)·√d)²` (steady-state ω mix, T ∝ ω²).
    SqrtSquared { k: f32 },
}

/// Full parse result for consumers that need more than the param snapshot
/// (the firmware bake). [`config_from_str`] remains the thin params-only
/// view used by the sim.
#[derive(Debug)]
pub struct VehicleConfig {
    pub params: FirmwareConfig,
    /// Physical-airframe identity from `airframe.name`, if declared. The
    /// join key to the mocap subject and the cybgcs fleet roster; baked
    /// into the firmware so a board can report which airframe it believes
    /// it is.
    pub airframe_name: Option<String>,
    /// The declared thrust model, if any (the firmware bake requires it).
    pub thrust_model: Option<ThrustModelYaml>,
    /// The `build:` section, if declared (validated).
    pub build: Option<BuildYaml>,
    /// The declared default mission name, if any.
    pub default_mission: Option<String>,
    /// Fixed ENU anchor, if declared. `None` → anchor on the first
    /// RTK-fixed PVT at runtime.
    pub origin: Option<OriginYaml>,
    /// Host-simulation plant physics. Always populated — an absent
    /// `sim:` section yields [`SimYaml::default`], the drag-free
    /// perfectly-matched plant.
    pub sim: SimYaml,
    /// Learned-cost policy stem to bake
    /// (`crates/cybflight/data/cost_policies/<name>.bin`), if declared.
    pub mpc_cost_policy: Option<String>,
    /// Keys explicitly present under `tuning:` — lets the bake warn about
    /// unpinned hardware-coupled keys without re-parsing.
    pub tuning_keys: BTreeSet<String>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MotorYaml {
    pos_m: [f32; 2],
    spin: String,
    max_thrust_n: f32,
    torque_coeff_m: f32,
}

/// Parse and validate a vehicle definition, params-only view. `label` is
/// used in error messages (typically the vehicle name or file path).
pub fn config_from_str(label: &str, yaml: &str) -> Result<FirmwareConfig, String> {
    load(label, yaml).map(|v| v.params)
}

/// Parse and validate a vehicle definition, full view (params + thrust
/// model + pinned-key set).
pub fn load(label: &str, yaml: &str) -> Result<VehicleConfig, String> {
    let vy: VehicleYaml = serde_yaml::from_str(yaml).map_err(|e| format!("{label}: {e}"))?;

    if let Some(b) = &vy.build {
        b.validate(label)?;
    }

    if let Some(tm) = &vy.airframe.thrust_model {
        match tm {
            ThrustModelYaml::Table { table } => {
                if table.is_empty()
                    || !table
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    return Err(format!(
                        "{label}: airframe.thrust_model.table must be a CSV stem \
                         (alphanumeric/underscore), got {table:?}"
                    ));
                }
            }
            ThrustModelYaml::Quadratic { k } | ThrustModelYaml::SqrtSquared { k } => {
                if !(k.is_finite() && (0.0..=1.0).contains(k)) {
                    return Err(format!(
                        "{label}: airframe.thrust_model.k must be in [0, 1], got {k}"
                    ));
                }
            }
        }
    }

    let a = &vy.airframe;
    // Identity is a join key across three registries (this file, the mocap
    // subject, cybgcs's fleet.toml) and is baked into the firmware, so
    // restrict it to characters that survive all of them intact.
    if let Some(n) = &a.name
        && (n.is_empty()
            || !n
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'))
    {
        return Err(format!(
            "{label}: airframe.name must be non-empty ASCII alphanumeric/_/- \
             (it is the join key to the mocap subject and the cybgcs fleet \
             roster), got {n:?}"
        ));
    }
    if !(a.mass_kg.is_finite() && a.mass_kg > 0.0) {
        return Err(format!("{label}: airframe.mass_kg must be finite and > 0"));
    }
    // Every entry finite (NaN/inf would otherwise flow into the sim's
    // physics silently, and into invalid emitted literals at the bake);
    // diagonals additionally positive.
    if !a.inertia_kg_m2.iter().all(|v| v.is_finite()) {
        return Err(format!("{label}: airframe.inertia_kg_m2 must be finite"));
    }
    for (i, name) in [(0, "Ixx"), (4, "Iyy"), (8, "Izz")] {
        if a.inertia_kg_m2[i] <= 0.0 {
            return Err(format!("{label}: airframe.inertia_kg_m2 {name} must be > 0"));
        }
    }
    if !a.max_rate_rad_s.iter().all(|&r| r.is_finite() && r > 0.0) {
        return Err(format!(
            "{label}: airframe.max_rate_rad_s must be finite and > 0"
        ));
    }
    if a.motors.len() != 4 {
        return Err(format!("{label}: airframe.motors must list exactly 4 motors"));
    }

    let mut cfg = FirmwareConfig::scaffold();
    cfg.airframe.body.mass_kg = a.mass_kg;
    cfg.airframe.body.inertia_kg_m2 = a.inertia_kg_m2;
    cfg.airframe.body.max_rate_rad_s = a.max_rate_rad_s;
    for (i, m) in a.motors.iter().enumerate() {
        if !(m.max_thrust_n.is_finite() && m.max_thrust_n > 0.0) {
            return Err(format!("{label}: motor {i} max_thrust_n must be finite and > 0"));
        }
        if !m.pos_m.iter().all(|v| v.is_finite()) {
            return Err(format!("{label}: motor {i} pos_m must be finite"));
        }
        if !m.torque_coeff_m.is_finite() {
            return Err(format!("{label}: motor {i} torque_coeff_m must be finite"));
        }
        cfg.airframe.motors[i].position_m = m.pos_m;
        cfg.airframe.motors[i].spin_dir = match m.spin.as_str() {
            "cw" => SpinDir::Cw,
            "ccw" => SpinDir::Ccw,
            other => {
                return Err(format!("{label}: motor {i} spin must be cw|ccw, got {other:?}"));
            }
        };
        cfg.airframe.motors[i].max_thrust_n = m.max_thrust_n;
        cfg.airframe.motors[i].torque_coeff_m = m.torque_coeff_m;
    }

    for (key, value) in &vy.tuning {
        if !value.is_finite() {
            return Err(format!("{label}: tuning.{key} must be finite"));
        }
        let idx = <FirmwareConfig as ParamGroup>::param_find(key).ok_or_else(|| {
            format!("{label}: unknown tuning key {key:?} (not a registry parameter name)")
        })?;
        let meta = FirmwareConfig::param_meta(idx);
        if !meta.in_range(*value) {
            return Err(format!(
                "{label}: tuning.{key} = {value} out of range [{}, {}] {}",
                meta.min, meta.max, meta.unit,
            ));
        }
        // idx came from param_find — set cannot fail.
        cfg.param_set_f32(idx, *value);
    }
    if let Some(o) = &vy.origin {
        if !(o.lat_deg.is_finite() && o.lon_deg.is_finite() && o.alt_msl_m.is_finite()) {
            return Err(format!("{label}: origin lat/lon/alt must be finite"));
        }
        if !(-90.0..=90.0).contains(&o.lat_deg) {
            return Err(format!(
                "{label}: origin.lat_deg = {} out of range [-90, 90]",
                o.lat_deg
            ));
        }
        if !(-180.0..=180.0).contains(&o.lon_deg) {
            return Err(format!(
                "{label}: origin.lon_deg = {} out of range [-180, 180]",
                o.lon_deg
            ));
        }
    }

    let sim = vy.sim.unwrap_or_default();
    sim.validate(label)?;

    Ok(VehicleConfig {
        params: cfg,
        airframe_name: vy.airframe.name,
        thrust_model: vy.airframe.thrust_model,
        build: vy.build,
        default_mission: vy.default_mission,
        origin: vy.origin,
        sim,
        mpc_cost_policy: vy.mpc_cost_policy,
        tuning_keys: vy.tuning.keys().cloned().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
airframe:
  mass_kg: 0.6
  inertia_kg_m2: [0.0021, 0.0, 0.0, 0.0, 0.0018, 0.0, 0.0, 0.0, 0.003]
  max_rate_rad_s: [10.0, 10.0, 6.0]
  motors:
    - { pos_m: [-0.075, -0.1], spin: cw,  max_thrust_n: 12.0, torque_coeff_m: 0.022 }
    - { pos_m: [ 0.075, -0.1], spin: ccw, max_thrust_n: 12.0, torque_coeff_m: 0.022 }
    - { pos_m: [-0.075,  0.1], spin: ccw, max_thrust_n: 12.0, torque_coeff_m: 0.022 }
    - { pos_m: [ 0.075,  0.1], spin: cw,  max_thrust_n: 12.0, torque_coeff_m: 0.022 }
tuning:
  mpc_w_pos_x: 400.0
"#;

    #[test]
    fn origin_absent_is_none() {
        let cfg = load("test", MINIMAL).unwrap();
        assert_eq!(cfg.origin, None);
    }

    /// Latitude must survive the round trip at full f64 precision — the
    /// whole reason `origin:` is its own section instead of three
    /// `tuning:` keys (that map is f32, ~0.6 m per ULP at this latitude).
    #[test]
    fn origin_parses_at_full_precision() {
        let yaml = format!(
            "{MINIMAL}origin:\n  lat_deg: 47.3977419\n  lon_deg: 8.5455938\n  alt_msl_m: 488.5\n"
        );
        let o = load("test", &yaml).unwrap().origin.unwrap();
        assert_eq!(o.lat_deg, 47.3977419_f64);
        assert_eq!(o.lon_deg, 8.5455938_f64);
        assert_eq!(o.alt_msl_m, 488.5_f32);
        // An f32 round trip would lose ~1e-6 deg here; assert we did not.
        assert!(
            (o.lat_deg - 47.3977419_f32 as f64).abs() > 1e-9,
            "test fixture no longer distinguishes f32 from f64 storage"
        );
    }

    #[test]
    fn origin_out_of_range_is_an_error() {
        for (lat, lon, want) in [
            (91.0, 8.0, "lat_deg"),
            (47.0, 181.0, "lon_deg"),
        ] {
            let yaml = format!(
                "{MINIMAL}origin:\n  lat_deg: {lat}\n  lon_deg: {lon}\n  alt_msl_m: 0.0\n"
            );
            let err = load("test", &yaml).unwrap_err();
            assert!(err.contains(want), "expected {want} in {err}");
        }
    }

    #[test]
    fn origin_missing_field_is_an_error() {
        let yaml = format!("{MINIMAL}origin:\n  lat_deg: 47.0\n  lon_deg: 8.0\n");
        assert!(load("test", &yaml).is_err());
    }

    #[test]
    fn loads_and_applies_tuning() {
        let cfg = config_from_str("test", MINIMAL).unwrap();
        assert_eq!(cfg.airframe.body.mass_kg, 0.6);
        assert_eq!(cfg.airframe.motors[1].spin_dir, SpinDir::Ccw);
        assert_eq!(cfg.mpc.pos_weight[0], 400.0);
    }

    #[test]
    fn missing_mass_is_an_error() {
        let broken = MINIMAL.replace("  mass_kg: 0.6\n", "");
        let err = config_from_str("test", &broken).unwrap_err();
        assert!(err.contains("mass_kg"), "{err}");
    }

    /// `gps_dual_antenna` is a `build:` knob precisely so this pairing is
    /// a build error. As a runtime param it was a boot warning on a defmt
    /// link nobody watches, and the vehicle flew with ANT2 wired, the
    /// baseline extrinsics pinned, and no heading ever fused.
    #[test]
    fn dual_antenna_without_unicore_is_an_error() {
        let yaml = MINIMAL.replace(
            "airframe:",
            "build:\n  gps_model: ublox\n  gps_dual_antenna: yes\nairframe:",
        );
        let err = load("test", &yaml).unwrap_err();
        assert!(err.contains("gps_dual_antenna"), "{err}");
        assert!(err.contains("unicore"), "{err}");
    }

    /// ...and the consistent pairing loads.
    #[test]
    fn dual_antenna_with_unicore_loads() {
        let yaml = MINIMAL.replace(
            "airframe:",
            "build:\n  gps_model: unicore\n  gps_dual_antenna: yes\nairframe:",
        );
        let v = load("test", &yaml).expect("unicore + dual antenna is legal");
        assert_eq!(
            v.build.as_ref().and_then(|b| b.gps_dual_antenna.as_deref()),
            Some("yes")
        );
    }

    /// A single-antenna UM982 is a real install (sakura_bench_hunter_outdoor),
    /// so the receiver model must not imply the antenna count.
    #[test]
    fn unicore_without_dual_antenna_loads() {
        let yaml = MINIMAL.replace("airframe:", "build:\n  gps_model: unicore\nairframe:");
        load("test", &yaml).expect("heading-capable receiver, one antenna fitted");
    }

    #[test]
    fn unknown_tuning_key_is_an_error() {
        let broken = MINIMAL.replace("mpc_w_pos_x", "mpc_w_pos_q");
        let err = config_from_str("test", &broken).unwrap_err();
        assert!(err.contains("unknown tuning key"), "{err}");
    }

    #[test]
    fn out_of_range_tuning_is_an_error() {
        let broken = MINIMAL.replace("mpc_w_pos_x: 400.0", "mpc_thrust_frac: 3.0");
        let err = config_from_str("test", &broken).unwrap_err();
        assert!(err.contains("out of range"), "{err}");
    }

    #[test]
    fn thrust_model_absent_is_none() {
        let v = load("test", MINIMAL).unwrap();
        assert_eq!(v.thrust_model, None);
        assert!(v.tuning_keys.contains("mpc_w_pos_x"));
    }

    #[test]
    fn thrust_model_table_parses() {
        let yaml = MINIMAL.replace(
            "  motors:",
            "  thrust_model: { type: table, table: a2rl_0114 }\n  motors:",
        );
        let v = load("test", &yaml).unwrap();
        assert_eq!(
            v.thrust_model,
            Some(ThrustModelYaml::Table {
                table: "a2rl_0114".into()
            })
        );
    }

    #[test]
    fn thrust_model_quadratic_parses_and_validates_k() {
        let yaml = MINIMAL.replace(
            "  motors:",
            "  thrust_model: { type: quadratic, k: 0.518 }\n  motors:",
        );
        let v = load("test", &yaml).unwrap();
        assert_eq!(v.thrust_model, Some(ThrustModelYaml::Quadratic { k: 0.518 }));

        let bad = MINIMAL.replace(
            "  motors:",
            "  thrust_model: { type: quadratic, k: 3.0 }\n  motors:",
        );
        let err = load("test", &bad).unwrap_err();
        assert!(err.contains("must be in [0, 1]"), "{err}");
    }

    #[test]
    fn thrust_model_missing_k_is_an_error() {
        let bad = MINIMAL.replace(
            "  motors:",
            "  thrust_model: { type: sqrt_squared }\n  motors:",
        );
        assert!(load("test", &bad).is_err());
    }

    #[test]
    fn thrust_model_unknown_type_is_an_error() {
        let bad = MINIMAL.replace(
            "  motors:",
            "  thrust_model: { type: cubic, k: 0.5 }\n  motors:",
        );
        assert!(load("test", &bad).is_err());
    }

    #[test]
    fn thrust_model_bad_table_stem_is_an_error() {
        let bad = MINIMAL.replace(
            "  motors:",
            "  thrust_model: { type: table, table: \"../evil\" }\n  motors:",
        );
        let err = load("test", &bad).unwrap_err();
        assert!(err.contains("CSV stem"), "{err}");
    }

    #[test]
    fn build_section_absent_is_none() {
        assert_eq!(load("test", MINIMAL).unwrap().build, None);
    }

    #[test]
    fn build_section_parses_and_validates() {
        let yaml = format!(
            "build:\n  board: sakurah743\n  rc_protocol: crsf\n  outer_loop: mpc\n  \
             pos_source: gps\n  gps_model: ublox\n  role: chaser\n{MINIMAL}"
        );
        let b = load("test", &yaml).unwrap().build.unwrap();
        assert_eq!(b.board.as_deref(), Some("sakurah743"));
        assert_eq!(b.role.as_deref(), Some("chaser"));
    }

    #[test]
    fn build_partial_declaration_is_ok() {
        let yaml = format!("build:\n  board: foxeerh743\n{MINIMAL}");
        let b = load("test", &yaml).unwrap().build.unwrap();
        assert_eq!(b.board.as_deref(), Some("foxeerh743"));
        assert_eq!(b.outer_loop, None);
    }

    #[test]
    fn build_bad_value_is_an_error() {
        let yaml = format!("build:\n  board: pixhawk\n{MINIMAL}");
        let err = load("test", &yaml).unwrap_err();
        assert!(err.contains("build.board"), "{err}");
    }

    #[test]
    fn build_unknown_knob_is_an_error() {
        let yaml = format!("build:\n  cpu: h743\n{MINIMAL}");
        assert!(load("test", &yaml).is_err());
    }

    #[test]
    fn default_mission_parses_and_defaults_none() {
        assert_eq!(load("test", MINIMAL).unwrap().default_mission, None);
        let yaml = format!("default_mission: outdoor_splits_slow\n{MINIMAL}");
        assert_eq!(
            load("test", &yaml).unwrap().default_mission.as_deref(),
            Some("outdoor_splits_slow")
        );
    }
}
