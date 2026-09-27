//! Simulation report artifacts.
//!
//! Report output:
//!  - `report.json` is the primary, always-written artifact.
//!  - `report.md` and `timeseries.csv` are minimal stubs, enabled for
//!    future-proofing when an agent workflow wants either format.
//!
//! JSON is kept small (summary + verdict + criteria, no timeseries) so it
//! can be pasted whole into an agent context window.

use std::fs;
use std::io::Write;
use std::path::Path;

use crate::runner::{RunOutput, StepRecord, SummaryMetrics};
use crate::scenario::{PassCriteria, Scenario, Verdict};

#[derive(serde::Serialize)]
pub struct SimulationReport<'a> {
    pub scenario: &'a str,
    pub controller: &'a str,
    pub verdict: Verdict,
    pub failure_reasons: &'a [String],
    pub summary: &'a SummaryMetrics,
    pub pass_criteria: SerializableCriteria,
}

#[derive(serde::Serialize)]
pub struct SerializableCriteria {
    pub terminal_pos_err_m: f32,
    pub rms_pos_err_m: f32,
    pub peak_tilt_rad: f32,
    pub geofence_min: [f32; 3],
    pub geofence_max: [f32; 3],
}

impl From<&PassCriteria> for SerializableCriteria {
    fn from(c: &PassCriteria) -> Self {
        Self {
            terminal_pos_err_m: c.terminal_pos_err_m,
            rms_pos_err_m: c.rms_pos_err_m,
            peak_tilt_rad: c.peak_tilt_rad,
            geofence_min: [c.geofence_min.x, c.geofence_min.y, c.geofence_min.z],
            geofence_max: [c.geofence_max.x, c.geofence_max.y, c.geofence_max.z],
        }
    }
}

/// Write `report.json` to `dir`. Returns the full path written.
pub fn write_json(
    dir: &Path,
    scenario: &Scenario,
    controller_name: &str,
    out: &RunOutput,
) -> std::io::Result<std::path::PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join("report.json");
    let report = SimulationReport {
        scenario: &scenario.name,
        controller: controller_name,
        verdict: out.verdict,
        failure_reasons: &out.failure_reasons,
        summary: &out.summary,
        pass_criteria: (&scenario.pass_criteria).into(),
    };
    let json = serde_json::to_string_pretty(&report).expect("report serialization");
    fs::write(&path, json)?;
    Ok(path)
}

/// Write `report.md` — minimal one-screen summary. Future-proofing only.
pub fn write_markdown(
    dir: &Path,
    scenario: &Scenario,
    controller_name: &str,
    out: &RunOutput,
) -> std::io::Result<std::path::PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join("report.md");
    let s = &out.summary;
    let verdict = match out.verdict {
        Verdict::Pass => "PASS",
        Verdict::Fail => "FAIL",
    };
    let mut buf = String::new();
    buf.push_str(&format!("# {} — {}\n\n", scenario.name, verdict));
    buf.push_str(&format!("controller: `{}`\n\n", controller_name));
    buf.push_str("| metric | value |\n|---|---|\n");
    buf.push_str(&format!("| rms_pos_err_m | {:.3} |\n", s.rms_pos_err_m));
    buf.push_str(&format!(
        "| terminal_pos_err_m | {:.3} |\n",
        s.terminal_pos_err_m
    ));
    buf.push_str(&format!(
        "| peak_tilt_deg | {:.1} |\n",
        s.peak_tilt_rad.to_degrees()
    ));
    buf.push_str(&format!("| peak_pos_err_m | {:.3} |\n", s.peak_pos_err_m));
    buf.push_str(&format!(
        "| peak_motor_sat_pct | {:.1} |\n",
        s.peak_motor_saturation_pct
    ));
    buf.push_str(&format!(
        "| total_sim_time_s | {:.2} |\n",
        s.total_sim_time_s
    ));
    if !out.failure_reasons.is_empty() {
        buf.push_str("\n**failures:**\n");
        for r in &out.failure_reasons {
            buf.push_str(&format!("- {r}\n"));
        }
    }
    fs::write(&path, buf)?;
    Ok(path)
}

/// Write `timeseries.csv`. Minimal schema: t, position, setpoint, tilt,
/// motor forces. Opt-in via the CLI / tests — large.
pub fn write_csv(
    dir: &Path,
    history: &[StepRecord],
) -> std::io::Result<std::path::PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join("timeseries.csv");
    let mut f = fs::File::create(&path)?;
    writeln!(
        f,
        "t,px,py,pz,vx,vy,vz,tilt_deg,sp_px,sp_py,sp_pz,u0,u1,u2,u3"
    )?;
    for r in history {
        writeln!(
            f,
            "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.4},{:.4},{:.4},{:.3},{:.3},{:.3},{:.3}",
            r.t,
            r.position.x,
            r.position.y,
            r.position.z,
            r.velocity.x,
            r.velocity.y,
            r.velocity.z,
            r.tilt_rad.to_degrees(),
            r.setpoint.position.x,
            r.setpoint.position.y,
            r.setpoint.position.z,
            r.motor_forces[0],
            r.motor_forces[1],
            r.motor_forces[2],
            r.motor_forces[3],
        )?;
    }
    Ok(path)
}
