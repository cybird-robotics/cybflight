//! Rerun visualization integration.
//!
//! Mirrors the layout used by
//! `/home/hs293go/src/autopilot/examples/src/cascade_controller_sim.cpp` so
//! recordings look familiar when flipping between the C++ and Rust sims:
//! a 3D scene entity per transform, motor thrust scalars, and position
//! error scalars. Keep this module dependency-free outside `rerun` itself —
//! it should be safe to call or skip without affecting runner behavior.

use rerun::{RecordingStream, RecordingStreamBuilder};

use crate::runner::StepRecord;

pub struct RerunLogger {
    rec: RecordingStream,
}

impl RerunLogger {
    /// Spawn a new recording and connect to the viewer if available.
    ///
    /// `app_id` controls the stream name shown in the viewer. Swallows the
    /// viewer-spawn error — visualization is strictly optional, so a missing
    /// `rerun` binary should never fail a simulation.
    pub fn spawn(app_id: &str) -> Option<Self> {
        let rec = RecordingStreamBuilder::new(app_id.to_string())
            .spawn()
            .ok()?;
        Some(Self { rec })
    }

    pub fn log_scenario(&self, name: &str) {
        let _ = self.rec.log_static(
            "scenario",
            &rerun::TextDocument::new(name.to_string()),
        );
    }

    /// Log a history vector. Time axis = simulation time [s].
    ///
    /// Emits two static polylines — the planned reference (yellow) and the
    /// actual executed path (cyan) — plus per-frame transform, scalars, and
    /// motor thrusts. The polylines are static so they stay visible across
    /// the whole timeline while scrubbing.
    pub fn log_history(&self, history: &[StepRecord]) {
        let planned: Vec<[f32; 3]> = history
            .iter()
            .map(|r| [r.setpoint.position.x, r.setpoint.position.y, r.setpoint.position.z])
            .collect();
        let actual: Vec<[f32; 3]> = history
            .iter()
            .map(|r| [r.position.x, r.position.y, r.position.z])
            .collect();
        if planned.len() >= 2 {
            let _ = self.rec.log_static(
                "world/trajectory/planned",
                &rerun::LineStrips3D::new([planned])
                    .with_colors([rerun::Color::from_rgb(255, 200, 0)])
                    .with_radii([0.01_f32]),
            );
        }
        if actual.len() >= 2 {
            let _ = self.rec.log_static(
                "world/trajectory/actual",
                &rerun::LineStrips3D::new([actual])
                    .with_colors([rerun::Color::from_rgb(0, 220, 220)])
                    .with_radii([0.01_f32]),
            );
        }

        for rec in history {
            self.rec.set_duration_secs("sim_time", rec.t as f64);

            let p = rec.position;
            let sp = rec.setpoint.position;
            let q = rec.attitude.into_inner();

            let _ = self.rec.log(
                "world/quadrotor",
                &rerun::Transform3D::from_translation_rotation(
                    [p.x, p.y, p.z],
                    rerun::Quaternion::from_xyzw([q.i, q.j, q.k, q.w]),
                ),
            );
            let _ = self.rec.log(
                "world/setpoint",
                &rerun::Points3D::new([[sp.x, sp.y, sp.z]])
                    .with_colors([rerun::Color::from_rgb(255, 200, 0)])
                    .with_radii([0.05_f32]),
            );

            let err = (p - sp).norm();
            let _ = self
                .rec
                .log("metrics/pos_err", &rerun::Scalars::new([err as f64]));
            let _ = self.rec.log(
                "metrics/tilt_deg",
                &rerun::Scalars::new([rec.tilt_rad.to_degrees() as f64]),
            );

            for (i, f) in rec.motor_forces.iter().enumerate() {
                let _ = self.rec.log(
                    format!("motors/m{i}"),
                    &rerun::Scalars::new([*f as f64]),
                );
            }
        }
    }
}
