//! Scaffolding shared by the neural gate-racing tests.
//!
//! Both `nn_gate_race` and `acmpc_gate_race` fly the same course under the
//! same race rules and report the same numbers; only the policy and the
//! actuation interface differ. The rules and the bookkeeping live here so
//! the two results are comparable by construction rather than by two
//! transcriptions of the same termination conditions agreeing by luck.

use cybflight_core::nn::race_policy::{GateEvent, Track, VehicleState};
use nalgebra::Vector3;

/// Horizontal half-extent of the arena [m] (`bound_xy` upstream).
pub const BOUND_XY_M: f32 = 5.0;

/// Read a `<f4` fixture file as `f32`.
pub fn fixture(dir: &str, name: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{dir}/{name}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{name} is not a whole number of f32");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One flight's result: what the course did, plus the envelope the vehicle
/// reached getting there.
pub struct RaceOutcome {
    pub passes: u32,
    pub clips: u32,
    pub steps: usize,
    pub ended: &'static str,
    pub max_speed_m_s: f32,
    pub max_tilt_rad: f32,
    /// Position at every control period, for trajectory comparison.
    pub positions: Vec<Vector3<f32>>,
}

impl RaceOutcome {
    pub fn new(capacity: usize) -> Self {
        Self {
            passes: 0,
            clips: 0,
            steps: 0,
            ended: "completed",
            max_speed_m_s: 0.0,
            max_tilt_rad: 0.0,
            positions: Vec::with_capacity(capacity),
        }
    }

    /// Record the state the policy is about to be shown.
    pub fn sample(&mut self, state: &VehicleState) {
        self.positions.push(state.position_m);
        self.max_speed_m_s = self.max_speed_m_s.max(state.velocity_m_s.norm());
        let up = state.attitude * Vector3::z();
        self.max_tilt_rad = self.max_tilt_rad.max(up.z.clamp(-1.0, 1.0).acos());
    }

    /// Apply the gate state machine and the arena rules to one completed
    /// control period. Returns `false` once the flight has ended.
    pub fn advance(
        &mut self,
        track: &mut Track,
        prev: Vector3<f32>,
        new: Vector3<f32>,
    ) -> bool {
        self.steps += 1;
        match track.update_gate(prev, new) {
            GateEvent::Passed { .. } => self.passes += 1,
            GateEvent::Clipped { .. } => {
                self.clips += 1;
                self.ended = "gate collision";
                return false;
            }
            GateEvent::None => {}
        }
        if new.z < 0.0 {
            self.ended = "ground";
            return false;
        }
        if new.x.abs() > BOUND_XY_M || new.y.abs() > BOUND_XY_M {
            self.ended = "out of bounds";
            return false;
        }
        true
    }

    pub fn report(&self, label: &str, dt_s: f32, gates: usize) {
        println!(
            "{label}: {} gate passes ({} laps) in {} steps ({:.1} s), {} clips, ended '{}'",
            self.passes,
            self.passes / gates as u32,
            self.steps,
            self.steps as f32 * dt_s,
            self.clips,
            self.ended,
        );
        println!(
            "  peak speed {:.1} m/s, peak tilt {:.0}°",
            self.max_speed_m_s,
            self.max_tilt_rad.to_degrees(),
        );
    }
}
