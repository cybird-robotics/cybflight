//! Host-side simulation harness for cybflight.
//!
//! Reuses the math in `cybflight-core` (FullQuadModel, cascade/geometric
//! controllers, MINCO/BFGS planner) and wires it into a MissionRunner-style
//! loop that can exercise hover, point-to-point, and multi-waypoint missions
//! without the Embassy runtime.
//!
//! The firmware's task wiring (channels, tickers, failsafe) is intentionally
//! not modelled here — those concerns belong in HIL. This crate tests the
//! control and planning math end-to-end.

pub mod controller;
pub mod plant;
pub mod report;
pub mod runner;
pub mod scenario;
pub mod trajectory;
pub mod viz;

pub use controller::{CascadeController, Controller};
pub use plant::{QuadPlant, VEHICLE};
pub use report::SimulationReport;
pub use runner::{MissionRunner, RunnerConfig, StepRecord};
pub use scenario::{PassCriteria, Scenario, Verdict};
pub use trajectory::{MissionSetpoints, Setpoint, SetpointSource};
