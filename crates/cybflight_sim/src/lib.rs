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

/// Diagnostic baseline controllers. The geometric attitude controller was
/// restored to `cybflight_core::attitude_control` (the firmware's
/// `outer_geometric` path is a maintained alternative to MPC+INDI, and both
/// crates share one copy); re-export it under the old path.
pub mod baselines {
    pub use cybflight_core::attitude_control::*;
}
pub mod controller;
pub mod plant;
pub mod primitives;
pub mod report;
pub mod rl_env;
pub mod rl_reference;
pub mod runner;
pub mod scenario;
pub mod sensors;
pub mod trajectory;
#[cfg(feature = "viz")]
pub mod viz;

pub use controller::{
    CascadeController, Controller, MpcDirectController, MpcFullIndiController,
    MpcIndiController, ThrustCommandMap,
};
pub use plant::{PlantParams, QuadPlant, NX_PLANT, VEHICLE};
pub use report::SimulationReport;
pub use runner::{MissionRunner, RunnerConfig, StepRecord};
pub use scenario::{
    default_sim_params, default_vehicle, tweaked_vehicle, PassCriteria, Scenario, Verdict,
};
pub use sensors::{
    FaultedGps, GpsMeasurement, GpsModel, ImuMeasurement, ImuModel, NoRotorTelemetry, NoisyGps,
    NoisyImu, NoisyRotorTelemetry, OutageGps, PerfectGps, PerfectImu, PerfectRotorTelemetry,
    RotorModel, RotorTelemetry,
};
pub use trajectory::{MissionSetpoints, Setpoint, SetpointSource};
