#![cfg_attr(not(test), no_std)]
pub mod acmpc;
pub mod attitude_control;
pub mod blackbox_wire;
pub mod cbor;
pub mod eskf;
pub mod geodetic;
pub mod imu_stamp;
pub mod indi;
pub mod mahony;
pub mod mixer;
pub mod mpc;
pub mod nn;
// Let the `#[derive(Params)]` expansion's `::cybflight_core::…` paths
// resolve inside this crate too.
extern crate self as cybflight_core;

pub mod param_registry;
pub mod param_store;
pub mod params;
pub mod position_control;
pub mod rc;
pub mod rotation;
pub mod shell_complete;
pub mod trajectory_planning;
