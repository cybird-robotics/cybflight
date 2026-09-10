//! CLI entry point: run a named scenario, emit artifacts, optionally stream
//! to rerun.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use cybflight_sim::{
    controller::{
        CascadeController, Controller, GeometricIndiController, MpcDirectController,
        MpcFullIndiController, MpcIndiController, TinyMpcIndiController,
    },
    plant::QuadPlant,
    report,
    runner::MissionRunner,
    scenario::{Scenario, Verdict},
    sensors::{GpsModel, ImuModel, NoisyGps, NoisyImu},
    viz::RerunLogger,
};
use nalgebra::Vector3;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ControllerKind {
    /// 10-state MPC at 100 Hz + INDI at 8 kHz (firmware-match topology).
    MpcIndi,
    /// TinyMPC (ADMM, hover-linearised model) at 100 Hz + INDI at 8 kHz.
    TinympcIndi,
    /// Geometric tracking controller (RPG position controller port) at
    /// 500 Hz + INDI at 8 kHz.
    GeometricIndi,
    /// 13-state MPC at 100 Hz → (T_d, α_d) → INDI α loop at 8 kHz
    /// (`outer_loop: mpc_full` prototype, Sun et al. T-RO 2022 Fig. 3).
    MpcFullIndi,
    /// Same, but the inner loop degrades to static inversion (no INDI
    /// increments) — the paper's "NMPC w/o INDI" ablation.
    MpcFullNoindi,
    /// 13-state MPC at 100 Hz, per-motor output (diagnostic upper bound).
    MpcDirect,
    /// PD position + geometric attitude + rate-P + mixer (legacy baseline).
    Cascade,
}

/// IMU-noise presets. Only `MpcIndi` actually consumes the IMU; ground-
/// truth controllers (cascade, mpc_direct) read perfect state and ignore
/// the noise setting. CLI accepts the flag for any controller so sweeps
/// stay uniform.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum NoisePreset {
    /// Zero noise — `PerfectImu` (default).
    None,
    /// Consumer-grade MEMS IMU: σ_gyro=0.03 rad/s, σ_accel=0.3 m/s².
    Mems,
    /// Aggressive: σ_gyro=0.1 rad/s, σ_accel=1.0 m/s² — stresses INDI.
    Aggressive,
}

impl NoisePreset {
    fn build_imu(self, seed: u64) -> Option<Box<dyn ImuModel>> {
        match self {
            Self::None => None,
            Self::Mems => Some(Box::new(NoisyImu::isotropic(seed, 0.03, 0.3))),
            Self::Aggressive => Some(Box::new(NoisyImu::isotropic(seed, 0.1, 1.0))),
        }
    }
}

/// GPS presets. Attaching any non-`None` preset activates the in-sim ESKF
/// (runner feeds GPS updates + IMU predict), so controllers see estimator
/// output instead of plant truth — mirroring firmware `est_pos_gps`.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum GpsPreset {
    /// No GPS — runner uses plant ground truth (default).
    None,
    /// Open-sky u-blox M10 with SBAS: 5 Hz, σ_pos=0.5 m, σ_vel=0.2 m/s.
    Sbas,
    /// Degraded GPS: 5 Hz, σ_pos=2.0 m, σ_vel=0.5 m/s.
    Degraded,
}

impl GpsPreset {
    fn build_gps(self, seed: u64) -> Option<Box<dyn GpsModel>> {
        match self {
            Self::None => None,
            Self::Sbas => Some(Box::new(NoisyGps::isotropic(seed, 5.0, 0.5, 0.2))),
            Self::Degraded => Some(Box::new(NoisyGps::isotropic(seed, 5.0, 2.0, 0.5))),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "sim-run", about = "cybflight simulation runner")]
struct Args {
    /// Scenario to run.
    #[arg(long, default_value = "mission_square")]
    scenario: String,

    /// Controller: mpc-indi (default, firmware-match), mpc-full-indi
    /// (full-model NMPC + INDI α inner loop), mpc-full-noindi (its
    /// static-inversion ablation), mpc-direct, or cascade.
    #[arg(long, value_enum, default_value_t = ControllerKind::MpcIndi)]
    controller: ControllerKind,

    /// Output directory for JSON/MD/CSV artifacts.
    #[arg(long, default_value = "target/sim-out")]
    out_dir: PathBuf,

    /// Also emit timeseries.csv.
    #[arg(long)]
    csv: bool,

    /// Also emit report.md.
    #[arg(long)]
    markdown: bool,

    /// Stream to a running rerun viewer (spawns one if available).
    #[arg(long)]
    viz: bool,

    /// IMU noise model. Applied as an overlay on the scenario's default
    /// `PerfectImu`. Only affects controllers that consume the IMU (MpcIndi);
    /// ground-truth controllers silently ignore it.
    #[arg(long, value_enum, default_value_t = NoisePreset::None)]
    noise: NoisePreset,

    /// Seed for the noise PRNG. Deterministic across runs.
    #[arg(long, default_value_t = 0xC0FFEE)]
    noise_seed: u64,

    /// GPS model. Non-none activates the in-sim ESKF, so the controller
    /// sees estimator output instead of plant truth (matches firmware
    /// `est_pos_gps` topology).
    #[arg(long, value_enum, default_value_t = GpsPreset::None)]
    gps: GpsPreset,

    /// Seed for the GPS noise PRNG. Deterministic across runs.
    #[arg(long, default_value_t = 0xDEADBEEF)]
    gps_seed: u64,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let mut scenario = build_scenario(&args.scenario);
    if let Some(imu) = args.noise.build_imu(args.noise_seed) {
        scenario = scenario.with_imu(imu);
    }
    if let Some(gps) = args.gps.build_gps(args.gps_seed) {
        scenario = scenario.with_gps(gps);
    }
    // Plant dt must match the runner's dt_sim (default 1/8000) so the
    // runner's tick accounting and the plant's integration clock stay in
    // lockstep. Matches what autotest_mission and autotest_noisy use.
    let mut plant = QuadPlant::new(scenario.vehicle_params.clone(), &scenario.sim_params, 1.0 / 8000.0);
    let mut controller: Box<dyn Controller> = match args.controller {
        ControllerKind::MpcIndi => {
            Box::new(MpcIndiController::from_params(&scenario.vehicle_params))
        }
        ControllerKind::TinympcIndi => {
            Box::new(TinyMpcIndiController::from_params(&scenario.vehicle_params))
        }
        ControllerKind::GeometricIndi => {
            Box::new(GeometricIndiController::from_params(&scenario.vehicle_params))
        }
        ControllerKind::MpcFullIndi => {
            Box::new(MpcFullIndiController::from_params(&scenario.vehicle_params))
        }
        ControllerKind::MpcFullNoindi => Box::new(MpcFullIndiController::with_options(
            &scenario.vehicle_params,
            100.0,
            false,
        )),
        ControllerKind::MpcDirect => {
            Box::new(MpcDirectController::from_params(&scenario.vehicle_params, &scenario.sim_params))
        }
        ControllerKind::Cascade => {
            Box::new(CascadeController::from_params(&scenario.vehicle_params, &scenario.sim_params))
        }
    };
    let controller_name = controller.name();

    let runner = MissionRunner::new(Default::default());
    let out = runner.run(&mut scenario, &mut plant, &mut *controller);

    let dir = args.out_dir.join(&scenario.name);
    let json = report::write_json(&dir, &scenario, controller_name, &out).expect("write json");
    println!("wrote {}", json.display());
    if args.markdown {
        let md = report::write_markdown(&dir, &scenario, controller_name, &out).expect("write md");
        println!("wrote {}", md.display());
    }
    if args.csv {
        let csv = report::write_csv(&dir, &out.history).expect("write csv");
        println!("wrote {}", csv.display());
    }
    if args.viz {
        if let Some(logger) = RerunLogger::spawn(&format!("cybflight-sim:{}", scenario.name)) {
            logger.log_scenario(&scenario.name);
            logger.log_history(&out.history);
        } else {
            eprintln!("warning: could not spawn rerun viewer — skipping visualization");
        }
    }

    println!(
        "{:?} [{} noise={:?}]: verdict={:?} rms_err={:.3}m terminal_err={:.3}m peak_tilt={:.1}°",
        scenario.name,
        controller_name,
        args.noise,
        out.verdict,
        out.summary.rms_pos_err_m,
        out.summary.terminal_pos_err_m,
        out.summary.peak_tilt_rad.to_degrees()
    );
    for reason in &out.failure_reasons {
        eprintln!("  fail: {reason}");
    }
    match out.verdict {
        Verdict::Pass => ExitCode::SUCCESS,
        Verdict::Fail => ExitCode::from(1),
    }
}

fn build_scenario(name: &str) -> Scenario {
    match name {
        "hover_level" => Scenario::hover("hover_level", Vector3::new(0.0, 0.0, 1.0), 0.0),
        "hover_tilt30" => Scenario::hover(
            "hover_tilt30",
            Vector3::new(0.0, 0.0, 1.0),
            30.0_f32.to_radians(),
        ),
        "p2p_x3" => Scenario::point_to_point(
            "p2p_x3",
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(3.0, 0.0, 1.0),
        ),
        "mission_square" => Scenario::mission(
            "mission_square",
            Vector3::new(0.0, 0.0, 1.0),
            &[
                Vector3::new(3.0, 0.0, 1.0),
                Vector3::new(3.0, 3.0, 1.0),
                Vector3::new(0.0, 3.0, 1.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
        ),
        other => panic!("unknown scenario: {other}"),
    }
}
