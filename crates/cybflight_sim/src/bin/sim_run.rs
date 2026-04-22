//! CLI entry point: run a named scenario, emit artifacts, optionally stream
//! to rerun.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use cybflight_sim::{
    controller::{CascadeController, Controller, MpcDirectController, MpcIndiController},
    plant::QuadPlant,
    report,
    runner::MissionRunner,
    scenario::{Scenario, Verdict},
    viz::RerunLogger,
};
use nalgebra::Vector3;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ControllerKind {
    /// 10-state MPC at 100 Hz + INDI at 8 kHz (firmware-match topology).
    MpcIndi,
    /// 13-state MPC at 100 Hz, per-motor output (diagnostic upper bound).
    MpcDirect,
    /// PD position + geometric attitude + rate-P + mixer (legacy baseline).
    Cascade,
}

#[derive(Parser, Debug)]
#[command(name = "sim-run", about = "cybflight simulation runner")]
struct Args {
    /// Scenario to run.
    #[arg(long, default_value = "mission_square")]
    scenario: String,

    /// Controller: mpc-indi (default, firmware-match), mpc-direct, or cascade.
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
}

fn main() -> ExitCode {
    let args = Args::parse();

    let mut scenario = build_scenario(&args.scenario);
    let mut plant = QuadPlant::new(scenario.vehicle_params.clone(), 0.002);
    let mut controller: Box<dyn Controller> = match args.controller {
        ControllerKind::MpcIndi => {
            Box::new(MpcIndiController::from_params(&scenario.vehicle_params))
        }
        ControllerKind::MpcDirect => {
            Box::new(MpcDirectController::from_params(&scenario.vehicle_params))
        }
        ControllerKind::Cascade => {
            Box::new(CascadeController::from_params(&scenario.vehicle_params))
        }
    };
    let controller_name = controller.name();

    let runner = MissionRunner::new(Default::default());
    let out = runner.run(&mut scenario, &mut plant, &mut *controller);

    let dir = args.out_dir.join(&scenario.name);
    let json = report::write_json(&dir, &scenario, controller_name, &out).expect("write json");
    println!("wrote {}", json.display());
    if args.markdown {
        let md = report::write_markdown(&dir, &scenario, controller_name, &out)
            .expect("write md");
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
        "{:?} [{}]: verdict={:?} rms_err={:.3}m terminal_err={:.3}m peak_tilt={:.1}°",
        scenario.name,
        controller_name,
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
