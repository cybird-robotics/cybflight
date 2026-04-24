//! End-to-end regression oracle for `trajectory_planning`.
//!
//! Compares the planner's current output against a binary fixture
//! (`tests/fixtures/trajectory_oracle.bin`) generated from the
//! pre-refactor implementation. The fixture freezes a small set of
//! representative scenarios as f32 little-endian bytes so the test
//! survives independently of the in-process oracle that was retired
//! with the legacy_reference module.
//!
//! ## Tolerance
//! `EPS_REL = 1e-4` (with `EPS_ABS = 1e-4` floor) on planner end-to-end:
//! BFGS amplifies any per-call FP drift across hundreds of iterations,
//! so anything tighter is too sensitive to LLVM codegen variation.
//!
//! ## Regenerate
//! ```sh
//! REGEN_TRAJECTORY_ORACLE=1 cargo test -p cybflight-core \
//!     --target $(rustc -vV | sed -n 's/^host: //p') --release \
//!     --test trajectory_oracle regenerate_fixture -- --ignored
//! ```

use std::env;
use std::fs;
use std::path::PathBuf;

use cybflight_core::trajectory_planning::planner::{plan, PlannerInput, PlannerResult};
use cybflight_core::trajectory_planning::quad_planning_config::QuadPlanningConfig;

const EPS_REL: f32 = 1e-4;
const EPS_ABS: f32 = 1e-4;

#[track_caller]
fn cmp_rel(label: &str, a: f32, b: f32) {
    if a.is_nan() && b.is_nan() {
        return;
    }
    let diff = (a - b).abs();
    let allowed = EPS_ABS.max(a.abs().max(b.abs()) * EPS_REL);
    assert!(
        diff <= allowed,
        "{label}: |{a} - {b}| = {diff} > {allowed} (eps_rel={EPS_REL}, eps_abs={EPS_ABS})"
    );
}

struct Scenario {
    name: &'static str,
    start_pos: [f32; 3],
    start_vel: [f32; 3],
    targets: Vec<[f32; 3]>,
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "goto_short",
            start_pos: [0.0, 0.0, 1.0],
            start_vel: [0.0, 0.0, 0.0],
            targets: vec![[2.0, 0.0, 1.0]],
        },
        Scenario {
            name: "goto_diagonal",
            start_pos: [0.0, 0.0, 1.0],
            start_vel: [0.5, 0.0, 0.0],
            targets: vec![[3.0, 2.0, 1.5]],
        },
        Scenario {
            name: "two_waypoints",
            start_pos: [0.0, 0.0, 1.0],
            start_vel: [0.0; 3],
            targets: vec![[2.0, 1.0, 1.0], [4.0, 0.0, 1.5]],
        },
        Scenario {
            name: "three_waypoints",
            start_pos: [0.0, 0.0, 1.0],
            start_vel: [0.0; 3],
            targets: vec![[2.0, 0.0, 1.0], [4.0, 2.0, 1.0], [6.0, 0.0, 1.0]],
        },
        Scenario {
            name: "ascend_zigzag",
            start_pos: [0.0, 0.0, 0.5],
            start_vel: [0.0; 3],
            targets: vec![[1.5, 1.0, 1.5], [3.0, -1.0, 2.5], [4.5, 0.0, 3.0]],
        },
    ]
}

fn run_scenario(sc: &Scenario, config: &QuadPlanningConfig) -> PlannerResult {
    let targets: Vec<_> = sc.targets.iter().map(|t| (*t).into()).collect();
    let input = PlannerInput::waypoints(sc.start_pos, sc.start_vel, &targets);
    plan(&input, config)
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("trajectory_oracle.bin")
}

fn write_scenario(buf: &mut Vec<u8>, res: &PlannerResult) {
    buf.extend_from_slice(&(res.num_pieces as u32).to_le_bytes());
    buf.extend_from_slice(&res.final_cost.to_le_bytes());
    let dur = res.trajectory.total_duration();
    for s in 0..=24 {
        let t = dur * s as f32 / 24.0;
        let p = res.trajectory.get_pos(t);
        for d in 0..3 {
            buf.extend_from_slice(&p[d].to_le_bytes());
        }
    }
}

fn read_scenario(bytes: &[u8], cursor: &mut usize) -> (usize, f32, [[f32; 3]; 25]) {
    let np = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap()) as usize;
    *cursor += 4;
    let cost = f32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    let mut samples = [[0.0f32; 3]; 25];
    for s in 0..25 {
        for d in 0..3 {
            samples[s][d] = f32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
            *cursor += 4;
        }
    }
    (np, cost, samples)
}

#[test]
#[ignore]
fn regenerate_fixture() {
    if env::var("REGEN_TRAJECTORY_ORACLE").is_err() {
        eprintln!("Skipping fixture regen — set REGEN_TRAJECTORY_ORACLE=1 to enable.");
        return;
    }
    let config = QuadPlanningConfig::default();
    let mut buf = Vec::new();
    let scs = scenarios();
    buf.extend_from_slice(&(scs.len() as u32).to_le_bytes());
    for sc in &scs {
        let res = run_scenario(sc, &config);
        write_scenario(&mut buf, &res);
    }
    let path = fixture_path();
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, &buf).unwrap();
    eprintln!("wrote fixture: {} ({} bytes)", path.display(), buf.len());
}

#[test]
fn planner_end_to_end_matches_fixture() {
    let path = fixture_path();
    if !path.exists() {
        eprintln!(
            "Fixture {} not present — generate with REGEN_TRAJECTORY_ORACLE=1 \
             cargo test ... regenerate_fixture -- --ignored",
            path.display()
        );
        return;
    }
    let bytes = fs::read(&path).expect("read fixture");
    let mut cursor = 0;
    let n_scenarios =
        u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
    cursor += 4;

    let config = QuadPlanningConfig::default();
    let scs = scenarios();
    assert_eq!(
        n_scenarios,
        scs.len(),
        "fixture has {n_scenarios} scenarios; code has {}",
        scs.len()
    );

    for sc in &scs {
        let (np_fix, cost_fix, samples_fix) = read_scenario(&bytes, &mut cursor);
        let res = run_scenario(sc, &config);
        assert_eq!(res.num_pieces, np_fix, "{}: num_pieces", sc.name);
        cmp_rel(&format!("{}: fixture cost", sc.name), res.final_cost, cost_fix);
        let dur = res.trajectory.total_duration();
        for s in 0..=24 {
            let t = dur * s as f32 / 24.0;
            let p = res.trajectory.get_pos(t);
            for d in 0..3 {
                cmp_rel(
                    &format!("{}: fixture pos@{t}[{d}]", sc.name),
                    p[d],
                    samples_fix[s][d],
                );
            }
        }
    }
}
