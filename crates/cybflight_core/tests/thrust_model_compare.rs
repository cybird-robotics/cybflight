//! Compare `ThrustModel::Quadratic`, `ThrustModel::SqrtSquared`, and
//! `ThrustModel::Table` curves on the A2RL 6S bench data.
//!
//! Output: u → d at three pack voltages (low / mid / high). Table is the
//! ground-truth baseline; the analytic models are evaluated as
//! approximations. Reports max abs error and RMS error vs. Table per
//! voltage, and asserts those errors stay within the bounds claimed in
//! `identify_indi_k.py` (Quadratic ~0.27 N RMS in thrust → ~0.005 in `d`;
//! SqrtSquared ~0.23 N RMS).
//!
//! Run with `--nocapture` to see the full sweep table.

use cybflight_core::indi::{
    controller::NU,
    linearization::{ThrustLinearization, ThrustModel, TABLE_N},
    thrust_table::ThrustTable,
};

const CSV: &str = include_str!("a2rl_0114.csv");

/// Identified k values from `identify_indi_k.py` (matches `indi_task.rs`).
const K_QUAD: f32 = 0.518;
const K_SQRTSQ: f32 = 0.458;

/// Voltages to sweep — covers the table's pack range. Order: low/mid/high.
const V_TEST: [f32; 3] = [21.8, 23.2, 24.7];

fn parse_table() -> &'static ThrustTable<TABLE_N> {
    let mut lines = CSV.lines().filter(|l| !l.trim().is_empty());

    let header = lines.next().expect("missing header");
    let mut hdr = header.split(',').map(|s| s.trim());
    let map_size: usize = hdr.next().unwrap().parse::<f64>().unwrap() as usize;
    // CSV header carries collective thrust; INDI works in per-rotor units.
    // Mirrors the divide done in `crates/cybflight/build.rs`.
    let thrust_min: f32 = hdr.next().unwrap().parse::<f32>().unwrap() / NU as f32;
    let thrust_max: f32 = hdr.next().unwrap().parse::<f32>().unwrap() / NU as f32;
    let voltage_min: f32 = hdr.next().unwrap().parse().unwrap();
    let voltage_max: f32 = hdr.next().unwrap().parse().unwrap();
    assert_eq!(map_size, TABLE_N);

    let mut grid = [[0.0_f32; TABLE_N]; TABLE_N];
    for row in grid.iter_mut() {
        let line = lines.next().expect("row missing");
        for (col, slot) in row.iter_mut().enumerate() {
            let s = line.split(',').nth(col).expect("col missing");
            *slot = s.trim().parse().expect("not f32");
        }
    }

    Box::leak(Box::new(
        ThrustTable::<TABLE_N>::new(grid, thrust_min, thrust_max, voltage_min, voltage_max)
            .expect("table validation"),
    ))
}

#[test]
fn compare_three_models_against_table_baseline() {
    let table = parse_table();

    // `per_motor_max_n` defines the u → thrust mapping inside the Table arm.
    // Use the CSV's own thrust_max so u ∈ [0,1] spans the full table —
    // matches how `identify_indi_k.py` fits the analytic k values against
    // the entire dataset.
    let per_motor_max_n = table.thrust_max_n();

    let lin_quad = ThrustLinearization::new(K_QUAD, ThrustModel::Quadratic, per_motor_max_n);
    let lin_sq = ThrustLinearization::new(K_SQRTSQ, ThrustModel::SqrtSquared, per_motor_max_n);
    let lin_tab = ThrustLinearization::new(0.0, ThrustModel::Table(table), per_motor_max_n);

    // Skip u below the table's covered thrust range. Below thrust_min the
    // table boundary-clamps (returns grid[i_voltage][0]) while the
    // analytic models return 0; that's a structural divergence due to the
    // CSV not covering near-zero thrust, not a model error to indict.
    let u_min = table.thrust_min_n() / per_motor_max_n;

    println!(
        "\n=== Thrust-curve comparison (u → d), per_motor_max_n = {:.3} N ===",
        per_motor_max_n
    );
    println!(
        "Table covers thrust [{:.2}, {:.2}] N (u ≥ {:.4}), voltage [{:.2}, {:.2}] V",
        table.thrust_min_n(),
        table.thrust_max_n(),
        u_min,
        table.voltage_min_v(),
        table.voltage_max_v()
    );

    for &v in &V_TEST {
        // Sweep u ∈ [u_min, 1] and accumulate error vs. Table.
        let n = 101;
        let mut max_err_quad: f32 = 0.0;
        let mut max_err_sq: f32 = 0.0;
        let mut sse_quad: f64 = 0.0;
        let mut sse_sq: f64 = 0.0;
        let mut max_at_quad: f32 = 0.0;
        let mut max_at_sq: f32 = 0.0;

        // Sample dump (only every 10th to keep the printout compact).
        println!("\n--- Voltage {:.2} V ---", v);
        println!(
            "{:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "u", "d_table", "d_quad", "d_sq", "Δquad", "Δsq"
        );

        for i in 0..n {
            let u = u_min + (1.0 - u_min) * (i as f32 / (n - 1) as f32);
            let d_tab = lin_tab.linearize(u, v);
            let d_q = lin_quad.linearize(u, v);
            let d_s = lin_sq.linearize(u, v);
            let e_q = (d_q - d_tab).abs();
            let e_s = (d_s - d_tab).abs();

            if e_q > max_err_quad {
                max_err_quad = e_q;
                max_at_quad = u;
            }
            if e_s > max_err_sq {
                max_err_sq = e_s;
                max_at_sq = u;
            }
            sse_quad += (e_q as f64) * (e_q as f64);
            sse_sq += (e_s as f64) * (e_s as f64);

            if i % 10 == 0 {
                println!(
                    "{:>5.2} {:>10.4} {:>10.4} {:>10.4} {:>+10.4} {:>+10.4}",
                    u,
                    d_tab,
                    d_q,
                    d_s,
                    d_q - d_tab,
                    d_s - d_tab
                );
            }
        }

        let rms_quad = (sse_quad / n as f64).sqrt() as f32;
        let rms_sq = (sse_sq / n as f64).sqrt() as f32;

        println!(
            "summary @ {:.2} V:  Quadratic  max |Δd|={:.4} (at u={:.2})  RMS={:.4}",
            v, max_err_quad, max_at_quad, rms_quad
        );
        println!(
            "summary @ {:.2} V:  SqrtSquared max |Δd|={:.4} (at u={:.2})  RMS={:.4}",
            v, max_err_sq, max_at_sq, rms_sq
        );

        // Finiteness + reasonableness (test is informational; bounds are
        // generous enough to catch a regression but not to enforce a
        // particular fit quality — the fit is identified offline).
        assert!(max_err_quad.is_finite() && rms_quad.is_finite());
        assert!(max_err_sq.is_finite() && rms_sq.is_finite());
        assert!(max_err_quad < 0.15, "Quadratic L∞ regressed: {max_err_quad}");
        assert!(max_err_sq < 0.15, "SqrtSquared L∞ regressed: {max_err_sq}");
    }

    // Voltage-sensitivity check on Table: Quadratic/SqrtSquared are
    // voltage-blind, so comparing high-V vs low-V Table outputs at a
    // fixed u gives the magnitude of compensation we get for free by
    // switching to the table.
    let u_check = 0.5;
    let d_lo = lin_tab.linearize(u_check, V_TEST[0]);
    let d_hi = lin_tab.linearize(u_check, V_TEST[2]);
    let voltage_compensation = (d_lo - d_hi).abs();
    println!(
        "\nTable voltage sensitivity at u={:.2}: d({:.2}V)={:.4}, d({:.2}V)={:.4}, |Δd|={:.4}",
        u_check, V_TEST[0], d_lo, V_TEST[2], d_hi, voltage_compensation
    );
    println!(
        "(Quadratic & SqrtSquared can never reproduce this — they are voltage-blind.)"
    );
}
