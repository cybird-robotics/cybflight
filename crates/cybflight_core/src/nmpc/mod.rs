// NMPC module — public API and test scenario.
// Declared from usb_serial.rs via:  #[path = "nmpc/mod.rs"] mod nmpc;

mod lbfgs;
mod model;
mod solver;

use model::{Control, State};
use solver::N;
pub use solver::NmpcSolver;

/// Result of one NMPC solve.
pub struct NmpcResult {
    /// Optimal first control: [thrust (N), omega_x, omega_y, omega_z] (rad/s)
    pub u_opt: [f32; 4],
    /// Vehicle position at the current time step [x, y, z] (m)
    pub pos: [f32; 3],
    pub iterations: i32,
    pub converged: bool,
}

/// Run one NMPC solve with a simulated hovering drone scenario.
///
/// Initial state: 0.5 m above the hover target (z=1.5 m), slightly tilted.
/// Reference:     hover at z=1.0 m, level attitude, zero velocity.
/// u_ref:         [mass*g, 0, 0, 0]  — hover thrust.
///
/// This non-trivial offset exercises the full L-BFGS loop so convergence
/// and solve time are meaningful. After the first call the solver warmstarts
/// from the previous solution.
pub fn run_once(solver: &mut NmpcSolver) -> NmpcResult {
    // Reference: 1 m altitude, level (identity quaternion), stationary
    let hover_ref = State::from([
        0.0, 0.0, 1.0, // position target
        0.0, 0.0, 0.0, 1.0, // identity quaternion
        0.0, 0.0, 0.0, // zero velocity
    ]);
    // Initial state (first call only): offset from hover target
    let x0_init = State::from([
        0.1, -0.2, 0.0, // position offset
        0.1494381, 0.0, 0.0, 0.9887711, // level attitude
        -0.01, 0.02, 0.0, // zero velocity
    ]);
    // Use propagated state from previous solve, or start from x0_init
    let x0 = solver.sim_x0.unwrap_or(x0_init);

    let mass = solver.model.mass;
    let grav = solver.model.grav;
    let u_ref = Control::from([mass * grav, 0.0, 0.0, 0.0]);

    // Proximity check: if already close to hover, skip the solve.
    // Thresholds: 5 cm position, 5 cm/s velocity, attitude vector part < 0.05 rad.
    let dx = x0[0] - hover_ref[0];
    let dy = x0[1] - hover_ref[1];
    let dz = x0[2] - hover_ref[2];
    let pos_err2 = dx * dx + dy * dy + dz * dz;
    let vel_err2 = x0[7] * x0[7] + x0[8] * x0[8] + x0[9] * x0[9];
    let att_err2 = x0[3] * x0[3] + x0[4] * x0[4] + x0[5] * x0[5]; // |q_vec|^2
    if pos_err2 < 0.0001 && vel_err2 < 0.0025 && att_err2 < 0.0009 {
        // Close enough — propagate with hover thrust and return without solving.
        solver.sim_x0 = Some(solver.model.propagate_rk4(&x0, &u_ref));
        return NmpcResult {
            u_opt: [u_ref[0], u_ref[1], u_ref[2], u_ref[3]],
            pos: [x0[0], x0[1], x0[2]],
            iterations: 0,
            converged: true,
        };
    }

    let mut x_refs = [State::zeros(); N + 1];
    let mut u_refs = [Control::zeros(); N];
    for i in 0..=N {
        x_refs[i] = hover_ref;
    }
    for i in 0..N {
        u_refs[i] = u_ref;
    }

    let (u_opt, cost, iters, converged) = solver.solve(&x0, &x_refs, &u_refs);

    // Propagate x0 forward with the optimal first control for next call
    solver.sim_x0 = Some(solver.model.propagate_rk4(&x0, &u_opt));

    NmpcResult {
        u_opt: [u_opt[0], u_opt[1], u_opt[2], u_opt[3]],
        pos: [x0[0], x0[1], x0[2]],
        iterations: iters,
        converged,
    }
}
