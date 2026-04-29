//! 2D `(thrust_N, voltage_V) → command [0,1]` lookup table for INDI thrust
//! linearization. Bilinear interpolation on a fixed-size grid.
//!
//! **Thrust axis convention.** `thrust_N` is *per-rotor* force in newtons,
//! matching INDI's `u * per_motor_max_n` convention. Bench rigs that report
//! collective (sum-of-N-motors) thrust must divide by the motor count
//! before constructing the table. The bake step in
//! `crates/cybflight/build.rs` does this conversion automatically for CSVs
//! produced by the `tmp/thrust_map/` rig.
//!
//! Mirrors `tmp/thrust_map/code/thrust_map.cpp` but with three changes:
//!
//! 1. Inverse deltas are pre-computed at construction so the hot path does
//!    no divisions — relevant at 8 kHz × 4 motors on STM32H743.
//! 2. Out-of-range inputs **clamp to the boundary** (both index and
//!    barycentric weights), instead of letting the bilinear corner
//!    extrapolate. Safer at low-voltage + max-thrust.
//! 3. `f32` storage (the C++ uses double-precision `Scalar`). Resolution is
//!    plenty for the 50×50 grid; halves RAM and is faster on the M7 FPU.
//!
//! The forward map gives `command` for a desired `(thrust, voltage)`. INDI
//! also needs the *inverse* — `output_curve(d, V) → thrust` — for
//! actuator-state estimation. The table is monotone in thrust along each
//! voltage row (the bench data is), so the inverse is computed by binary
//! search over the row at runtime. Cost: ~7 iters × a few muls per call,
//! still well under 1 µs on M7.


/// Pre-computed `(thrust, voltage) → command` bilinear lookup.
///
/// `N` is the grid resolution (square). Storage is `4·N²` bytes (`f32`).
/// For `N = 50` that's 10 kB — fits easily in DTCM/SRAM.
#[derive(Clone, Copy, Debug)]
pub struct ThrustTable<const N: usize> {
    /// `grid[i_voltage][i_thrust] → command ∈ [0,1]`.
    grid: [[f32; N]; N],
    thrust_min_n: f32,
    thrust_max_n: f32,
    voltage_min_v: f32,
    voltage_max_v: f32,
    /// `(N - 1) / (thrust_max - thrust_min)`. Pre-inverted so `lookup` does
    /// no divisions.
    inv_delta_thrust: f32,
    /// `(N - 1) / (voltage_max - voltage_min)`.
    inv_delta_voltage: f32,
}

impl<const N: usize> ThrustTable<N> {
    /// Construct from explicit ranges and a pre-built grid. Validates that
    /// `N ≥ 2` and the ranges are finite, monotone, and non-degenerate.
    /// Returns `None` if the table would be unusable.
    ///
    /// `const fn` so a baked table can live in flash via a `static`.
    pub const fn new(
        grid: [[f32; N]; N],
        thrust_min_n: f32,
        thrust_max_n: f32,
        voltage_min_v: f32,
        voltage_max_v: f32,
    ) -> Option<Self> {
        if N < 2 {
            return None;
        }
        if !(thrust_max_n > thrust_min_n) || !(voltage_max_v > voltage_min_v) {
            return None;
        }
        let inv_delta_thrust = (N as f32 - 1.0) / (thrust_max_n - thrust_min_n);
        let inv_delta_voltage = (N as f32 - 1.0) / (voltage_max_v - voltage_min_v);
        Some(Self {
            grid,
            thrust_min_n,
            thrust_max_n,
            voltage_min_v,
            voltage_max_v,
            inv_delta_thrust,
            inv_delta_voltage,
        })
    }

    /// Forward lookup: `(thrust_N, voltage_V) → command ∈ [0,1]`.
    ///
    /// Out-of-range inputs are clamped to the table boundary (both the cell
    /// index and the barycentric weight). NaN on either input falls back to
    /// the table's mid-cell on the corresponding axis — never produces NaN.
    #[inline]
    pub fn lookup(&self, thrust_n: f32, voltage_v: f32) -> f32 {
        // NaN guard: collapse non-finite to the midpoint. The table's job
        // is to never feed NaN to the WLS solver downstream — single
        // branch, predictable, costs nothing on the happy path.
        let t = if thrust_n.is_finite() { thrust_n } else { 0.5 * (self.thrust_min_n + self.thrust_max_n) };
        let v = if voltage_v.is_finite() { voltage_v } else { 0.5 * (self.voltage_min_v + self.voltage_max_v) };

        // Normalized cell coordinates ∈ [0, N-1]. Clamp first so the cast
        // to integer can't go out of range even for huge inputs.
        let n_minus_1 = (N - 1) as f32;
        let xt_unclamped = (t - self.thrust_min_n) * self.inv_delta_thrust;
        let xv_unclamped = (v - self.voltage_min_v) * self.inv_delta_voltage;
        let xt = if xt_unclamped < 0.0 { 0.0 } else if xt_unclamped > n_minus_1 { n_minus_1 } else { xt_unclamped };
        let xv = if xv_unclamped < 0.0 { 0.0 } else if xv_unclamped > n_minus_1 { n_minus_1 } else { xv_unclamped };

        // Cell origin and barycentric weight inside the cell.
        let it = (xt as usize).min(N - 2);
        let iv = (xv as usize).min(N - 2);
        let ft = xt - it as f32;
        let fv = xv - iv as f32;

        let c00 = self.grid[iv][it];
        let c01 = self.grid[iv][it + 1];
        let c10 = self.grid[iv + 1][it];
        let c11 = self.grid[iv + 1][it + 1];

        let row_lo = c00 + ft * (c01 - c00);
        let row_hi = c10 + ft * (c11 - c10);
        row_lo + fv * (row_hi - row_lo)
    }

    /// Inverse lookup: `(command, voltage_V) → thrust_N`.
    ///
    /// Used by INDI's `output_curve` (actuator-state estimation). Assumes
    /// the row at `voltage_v` is monotone non-decreasing in thrust, which
    /// is true for any physical thrust map. Binary search on the row,
    /// then linear interpolation between the bracketing cells.
    ///
    /// Out-of-range or non-monotone inputs return the boundary thrust.
    pub fn invert(&self, command: f32, voltage_v: f32) -> f32 {
        let v = if voltage_v.is_finite() { voltage_v } else { 0.5 * (self.voltage_min_v + self.voltage_max_v) };
        let n_minus_1 = (N - 1) as f32;
        let xv_unclamped = (v - self.voltage_min_v) * self.inv_delta_voltage;
        let xv = if xv_unclamped < 0.0 { 0.0 } else if xv_unclamped > n_minus_1 { n_minus_1 } else { xv_unclamped };
        let iv = (xv as usize).min(N - 2);
        let fv = xv - iv as f32;

        // Build the interpolated row at this voltage on the fly. It's only
        // accessed at the bracketing thrust columns we visit during the
        // binary search, so we don't materialize the whole row — we
        // interpolate the two cells we need each step.
        let row_at = |it: usize| -> f32 {
            let lo = self.grid[iv][it];
            let hi = self.grid[iv + 1][it];
            lo + fv * (hi - lo)
        };

        // Binary search for the largest `it` such that row[it] ≤ command.
        let target = if command.is_finite() { command } else { return self.thrust_min_n };
        let mut lo: usize = 0;
        let mut hi: usize = N - 1;
        // Boundary checks.
        if target <= row_at(0) {
            return self.thrust_min_n;
        }
        if target >= row_at(N - 1) {
            return self.thrust_max_n;
        }
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if row_at(mid) <= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let row_lo = row_at(lo);
        let row_hi = row_at(hi);
        let denom = row_hi - row_lo;
        let frac = if denom > 0.0 { (target - row_lo) / denom } else { 0.0 };
        let frac = if frac < 0.0 { 0.0 } else if frac > 1.0 { 1.0 } else { frac };

        let delta_thrust = (self.thrust_max_n - self.thrust_min_n) / n_minus_1;
        self.thrust_min_n + (lo as f32 + frac) * delta_thrust
    }

    pub fn thrust_min_n(&self) -> f32 { self.thrust_min_n }
    pub fn thrust_max_n(&self) -> f32 { self.thrust_max_n }
    pub fn voltage_min_v(&self) -> f32 { self.voltage_min_v }
    pub fn voltage_max_v(&self) -> f32 { self.voltage_max_v }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny synthetic linear map: command = (thrust - tmin)/(tmax - tmin),
    /// voltage-independent. Useful for analytical-truth tests.
    fn linear_table_3x3() -> ThrustTable<3> {
        // thrust ∈ [0, 10] N, voltage ∈ [20, 25] V.
        // command = thrust / 10, regardless of voltage.
        let grid = [
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ];
        ThrustTable::<3>::new(grid, 0.0, 10.0, 20.0, 25.0).unwrap()
    }

    /// Voltage-dependent: at low V we need more command for the same thrust.
    fn voltage_dependent_3x3() -> ThrustTable<3> {
        // Row 0 (low V): 1.5× command at any thrust (clamped).
        // Row 2 (high V): nominal.
        let grid = [
            [0.0, 0.75, 1.0], // 20 V — saturates earlier
            [0.0, 0.6,  1.0], // 22.5 V
            [0.0, 0.5,  1.0], // 25 V — best efficiency
        ];
        ThrustTable::<3>::new(grid, 0.0, 10.0, 20.0, 25.0).unwrap()
    }

    #[test]
    fn rejects_degenerate_ranges() {
        let g = [[0.0; 3]; 3];
        assert!(ThrustTable::<3>::new(g, 1.0, 1.0, 0.0, 1.0).is_none());
        assert!(ThrustTable::<3>::new(g, 1.0, 0.0, 0.0, 1.0).is_none());
        assert!(ThrustTable::<3>::new(g, 0.0, 1.0, 1.0, 1.0).is_none());
        assert!(ThrustTable::<3>::new(g, 0.0, 1.0, 1.0, 0.0).is_none());
    }

    #[test]
    fn corner_exact() {
        let t = linear_table_3x3();
        assert!((t.lookup(0.0, 20.0) - 0.0).abs() < 1e-6);
        assert!((t.lookup(10.0, 20.0) - 1.0).abs() < 1e-6);
        assert!((t.lookup(0.0, 25.0) - 0.0).abs() < 1e-6);
        assert!((t.lookup(10.0, 25.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linear_interior() {
        let t = linear_table_3x3();
        // Map is linear in thrust, voltage-independent → exact match.
        for i in 0..=10 {
            let thrust = i as f32;
            let expected = thrust / 10.0;
            for v in [20.0_f32, 22.5, 25.0] {
                let got = t.lookup(thrust, v);
                assert!((got - expected).abs() < 1e-5, "thrust={thrust} V={v}: {got} vs {expected}");
            }
        }
    }

    #[test]
    fn boundary_clamp() {
        let t = linear_table_3x3();
        // Below thrust_min → clamp to min.
        assert!((t.lookup(-100.0, 22.0) - 0.0).abs() < 1e-6);
        // Above thrust_max → clamp to max.
        assert!((t.lookup(1000.0, 22.0) - 1.0).abs() < 1e-6);
        // Below voltage_min and above voltage_max similarly clamp.
        assert!((t.lookup(5.0, 0.0) - 0.5).abs() < 1e-6);
        assert!((t.lookup(5.0, 100.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn nan_guard() {
        let t = linear_table_3x3();
        let r = t.lookup(f32::NAN, 22.0);
        assert!(r.is_finite());
        let r = t.lookup(5.0, f32::NAN);
        assert!(r.is_finite());
    }

    #[test]
    fn voltage_monotonicity() {
        // For fixed thrust, lower voltage should give ≥ command (more
        // throttle to make the same force).
        let t = voltage_dependent_3x3();
        let thrust = 5.0;
        let lo_v = t.lookup(thrust, 20.5);
        let hi_v = t.lookup(thrust, 24.5);
        assert!(lo_v >= hi_v - 1e-6, "lo_v={lo_v} hi_v={hi_v}");
    }

    #[test]
    fn invert_roundtrip_linear() {
        let t = linear_table_3x3();
        for i in 0..=10 {
            let thrust = i as f32;
            for v in [20.0_f32, 22.5, 25.0] {
                let cmd = t.lookup(thrust, v);
                let back = t.invert(cmd, v);
                assert!((back - thrust).abs() < 1e-3, "thrust={thrust} V={v} cmd={cmd} back={back}");
            }
        }
    }

    #[test]
    fn invert_clamps() {
        let t = linear_table_3x3();
        // Below the row floor → thrust_min.
        let r = t.invert(-1.0, 22.0);
        assert!((r - t.thrust_min_n()).abs() < 1e-6);
        // Above the row ceiling → thrust_max.
        let r = t.invert(2.0, 22.0);
        assert!((r - t.thrust_max_n()).abs() < 1e-6);
    }

    #[test]
    fn invert_voltage_dependent() {
        let t = voltage_dependent_3x3();
        // At the corner of voltage we know exactly: row [20V] = [0, 0.75, 1.0]
        // for thrusts [0, 5, 10]. So command 0.75 at 20V → thrust 5.
        let r = t.invert(0.75, 20.0);
        assert!((r - 5.0).abs() < 1e-3, "invert(0.75, 20V) = {r}, expected 5");
        // At 25V: command 0.5 → thrust 5.
        let r = t.invert(0.5, 25.0);
        assert!((r - 5.0).abs() < 1e-3, "invert(0.5, 25V) = {r}, expected 5");
    }
}
