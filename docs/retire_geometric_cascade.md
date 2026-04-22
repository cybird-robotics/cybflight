# Retiring the PD+FF / Geometric Cascade

Planning memo. Context: firmware is migrating to INDI + MPC as the sole
outer/inner pair. The PD position controller (`pd_ff_control`) and geometric
attitude controller (`geometric_controller`) in `cybflight-core` exist only
to feed the legacy `cascade_task.rs` path. Once that task goes away the
production code has no consumers — but tests keep the modules alive.

## Conclusion

Yes, the sim autotests currently hold the legacy controllers back. The fix
is straightforward: **inline a stripped-down cascade into the sim crate as a
diagnostic baseline**, then delete the modules from `cybflight-core`. No
production code path is affected — the sim baseline is non-asserting and its
only consumer is the comparison table in `just sim-compare`.

## Current consumers

| Consumer | Role | Retires with fw? |
|---|---|---|
| `cybflight/src/control/cascade_task.rs` | Firmware outer loop (gated on `outer_geometric`) | **Yes** — deleted when the feature goes |
| `cybflight/src/control/indi_task.rs` (L16, L22) | Dead `use` imports only | Yes — auto-clean with any rustc warning sweep |
| `cybflight-sim/src/controller.rs::CascadeController` | Non-asserting diagnostic baseline | **No — inline here** |
| `cybflight-core/tests/control_convergence.rs::cascade_converges_50_orientations` | Legacy convergence test (50-tilt sweep) | **No — migrate or delete** |

## What becomes deletable

Once the two remaining consumers are removed/inlined:

| Path | LOC | Notes |
|---|---|---|
| `crates/cybflight_core/src/attitude_control/` | 282 | `geometric_controller.rs` + `mod.rs` (shared types only used by it) |
| `crates/cybflight_core/src/position_control/` | 302 | `pd_ff_control.rs` + `mod.rs` |
| `ControlGains::{pos_kp, pos_kd, att_k_rate, rate_kp, rate_ki, rate_kd}` in `params.rs` | ~30 lines | Plus all `ParamKey::PosKp*`, `RateKp*` variants and their flash serialization. INDI uses its own `indi_controller.rate_gains`. |
| `DEFAULT_CONTROL_GAINS` in `crates/cybflight/src/vehicle.rs` | 8 lines | Dead once `ControlGains` is trimmed |
| `crates/cybflight_core/tests/control_convergence.rs::cascade_*` | ~150 lines | Sim already covers convergence with more scenarios |

Net: ~600 LOC of `cybflight-core` and a chunk of persisted-param surface area.

## Proposed sim-side inline

Replace `cybflight-sim/src/controller.rs::CascadeController::from_params`
with a self-contained impl. Math is ~40 lines total:

```rust
// Hardcoded gains, matching the retired DEFAULT_CONTROL_GAINS. The sim
// cascade is diagnostic-only — no need to thread them through params.
const POS_KP: [f32; 3] = [4.0, 4.0, 8.0];
const POS_KD: [f32; 3] = [4.0, 4.0, 6.0];
const ATT_K_RATE: [f32; 3] = [3.0, 3.0, 1.0];
const RATE_KP: [f32; 3] = [0.1, 0.08, 0.05];
const RATE_CLAMP_NM: [f32; 3] = [0.8, 0.6, 0.15];

impl CascadeController {
    fn compute(...) -> [f32; 4] {
        // 1. Position → desired acceleration (PD + gravity)
        let pos_err = sp.position - state.position;
        let vel_err = sp.velocity - state.velocity;
        let a_des = kp.component_mul(&pos_err)
                  + kd.component_mul(&vel_err)
                  + sp.acceleration
                  + g_vec;
        // 2. Desired attitude from (a_des, yaw)
        // 3. Geometric attitude error → rate setpoint
        // 4. Rate-P → torque → mixer
    }
}
```

The math is small and doesn't need to stay in sync with anything — the
whole point is that cascade is frozen.

## Concrete work items

Execute in this order; each step is one reviewable commit.

1. **Drop firmware cascade**: delete `feature = "outer_geometric"`, delete
   `cascade_task.rs`, remove the related entries in `control/mod.rs`,
   clean the stale `use` lines in `indi_task.rs`. Verify `just check-all`
   still passes across all remaining feature combinations.

2. **Inline sim cascade**: rewrite
   `cybflight-sim/src/controller.rs::CascadeController` to hold the math
   directly; drop the `cybflight_core::{attitude_control, position_control}`
   imports. `just sim-compare` must produce numerically identical rows for
   the cascade baseline.

3. **Migrate or delete the core convergence tests**: the 50-orientation
   cascade sweep in `control_convergence.rs` can either move into
   `cybflight-sim/tests/` as a scenario sweep (preferred — the sim has
   richer metrics and visualization) or be deleted outright. The MPC
   convergence tests in the same file don't depend on these modules; they
   stay where they are.

4. **Delete core modules**: remove `cybflight-core/src/attitude_control/`
   and `cybflight-core/src/position_control/` and their `pub mod` lines in
   `lib.rs`. `cargo check -p cybflight-core` must stay green.

5. **Trim `ControlGains`**: remove `pos_kp`, `pos_kd`, `att_k_rate`,
   `rate_kp`, `rate_ki`, `rate_kd` fields from `ControlGains`. Remove the
   corresponding `ParamKey` variants and their serialization entries.
   Bump the flash format version / decide on a migration story: existing
   devices have these fields persisted; either read-and-discard on
   deserialization or bump the schema and accept the field loss.

6. **Clean `vehicle.rs`**: drop `DEFAULT_CONTROL_GAINS` or reduce it to
   just the fields INDI/MPC still consume.

## Risks & open questions

- **Flash param migration (step 5)**. The existing deploy has
  `ControlGains` fields written to NVRAM. The serializer in `params.rs`
  uses a fixed layout; changing field counts changes byte offsets for
  anything serialized after. Two options: (a) bump schema version and
  reject old blobs (user has to re-flash params); (b) keep the reserved
  bytes in the serializer for one release, stop reading them, then clean
  up next release. Decide before step 5 lands.

- **Is the sim cascade baseline still worth having?** If the answer is
  "no, MPC is good enough, drop the comparison column entirely" then skip
  step 2 and just delete `CascadeController` in step 4. The argument for
  keeping it: it's a ~50-line sanity floor in the comparison table; if
  MPC or INDI ever regresses badly the cascade baseline makes it obvious.
  The argument against: it's legacy math and we've just said so.

- **Is `control_convergence.rs::mpc_alloc_matches_firmware_mixer` still
  load-bearing?** That test verifies the MPC's `alloc` function matches
  the mixer. It doesn't touch the geometric controller but lives in the
  same file. When step 3 empties the cascade portion, this test should
  stay — consider splitting the file into `mpc_convergence.rs` and
  retiring `cascade_convergence.rs`.

- **`control_convergence.rs::bench_solve_runtime*` should survive** —
  those benchmarks drive MPC solver perf telemetry and have no cascade
  dependency.

## Non-goals

- Doing any of the above right now. Firmware retirement of
  `outer_geometric` is the trigger event; everything else slots in behind
  it. This memo is the plan, not the PR.
