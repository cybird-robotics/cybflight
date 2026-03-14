# Motor Mixing and Vehicle Configuration

This document covers the physical effectiveness model, motor allocation
algorithm, and how to configure a new vehicle.

## Body Frame Convention

Cybflight uses **FLU** (Forward-Left-Up) throughout all control and mixer code:

```
         x (forward)
         ↑
         |
y (left) ←------
                \
                 z (up, out of page)
```

| Axis | Direction       | Positive rotation (right-hand rule) |
|------|-----------------|-------------------------------------|
| x    | Forward         | Roll left (left side rises)         |
| y    | Left            | Pitch up (nose rises)               |
| z    | Up              | Yaw CCW from above                  |

> **Betaflight comparison.** Betaflight uses NED/FRD (Forward-Right-Down).
> When comparing mixer coefficients with Betaflight's `motorMixer_t` table:
> - Roll column signs are identical (left motors positive, right motors negative).
> - Pitch column signs are identical.
> - **Yaw column signs are inverted** (Betaflight positive yaw = CW from above;
>   cybflight positive yaw = CCW from above).
> - Motor positions: right motors have **negative** y in FLU.

---

## Motor Ordering

Cybflight follows Betaflight's QuadX motor index convention:

```
        FRONT
   M3 (FL)   M1 (FR)
     CW  \ / CCW
          X
     CCW / \ CW
   M2 (RL)  M0 (RR)
        REAR
```

| Index | Name       | Spin (from above) | Position in FLU    |
|-------|------------|-------------------|--------------------|
| 0     | REAR_RIGHT | CW                | (−d, −d)           |
| 1     | FRONT_RIGHT| CCW               | (+d, −d)           |
| 2     | REAR_LEFT  | CCW               | (−d, +d)           |
| 3     | FRONT_LEFT | CW                | (+d, +d)           |

For a QuadX with 100 mm arms at 45°, `d = 0.1 / √2 ≈ 0.0707 m`.

DShot outputs are expected to be wired in this order (index 0 = ESC/motor 0 in
the Betaflight motor test order).

---

## Effectiveness Matrix (G1)

The G1 matrix maps per-motor **throttle commands** u ∈ [0, 1]^N to a
**virtual control vector** v:

```
v = G1 · u

v = [ collective_thrust_N, roll_Nm, pitch_Nm, yaw_Nm ]ᵀ
```

Each column of G1 is the contribution of one motor at full throttle (u = 1.0).
For motor i at position (pxᵢ, pyᵢ), max thrust Tᵢ, torque coefficient cᵢ, and
spin sign sᵢ (+1 for CW-from-above, −1 for CCW-from-above):

```
G1[:, i] = [ Tᵢ,  pyᵢ·Tᵢ,  −pxᵢ·Tᵢ,  sᵢ·cᵢ·Tᵢ ]ᵀ
```

**Derivation.** Thrust force in FLU is F = (0, 0, +T). The lever-arm torque is:

```
τ = r × F,  r = (px, py, 0),  F = (0, 0, T)
τ_x = py·T    (roll)
τ_y = −px·T   (pitch)
```

The yaw reaction torque comes from the propeller drag:
a CW-from-above propeller (ω_z < 0 in FLU) experiences drag in the +z direction,
which by Newton's third law applies a +z torque to the body (positive yaw).
Hence `spin_sign = +1` for CW, `−1` for CCW.

### QuadX G1 (numeric, d = 0.0707 m, T = 5.9 N, c = 0.01 m)

|        | M0 RR (CW) | M1 FR (CCW) | M2 RL (CCW) | M3 FL (CW) |
|--------|-----------|------------|------------|-----------|
| Thrust | +5.9      | +5.9       | +5.9       | +5.9      |
| Roll   | −0.417    | −0.417     | +0.417     | +0.417    |
| Pitch  | +0.417    | −0.417     | +0.417     | −0.417    |
| Yaw    | +0.059    | −0.059     | −0.059     | +0.059    |

---

## Allocation (Inverse Problem)

Given a demanded virtual control vector v, find per-motor throttle commands u:

```
u = G1⁺ · v
```

`G1⁺` is the **right pseudoinverse** computed at startup:

```
G1⁺ = G1ᵀ · (G1 · G1ᵀ)⁻¹
```

This requires only one 4×4 matrix inversion, regardless of motor count N.

- **N = 4 (quad):** G1 is square; G1⁺ = G1⁻¹. The solution is exact.
- **N > 4 (hex, oct):** G1 is 4×N (fat). G1⁺ gives the minimum-norm solution,
  distributing load evenly across the extra motors.

`LinearAllocator::allocate` computes `G1⁺ · v` and clamps each output to
[0.0, 1.0]. Simple saturation is used — load redistribution on saturation
requires the WLS upgrade (see below).

---

## Thrust/Torque → Throttle in the Attitude Controller

```
┌──────────────────────┐
│  RC interpreter      │  thrust ∈ [0,1], rates in rad/s
└──────────┬───────────┘
           │ ManualControlSetpoint
┌──────────▼───────────┐
│ Geometric controller │  torque_n_m ∈ ℝ³
└──────────┬───────────┘
           │
┌──────────▼───────────────────────────────┐
│ LinearAllocator::allocate                │
│                                          │
│  total_thrust_N = thrust × max_thrust_N  │
│  demand = [total_thrust_N, τ_x, τ_y, τ_z]│
│  throttles = G1⁺ · demand               │
│  throttles.clamp(0, 1)                  │
└──────────┬───────────────────────────────┘
           │ [throttle₀, throttle₁, throttle₂, throttle₃]
┌──────────▼───────────┐
│  DShot output        │
└──────────────────────┘
```

`max_thrust_N` = sum of all motor maximum thrusts = `G1` row 0, summed.
`LinearAllocator::max_collective_thrust_n()` returns this value.

---

## Vehicle Configuration (`vehicle.rs`)

All vehicle-specific constants live in `crates/cybflight/src/vehicle.rs`.

```rust
pub const QUADROTOR_BODY: RigidBodyParams = RigidBodyParams {
    mass_kg: 1.5,
    inertia_kg_m2: [Ixx, 0, 0, 0, Iyy, 0, 0, 0, Izz],
};

pub const QUADROTOR_MOTORS: [MotorParams; 4] = [ /* M0..M3 */ ];

pub fn quadrotor_allocator() -> LinearAllocator<4> {
    LinearAllocator::new(MotorEffectiveness::from_motors(&QUADROTOR_MOTORS))
}
```

`QUADROTOR_BODY` and `QUADROTOR_MOTORS` are `const` — they live in flash.
`quadrotor_allocator()` builds the G1 matrix and computes G1⁺ at runtime; call
it once during startup and store the result (e.g. in the attitude task struct).

### Calibrating motor parameters

| Parameter         | How to obtain                                                          |
|-------------------|------------------------------------------------------------------------|
| `max_thrust_n`    | Motor test stand with a load cell. Measure thrust at 100% throttle.   |
| `torque_coeff_m`  | Divide measured reaction torque (N·m) by thrust (N) at each throttle point. Typically 0.005–0.02 for 5" props. |
| `mass_kg`         | Scale measurement.                                                     |
| `inertia_kg_m2`   | Bifilar pendulum test or CAD mass properties.                          |
| `position_m`      | Physical measurement from centre of mass to motor shaft.              |

---

## Adding a New Airframe

### Same motor count, different geometry (e.g. stretched X, dead-cat)

Update `QUADROTOR_MOTORS` with the correct per-motor positions and spin
directions. `from_motors` recomputes G1 and G1⁺ automatically.

### Different motor count (e.g. hexacopter with N=6)

1. Add `HEXACOPTER_MOTORS: [MotorParams; 6]` in `vehicle.rs`.
2. Create `hexacopter_allocator() -> LinearAllocator<6>`.
3. Update `attitude_control_task` to use `hexacopter_allocator()` and change
   the `motor_commands` array size from 4 to 6.
4. Expand `ActuatorMotors` message and DShot output to support 6 channels if
   not already done.

The G1⁺ formula (right pseudoinverse) handles N > 4 automatically; no changes
to `MotorEffectiveness` or `LinearAllocator` are required.

---

## Active-Set WLS Allocation

This section explains Indiflight's active-set weighted least-squares allocator
in detail, justifies why it is not adopted in the current codebase, and gives a
concrete adoption plan for the future.

### What the pseudoinverse allocator cannot do

`LinearAllocator` solves:

```
u = G1⁺ · v
```

This is an unconstrained minimum-norm problem. It has two blind spots:

1. **No awareness of motor limits.** When the solution requires a motor below 0
   or above 1, the output is simply clamped. The torque shortfall caused by
   the saturated motor is silently discarded — the remaining unsaturated motors
   are not adjusted to compensate. A roll command that saturates one motor will
   produce less roll than requested, with no recovery.

2. **No awareness of motor dynamics.** The static G1 model treats every motor
   as an instantaneous thrust source. Real motors have spin-up/spin-down lag
   on the order of 20–100 ms. This lag introduces a phase delay between the
   throttle command and the actual torque produced, which limits the bandwidth
   of the inner attitude control loop and can destabilise aggressive manoeuvres.

The active-set WLS allocator addresses both.

---

### Indiflight's WLS formulation

**Source:** `lib/main/ActiveSetCtlAlloc/src/common/setupWLS.{h,c}` and
`solveActiveSet_qr.c` in the Indiflight reference tree.

The allocator solves the following constrained quadratic programme at every
control loop iteration (~1 kHz):

```
min_{u}   ‖ B·u − ν ‖²_{W_ν}  +  γ² ‖ u − u_d ‖²_{W_u}

s.t.      u_min[i] ≤ u[i] ≤ u_max[i]   ∀ i
```

| Symbol  | Dimension | Meaning |
|---------|-----------|---------|
| u       | N×1       | Per-motor throttle increments Δu (INDI operates on changes, not absolute values) |
| B       | 6×N       | Combined G1+G2 effectiveness (see below) |
| ν       | 6×1       | Desired virtual control increment Δv = v_desired − v_measured |
| W_ν     | 6×6 diag  | Priority weights on each virtual control axis |
| W_u     | N×N diag  | Penalty on motor effort deviation from u_d |
| γ       | scalar    | Regularisation: trades virtual-control accuracy for actuator smoothness |
| u_d     | N×1       | Preferred actuator state (usually 0 increment = hover in place) |
| u_min/max | N×1    | Per-motor bounds (typically [−u_current, 1−u_current]) |

The six virtual control rows are [fx, fy, fz, τ_roll, τ_pitch, τ_yaw] in
Indiflight's FRD convention. For a standard quadrotor, fx and fy are
uncontrollable and their W_ν entries are set to zero; the allocation effectively
reduces to our 4-row formulation. On a tilt-rotor or omnidirectional vehicle,
all six rows are non-zero.

#### Reduction to standard least squares

The WLS problem is algebraically equivalent to an unconstrained least-squares
problem with box constraints. The solver builds:

```
A = [ √(W_ν) · B     ]      b = [ √(W_ν) · ν_des  ]
    [ γ√(W_u) · I    ]          [ γ√(W_u) · u_d   ]
```

so the cost becomes `‖Au − b‖²₂` subject to box constraints. This is a
bounded-variable least-squares (BVLS) problem.

#### Automatic γ estimation

Rather than requiring the user to set γ manually, Indiflight estimates the
minimum γ that keeps the condition number of A below a configured bound
(`wlsCondBound`). It uses the Gershgorin circle theorem to cheaply upper-bound
the maximum eigenvalue of B·W_ν·B^T:

```
γ_min = √( max_eigenvalue(B·W_ν·B^T) / cond_target )
```

A second condition (`wlsTheta`, the "objective separation ratio") enforces that
the regularisation term does not dominate the primary objective:

```
γ = max( γ_min,  √(max_singular_value) · θ / max(W_u) )
```

This automatic tuning means γ only needs to be touched if the automatic estimate
produces instability in practice.

---

### The G2 matrix: motor dynamics

G1 describes the static relationship between throttle and force/torque. G2
describes the transient: how a *change in motor speed* affects the torques on
the body.

```
v_actual(t) ≈ G1·u(t)  +  G2·ω̇(t)
```

**Physical origin.** A propeller's reaction torque is proportional to its
angular acceleration (gyroscopic cross-coupling and blade flapping). When motor i
spins up at rate ω̇ᵢ, it exerts an additional torque on the body given by
G2[:, i] · ω̇ᵢ. G2 has three rows (roll, pitch, yaw only — no thrust effect).

**Storage in Indiflight.** G2 coefficients (`actG2[3][N]`) are stored as `int16_t`
scaled by 1×10⁻⁵. At runtime they are converted to float and normalised:

```
G2_scaler[i] = 0.5 · actMaxOmega[i]² / actTimeConstS[i]
```

where `actMaxOmega` (rad/s) and `actTimeConstS` (s) are calibrated per-motor
from step-response tests.

**Real-time scaling by ω.** At each loop iteration, the effective G2 column for
motor i is:

```
G2_effective[:, i] = G2_scaler[i] × (1/ω_i) × G2[:, i]
```

This requires a live ω estimate. Indiflight uses DShot bidirectional telemetry
(EDT protocol) to read RPM from each ESC at loop rate. Without this, G2 must be
zeroed out and the controller degrades to a static G1-only allocation.

**Combined effectiveness matrix.** The solver receives B = G1 + G2_effective,
updated at every iteration.

---

### The active-set algorithm

Bounded-variable least squares is solved with an **active-set method**
(`solveActiveSet_qr.c`). The algorithm maintains a partition of motors into
*free* (within bounds, contribute to the unconstrained optimum) and *active*
(clamped at a bound). At each step:

1. **Solve the unconstrained sub-problem** over the free motors, fixing active
   motors at their bounds. Uses a QR factorisation of A restricted to free
   columns, updated incrementally.

2. **Check feasibility.** If the solution moves a free motor outside its bounds,
   find the step length α ∈ (0, 1] to the first constraint hit. Move along α and
   add that motor to the active set. Repeat from step 1.

3. **Check optimality.** Compute the dual variables (Lagrange multipliers) for
   the active constraints. If all multipliers are ≤ 0, the current point is the
   constrained minimum — stop. Otherwise, remove the active motor with the
   most-positive multiplier from the active set and repeat from step 1.

**Warm-starting.** The active set from the previous loop iteration is used as
the initial partition. Because flight conditions change slowly relative to the
loop rate, the warm-started active set is almost always already optimal, and the
solver converges in 0–2 additional pivot steps. In practice Indiflight sets
`wlsMaxIter = 1` — a single iteration is enough with warm-starting.

**Available implementations.** Indiflight ships three variants:

| Variant | File | Notes |
|---------|------|-------|
| Naive QR | `solveActiveSet_qr.c` (AS_QR_NAIVE) | Recomputes QR from scratch each call. Robust, slow. |
| Updated QR | `solveActiveSet_qr.c` (AS_QR) | Rank-1 QR updates; recommended for production. |
| Cholesky | `solveActiveSet_chol.c` (AS_CHOL) | Fastest, but numerically fragile in single precision. |

The QR variant is the default and the one to port first.

---

### Why we are not adopting it now

The active-set WLS allocator is clearly the right long-term solution, but three
hard prerequisites are missing from the current codebase, in dependency order:

#### 1. DShot bidirectional telemetry (blocking)

G2 compensation — the primary benefit over a static pseudoinverse — requires
per-motor RPM at loop rate. This means DShot300+ with EDT (Extended Digital
Telemetry) enabled in the ESC firmware, and the flight controller must decode the
reversed-polarity telemetry frame that follows each DShot packet.

The current `motors/dshot.rs` implements one-way DShot only. There is no timer
capture path, no EDT frame decoder, and no `omega` channel. Without this, G2
must be zeroed — reducing the WLS allocator to a more expensive version of the
pseudoinverse with no accuracy gain.

#### 2. INDI incremental control law (architectural mismatch)

Indiflight's WLS operates on **increments** Δu and Δv (hence the name
Incremental NDI). The bounds u_min and u_max are computed as deviations from the
*current motor state* u_current:

```
du_min[i] = 0   − u_current[i]   (can't go below zero)
du_max[i] = 1   − u_current[i]   (can't exceed full throttle)
du_pref[i] = 0  − u_current[i]   (prefer staying put)
```

The current attitude controller is a **geometric controller** that outputs
absolute torque commands, not incremental ones. Wiring the WLS allocator to an
absolute-command controller requires either converting the geometric output to
incremental form (subtracting the previous allocation result) or rewriting the
control law as INDI. Neither is trivial — the incremental formulation
fundamentally changes how disturbance rejection works.

#### 3. Motor parameter identification (calibration gap)

The G2 matrix coefficients (`actMaxOmega`, `actTimeConstS`) require per-motor
system identification from step-response data. The torque coefficients in
`MotorParams::torque_coeff_m` are already approximate; the dynamic coefficients
needed for G2 are harder to measure and more sensitive to propeller wear and
temperature. Using uncalibrated G2 is worse than using no G2.

#### Summary

| Prerequisite | Status | Estimated effort |
|---|---|---|
| DShot bidirectional / EDT | Not started | High — new timer capture path, EDT protocol, channel plumbing |
| INDI incremental control architecture | Not started | Medium — control law redesign or adaptation layer |
| G2 parameter identification tooling | Not started | Low–Medium — bench test rig and identification script |
| `solveActiveSet` port to `no_std` Rust | Not started | Medium — algorithm port is mechanical; careful numerics required |

Adopting the WLS allocator before items 1–3 are complete would add significant
complexity with zero improvement in flight performance. The pseudoinverse
allocator is correct and sufficient for the current rate-mode geometric
controller, which does not saturate motors under normal acro flight at the
configured gains.

---

### Adoption plan

When the prerequisites are ready, adopt in this sequence:

#### Step 1 — DShot bidirectional telemetry

Implement EDT receive in `motors/dshot.rs`:

1. After each DShot packet, reconfigure the timer capture input on the same GPIO
   (using a turn-around delay of ~25 µs for DShot300).
2. Decode the 21-bit EDT frame: 3-bit type field + 12-bit eRPM + 6-bit CRC.
3. Convert eRPM to rad/s: `ω = eRPM × (2π / 60) / (pole_pairs / 2)`.
4. Publish ω per motor to a new `MOTOR_RPM: Signal<[f32; N]>` channel.
5. Add `pole_pairs: u8` to `MotorParams`.

#### Step 2 — G2 identification

Add G2 coefficients to `MotorParams`:

```rust
pub struct MotorParams {
    // ... existing fields ...
    pub max_omega_rad_s: f32,    // motor max angular speed (rad/s)
    pub time_constant_s: f32,   // first-order spin-up time constant (s)
}
```

`G2_scaler[i] = 0.5 * max_omega²[i] / time_constant[i]` (Indiflight formula).

Measure `time_constant_s` from a step-response test (10%–63% of steady-state
RPM). Measure `max_omega_rad_s` from telemetry at full throttle.

#### Step 3 — Incremental control adaptation

Rather than rewriting the geometric controller as INDI, add a thin adaptation
layer in the attitude task:

```rust
// Convert absolute torque demand to incremental form
let v_prev = effectiveness.g1 * u_prev;   // reconstruct previous virtual control
let delta_v = demand - v_prev;            // incremental demand
let delta_u = wls_alloc(&g1g2, &delta_v, &du_min, &du_max, &w_v, &w_u, gamma);
let u_cmd = u_prev + delta_u;             // absolute command
```

This is an approximation (it ignores the G2 term in v_prev), but it is a
correct and stable starting point. Full INDI requires feeding back the
*measured* angular accelerations (from differentiated gyro), which is a separate
project.

#### Step 4 — Port `solveActiveSet` to `no_std` Rust

Port Indiflight's `solveActiveSet_qr.c` (AS_QR variant with incremental QR
updates) as a standalone crate:

```
crates/wls-alloc/
  src/
    lib.rs       — public API, generic over const N and M
    qr.rs        — incremental QR factorisation
    active_set.rs — pivot loop
```

Signature:

```rust
pub fn solve_active_set<const N: usize, const M: usize>(
    a: &SMatrix<f32, M, N>,   // stacked [√W_ν·B; γ√W_u·I]
    b: &SVector<f32, M>,      // stacked [√W_ν·ν; γ√W_u·u_d]
    u_min: &SVector<f32, N>,
    u_max: &SVector<f32, N>,
    warm_set: &mut [i8; N],   // active-set warm-start state (±1 or 0)
    max_iter: usize,
) -> SVector<f32, N>
```

Note: M = 6 + N (6 virtual control rows + N actuator regularisation rows).

Key numerical considerations when porting:

- Indiflight uses single-precision float throughout. The QR variant is stable in
  `f32`; the Cholesky variant is not — do not port the Cholesky solver first.
- The incremental QR rank-1 update (`cholupdate` / `qrdelete`) must be
  implemented carefully: use Givens rotations for stability rather than
  Householder for speed, at least initially.
- Warm-starting state (`Ws` array, ±1 = clamped at upper/lower bound, 0 = free)
  must be preserved across calls by the caller.

#### Step 5 — Replace `LinearAllocator::allocate`

Gate the upgrade behind `cfg(feature = "wls")`:

```rust
pub fn allocate(&self, demand: Vector4<f32>,
                omega: &SVector<f32, N>,
                omega_dot: &SVector<f32, N>,
                u_prev: &SVector<f32, N>,
                warm_set: &mut [i8; N]) -> SVector<f32, N> {
    #[cfg(feature = "wls")]
    {
        let b_eff = self.g1g2_from_omega(omega);
        let (a, b) = setup_wls(&b_eff, &(demand - self.g2_correction(omega_dot)),
                               &self.w_v, &self.w_u, self.gamma);
        let du_min = -u_prev;
        let du_max = SVector::repeat(1.0) - u_prev;
        let delta_u = solve_active_set(&a, &b, &du_min, &du_max, warm_set, 4);
        (u_prev + delta_u).map(|v| v.clamp(0.0, 1.0))
    }
    #[cfg(not(feature = "wls"))]
    { self.g1_pinv_allocate(demand) }
}
```

The `MotorEffectiveness::g1` field is already in the correct format. No changes
to upstream callers (`attitude_control.rs`) are required.

**Coordinate convention reminder.** Indiflight stores G1 and G2 in FRD. This
codebase uses FLU. The yaw column signs in G2 must be negated when porting
coefficient tables; or derive G2 coefficients directly in FLU using the same
cross-product formula as G1 (τ = r × F, plus the gyroscopic term).
