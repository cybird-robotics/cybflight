# Cybflight — Codebase & Controller Overview

## High-Level

A Rust `no_std` quadcopter flight controller targeting STM32H743 (Embassy async runtime, defmt logging). The architecture cleanly separates **board-specific wiring** from **board-agnostic control logic** via static embassy `PubSubChannel`s.

```
HAL/BSP  →  drivers  →  sensor reader tasks  →  CHANNELS  →  estimation  →  control  →  motors
```

### Crate map
| Crate | Role |
|---|---|
| `bsp-{sakurah743,foxeerh743}` | Pin map + HAL init only |
| `cybflight-drivers` | Generic embedded-hal-async drivers (Icm426xx, Dps310, Mpu6x00, …) |
| `cybflight_core` | Pure-math control library: estimators, controllers, INDI, mixer, params |
| `cybflight` | Firmware binary: tasks, channels, board init, glue |

### Key control-relevant channels (`sensors/mod.rs`, `control/mod.rs`)
- `IMU_1` — fused IMU (`gyro_rad_s`, `accel_m_s2`) at ~8 kHz
- `VEHICLE_ATTITUDE` — Mahony filter output (quaternion only)
- `VEHICLE_ODOMETRY` — ESKF output (position, velocity, attitude, body-rate, biases)
- `RC_INPUT` — raw RC channels
- `MANUAL_CONTROL` / `AUTO_SETPOINT` — converted RC setpoints
- `ACTUATOR_MOTORS` — final motor commands (signal to DShot task)
- `DSHOT_TELEMETRY` — eRPM feedback from ESCs

### Two compile-time control configurations
| Feature | Estimator | Setpoint | Inner controller |
|---|---|---|---|
| `est_mahony` | Mahony filter (gyro+accel) | RC → tilt angle / yaw rate | cascaded **geometric attitude → rate PID → linear allocator** |
| `est_eskf` | Error-State Kalman Filter (Vicon-aided) | RC → ENU position | **position PD-FF → geometric attitude → INDI** (replaces rate PID + allocator) |

The IMU stream always drives the loop at 8 kHz; outer loops are decimated.

---

# Mathematical Walkthrough of the Controller

Signal path with explicit input/output for each module. Notation: world-frame (ENU/FLU) vectors get subscript `w`, body-frame `b`. Rotation `R = R_{wb}`; quaternion `q ∈ S³` represents the same.

## Module 1 — RC Interpreter (`control/rc_interpreter.rs`)

**Input:** raw RC channels `c ∈ ℕ¹⁶` (PWM µs).

### Mahony branch (manual mode)
Maps stick deflection through linear scaling (`RcMapper::aetr` with `RcSettings`):
$$
\theta_\text{roll}^\text{ref} = k_r\,\bar c_0,\quad
\theta_\text{pitch}^\text{ref} = k_p\,\bar c_1,\quad
\dot\psi^\text{ref} = k_\psi\,\bar c_3,\quad
T = \bar c_2
$$
with `k_r = k_p = 60°`, `k_ψ = 90°/s`. Output: `MANUAL_CONTROL` (tilt-angle commands + yaw rate + normalized thrust).

### ESKF branch (position mode)
Captures origin `p₀` from first converged odometry, then maps sticks to ENU position offsets:
$$
p^\text{ref} = p_0 + (\Delta x,\Delta y,\Delta z),\;\;
\Delta x = X_\text{half}\bar c_\text{pitch},\;\Delta y = X_\text{half}\bar c_\text{roll},\;\Delta z = Z_\text{range}\bar c_\text{thr}
$$
Output: `AUTO_SETPOINT` `(p^ref, v^ref=0, q^ref=I)`.
`X_half = 0.5 m`, `Z_range = 1.0 m`. A 1 cm dead-band suppresses re-publishing.

---

## Module 2 — Position Controller (`position_control/pd_ff_control.rs`)

Cascaded SE(3) controller from **Lee 2010**. Decimated to ~100 Hz (`POS_CTRL_DECIMATION = 80` IMU ticks).

**Input:** state `(p,v,q)` from ESKF; setpoint `(p^ref, v^ref, a^ref_{ff}, ψ^ref)`.

**Math:**

1. Saturated PD errors:
$$
e_p = \mathrm{clamp}(p^\text{ref}-p,\,\pm e_p^{max}),\quad
e_v = \mathrm{clamp}(v^\text{ref}-v,\,\pm e_v^{max})
$$
2. Desired world-frame acceleration (gravity-compensated):
$$
a_\text{des} = K_p\!\odot e_p + K_d\!\odot e_v + a^\text{ref}_{ff} + g\,\hat e_z
$$
3. Desired body-z (thrust direction) and yaw-aligned heading:
$$
z_b^\text{des} = \frac{a_\text{des}}{\|a_\text{des}\|},\qquad
x_c = (\cos\psi^\text{ref},\sin\psi^\text{ref},0)
$$
$$
y_b^\text{des} = \frac{z_b^\text{des}\times x_c}{\|z_b^\text{des}\times x_c\|},\qquad
x_b^\text{des} = y_b^\text{des}\times z_b^\text{des}
$$
$$
R_\text{des} = [x_b^\text{des}\;\;y_b^\text{des}\;\;z_b^\text{des}],\;\;\;q_\text{des} = \mathrm{quat}(R_\text{des})
$$
4. Collective thrust by projecting on **current** body z (Lee's formulation):
$$
F = m\,\bigl(a_\text{des}\cdot R\,e_z\bigr),\qquad F\leftarrow\max(F,0)
$$

**Output:** `(q_des, ω_des = 0, F)`. Note feedforward body-rate is currently zero.

In `est_eskf` it then divides `spf_sp_z = F/m` for INDI; otherwise `inner_loop` carries `F` forward as `collective_thrust_n`.

---

## Module 3 — Geometric Attitude Controller (`attitude_control/geometric_controller.rs`)

**Input:** state `(q, ω)`; setpoint `(q_des, ω^ref, α^ref)`.

Using the Lee SO(3) attitude error:
$$
e_R = \tfrac12\,\mathrm{vee}\!\bigl(R_\text{des}^\top R - R^\top R_\text{des}\bigr)\in\mathbb{R}^3
$$
The controller produces a **body-rate reference** (this is the path used by both flavors):
$$
\omega_\text{ref} = -\,\mathrm{clamp}\bigl(K_R^\text{rate}\!\odot e_R,\;\pm\omega_\text{max}\bigr)
$$
(The negative sign matches Lee's convention; `K_R^rate = att_k_rate` from params.)

It also computes a torque:
$$
\tau = -K_R^\tau\!\odot e_R + J\,\alpha^\text{ref}_\text{body} + \omega\times(J\omega)
$$
but the inner_loop **discards** `τ` to avoid double-acting on the attitude correction (only `ω_ref` is consumed).

**Output:** `body_rate_rad_s = ω_ref`, `torque_n_m = τ` (unused downstream).

---

## Module 4a — Rate PID + Linear Allocator (`est_mahony` only, `inner_loop.rs`)

**Input:** body-rate feedback `ω`, body-rate reference `ω_ref`, collective thrust `F`.

Three independent PID controllers (`discrete_pid` crate) at the IMU period `Δt = 125 µs`:
$$
\tau_i = K_{p,i}(\omega^\text{ref}_i - \omega_i) + K_{i,i}\!\!\int e\,dt + K_{d,i}\frac{de}{dt},\;\;i\in\{x,y,z\}
$$
The torque vector is the rate output (in `est_eskf` it is per-axis clamped, but the rate-PID path only runs in `est_mahony`).

Then **linear allocation** (`mixer::LinearAllocator`):
$$
\begin{bmatrix}F\\\tau_x\\\tau_y\\\tau_z\end{bmatrix}
= M\,\mathbf{u},\qquad \mathbf{u} = M^{+}\begin{bmatrix}F\\\tau\end{bmatrix}\in[0,1]^4
$$
where `M ∈ ℝ^{4×4}` is the geometric mixing matrix derived from `MotorParams` (arm lengths, spin direction, max thrust, drag-torque coefficient). For Mahony, throttles are linear; for ESKF (when this path is enabled) `cmd_i = √u_i` to invert the propeller `T ∝ ω²` map.

**Output:** `motor_commands ∈ [0,1]^4` published on `ACTUATOR_MOTORS`.

---

## Module 4b — INDI (`est_eskf`, `indi/controller.rs`)

This **replaces** the rate PID + linear allocator with **Incremental Nonlinear Dynamic Inversion + Weighted Least Squares allocation**, running every 125 µs.

**Inputs each tick:**
- `ω` (gyro, bias-corrected by ESKF)
- `f_b` (specific force, bias-corrected)
- `ω_ref` (from geometric attitude controller, decimated)
- `f_z^{sp} = F/m` (specific-force setpoint in body z)
- `armed` flag, `g2_valid[i]` (per-motor RPM-validity)
- Per-motor estimated rotor speed `Ω_i` (from `RpmTracker`/FOPDT EKF fed by DShot eRPM)

**Step 1 — sensor processing.** Finite-difference angular acceleration, all signals biquad LP-filtered at `sync_filter_hz`:
$$
\dot\omega_\text{raw} = (\omega_k - \omega_{k-1})\,f_s,\qquad
\dot\omega^{fs}, f_b^{fs}, u^{fs}, \Omega^{fs} = \mathrm{LPF}(\cdot)
$$
Motor-acceleration surrogate (since DShot provides Ω, not Ω̇):
$$
\dot\Omega_i^{fs} = \frac{\Delta u_{i,k-1}\,g_{2,\text{scale},i}}{\max(|\Omega_{i,k-1}^{fs}|,\,0.1\,\Omega_{\max,i})}
$$

**Step 2 — takeoff/ground detection.** Heuristic: `||ω||` small ∧ `||a||≈g` ∧ thrust setpoint low ⇒ `touching_ground=true` ⇒ disable INDI feedback (`do_indi_f = 0`).

**Step 3 — virtual rate command (P controller in angular acceleration):**
$$
\dot\omega^{sp} = K_\omega \odot (\omega^\text{ref} - \omega)
$$

**Step 4 — pseudo-control increment.** The "incremental" nature: rather than tracking absolute virtual control, track its **change** from the current filtered state.
$$
\Delta v =
\begin{bmatrix}0\\0\\f_z^{sp}\\\dot\omega^{sp}_x\\\dot\omega^{sp}_y\\\dot\omega^{sp}_z\end{bmatrix}
- \mathbb{1}_\text{indi}
\begin{bmatrix}0\\0\\f_{b,z}^{fs}\\\dot\omega^{fs}_x\\\dot\omega^{fs}_y\\\dot\omega^{fs}_z\end{bmatrix}
+ \mathbb{1}_\text{indi}\,G_2\,\dot\Omega^{fs}
$$
where `G₂ ∈ ℝ^{3×4}` accounts for the gyroscopic/coupled torque produced by **changing** rotor speeds.

**Step 5 — combined control-effectiveness matrix.**
$$
G = G_1(\Omega^{fs}) + G_2(\text{valid}) \in \mathbb{R}^{6\times4}
$$
- `G₁` is the linearized force/torque-per-throttle Jacobian, scaled by current rotor speeds (from `IndiEffectiveness::combined_g1g2`)
- `G₂` columns are zeroed for motors whose RPM telemetry is stale (`!g2_valid[i]`)

**Step 6 — WLS allocation.** Solve the box-constrained weighted least-squares problem for the **throttle increment** `Δu`:
$$
\min_{\Delta u}\;\;
\bigl\|W_v(G\,\Delta u - \Delta v)\bigr\|^2
+ \bigl\|W_u(\Delta u - \Delta u_\text{pref})\bigr\|^2
$$
$$
\text{s.t.}\quad \Delta u_\text{min}\le \Delta u\le \Delta u_\text{max}
$$
with
$$
\Delta u_\text{min} = -u^{fs},\quad \Delta u_\text{max} = u_\text{lim}-u^{fs},\quad \Delta u_\text{pref} = -u^{fs}
$$
(the pref pulls toward zero throttle when objectives are unconstrained). Solved by an active-set method (`flight_solver::cls::solve`) reusing warm-started working set `ws`.

**Step 7 — NaN guard.** On WLS NaN: increment counter, decay last `u` by 5%; ≥ `nan_limit` ⇒ failsafe.

**Step 8 — apply increment + thrust linearization.**
$$
u_i = \mathrm{clamp}(u^{fs}_i + \Delta u_i,\,0,\,u_{\text{lim},i})
$$
$$
\text{cmd}_i = \mathcal{L}_i(u_i) \;\;\text{(quadratic-blend thrust↦throttle linearization, parameter `nonlinearity`)}
$$
Then update `u_state` (PT1 actuator model with `α = Δt/(τ+Δt)`).

**Output:** `motor_commands ∈ [0,1]^4`, published as `ACTUATOR_MOTORS`. Heartbeat to `LAST_CONTROLLER_PUBLISH` is updated only on a successful, finite, fresh frame; staleness ⇒ silence ⇒ failsafe disarm.

### Auxiliary loops attached to INDI
- **RPM EKF** (`rpm_estimator.rs`): per-motor first-order-plus-dead-time observer fusing DShot eRPM and commanded throttle history. Output `Ω̂` feeds `G₁/G₂` scaling.
- **Slew-rate limiter** on raw eRPM: rejects samples whose `|ΔΩ|` exceeds `Ω_max/τ_min · Δt`, defending against GCR decode errors.
- **Online RLS Learner** (`indi/learner.rs`): when latched via prearm channel + learning toggle, estimates `G₁`/`G₂` from `(ω̇, f_b, Ω, u)`. Result committed to `IndiEffectivenessParams` on disarm and persisted to flash.

---

## Top-level signal flow (est_eskf, the active config)

```
RC ─► rc_interpreter ─► AUTO_SETPOINT (p^ref)
                                        │
ESKF (vicon+IMU) ─► VEHICLE_ODOMETRY ───┤
                                        ▼
   IMU 8 kHz ──► [decimate /80, 100 Hz] PositionController
                      │   (q_des, F)
                      ▼
                 GeometricAttitudeController
                      │   (ω_ref)
                      ▼
   IMU 8 kHz ──► INDI step (G₁,G₂, WLS)
                      │
                      ▼
                ACTUATOR_MOTORS ─► DShot task ─► ESCs
                      ▲
        DShot eRPM ─► RpmEstimator ─► Ω̂ ─┘  (also fed back into INDI G-matrices)
```

For `est_mahony` the chain instead is:
```
RC → tilt-angle setpoint → GeometricAttitudeController → ω_ref
                                       │
IMU 8kHz ──────────────────────────────┴─► RatePID (×3) → τ
                                                          │
                                       LinearAllocator (M⁺) → motor_commands
```

### Summary table — module I/O

| Module | Input | Output | Rate |
|---|---|---|---|
| RC interpreter | `RC_INPUT` | `MANUAL_CONTROL` *or* `AUTO_SETPOINT` | RC frame |
| Mahony / ESKF | `IMU_1` (+Vicon) | `VEHICLE_ATTITUDE` / `VEHICLE_ODOMETRY` | 8 kHz / 100 Hz |
| Position ctrl (Lee SE3) | `(p,v,q)`, `(p^ref,v^ref,ψ^ref)` | `(q_des, F)` | 100 Hz |
| Geometric att ctrl | `(q,ω)`, `(q_des,ω^ref)` | `ω_ref` (and unused τ) | 100 Hz |
| Rate PID (mahony) | `(ω, ω^ref)` | `τ` | 8 kHz |
| Linear allocator (mahony) | `(F,τ)` | `u∈[0,1]^4` | 8 kHz |
| INDI (eskf) | `(ω, f_b, ω^ref, f_z^{sp}, Ω̂, g2_valid, armed)` | `motor_commands∈[0,1]^4` | 8 kHz |
| RpmEstimator | DShot eRPM, throttle history | `Ω̂_i`, covariance | per DShot frame |
| Learner (RLS) | `(ω̇, f_b, Ω, u)` while armed+latched | `G₁`,`G₂` updates | 8 kHz |

Files referenced: `crates/cybflight/src/control/{rc_interpreter.rs,inner_loop.rs,indi_task.rs,mod.rs}`, `crates/cybflight_core/src/{position_control/pd_ff_control.rs,attitude_control/geometric_controller.rs,indi/controller.rs,mixer.rs}`.
