# System identification from blackbox logs

`analysis/sysid_mcap.py` identifies a quadrotor dynamics model — thrust,
drag, actuator, and moment coefficients — from a single blackbox MCAP
recording (see [blackbox.md](blackbox.md) for the recorder itself). It is
a port of the fits in `optimal_quad_control_RL/analyze.py`, adapted to
cybflight's wire format and FLU/ENU frame conventions.

Reading order: **§1–§4** are the operating manual (record → pull → fit →
interpret). **§5** derives the model equations and the estimation math.
**§6** lists limitations and the assumptions baked into each fit.

---

## 1. What gets identified

| Fit | Model | Parameters | Topics needed |
|---|---|---|---|
| Thrust | `az = k_w·Σωᵢ²` | `k_w` | `/imu1`, `/motor_state` |
| Drag | `ax = k_x·vbx·Σωᵢ`, `ay = k_y·vby·Σωᵢ` | `k_x`, `k_y` | + `/odometry` |
| Actuator | `ω_c = (w_max−w_min)·√(k·u²+(1−k)·u) + w_min`; `ω̇ = (ω_c−ω)/τ` | `w_min`, `w_max`, `k`, `τ` (per motor + pooled) | `/motors`, `/motor_state` |
| Moments | `ṗ = Σk_pᵢωᵢ²`, `q̇ = Σk_qᵢωᵢ²`, `ṙ = k_r·Σsᵢωᵢ + k_rd·Σsᵢω̇ᵢ` | `k_p1..4`, `k_q1..4`, `k_r`, `k_rd` | `/imu1`, `/motor_state` |

All coefficients are **mass- or inertia-normalized**: the fits regress
against measured *accelerations* (specific force from the accelerometer,
angular acceleration from the differentiated gyro), never forces or
torques, because the blackbox does not know the vehicle's mass or
inertia. For the moments that is turned around: the inertia is the
*output*. With the (accurate) mass and the measured motor positions,
the roll/pitch fits identify Ixx/Iyy, and the yaw fit closed through
the YAML's `torque_coeff_m` gives Izz — an `inertia_kg_m2` line that
the existing geometric-G1 / mpc_full / cascade structure consumes
unchanged (§4.2). §5.6 covers the conversions.

### Units

| Parameter | Units | Typical magnitude / sign (FLU) |
|---|---|---|
| `k_w` | (m/s²) / (rad/s)² | ~1e-6, **positive** (z-up: hover az ≈ +9.81) |
| `k_x`, `k_y` | (m/s²) / ((m/s)·(rad/s)) | ~−1e-4 .. −1e-3, negative |
| `w_min`, `w_max` | rad/s | a few hundred / a few thousand |
| `k` | dimensionless ∈ [0,1] | ~0.4–0.8 |
| `τ` | s | 0.01–0.03 (10–30 ms) |
| `k_p1..4`, `k_q1..4` | (rad/s²) / (rad/s)² | ~1e-5; signs recover motor geometry (§4) |
| `k_r` | (rad/s²) / (rad/s) = 1/s | positive with the default spin signs |
| `k_rd` | (rad/s²) / (rad/s²), dimensionless | positive, small |

---

## 2. Recording a sysid flight

### Setup

```
> blackbox set sysid
> param save
```

The `sysid` tier records **only what this script reads** — `/imu1_raw`,
`/motors`, `/motor_state`, `/odometry` (when an estimator runs) plus
the Small bracket topics (`/events` for the ARM/DISARM crop, `/rc`) —
and raises `/motors` / `/motor_state` from 100 Hz to ≥500 Hz. The rate
matters most for the **actuator fit**: motor time constants are
10–30 ms, and 100 Hz sampling (10 ms) barely resolves them. The topic
set is deliberately narrow: at a true 8 kHz IMU an all-topics set is
~1.2 MB/s against the ~600 KB/s a commodity card sustains, and the
recorder then loses half of every topic in ~30 ms holes — including
the ones the fits need. This set is ~600 KB/s at 8 kHz, right at the
card ceiling; check the pulled file with `read_mcap.py --summary` and
expect `seq-gaps` ≈ 0 on `/imu1_raw`. If it still shows holes,
`param set blackbox_rate_div 2` (4 kHz raw) is the next step. The
thrust/drag/moments fits also work on a `mid` log (100 Hz motor data,
`/imu1` instead of raw; the ω̇ low-pass is tightened automatically).

### Flight excitation — what each fit needs

The fits only identify what the flight excites. A gentle hover
identifies almost nothing (and a log with motors never spinning
identifies exactly nothing — the script warns and returns zeros).

- **Thrust (`k_w`)**: throttle variation. Climbs, descents, punch-outs.
  Hover alone gives one operating point — enough for a rough `k_w` but
  no validation of the ω² shape.
- **Drag (`k_x`, `k_y`)**: sustained translational speed. Fly forward
  and sideways passes at several speeds, the faster the better. Requires
  a position estimate (`/odometry` non-empty, i.e. mocap or GPS build).
- **Actuator (`w_min`, `w_max`, `k`, `τ`)**: fast, large throttle
  transients — stick chirps and steps in `outer_rate`/`outer_geometric`.
  Steps are the cleanest τ excitation; a slow smooth flight makes τ
  unobservable.
- **Moments (`k_p/k_q/k_r/k_rd`)**: aggressive rotational maneuvers —
  roll/pitch flicks, yaw steps, ideally in all three axes separately.
  **Per-motor independence matters**: maneuvers that command all four
  motors in near-identical patterns leave the per-motor coefficients
  collinear (the regression can then split the total effect arbitrarily
  between motors). Mixed single-axis flicks in both directions
  decorrelate them. The yaw fit specifically needs yaw excitation —
  yaw torque is an order of magnitude weaker than roll/pitch, and
  without deliberate yaw steps `k_r`/`k_rd` fit mostly noise.

A good all-in-one sysid flight: arm → hover 5 s → throttle chirps
(slow → fast) → punch-out and drop → forward passes both directions →
sideways passes → roll flicks left/right → pitch flicks fore/aft →
yaw steps both directions → hover → disarm. 60–90 s total.

### Record and pull

```
[ pilot arms via RC → flies the excitation → disarms ]
$ just blackbox-pull            # lands in logs/flight_NNNN.mcap
```

or bench-style with `blackbox record on/off` (see
[blackbox.md](blackbox.md) §4 — but note bench logs with props off
identify motor dynamics only, not thrust/drag/moments, which need the
airframe actually accelerating).

---

## 3. Running the fits

```sh
pip install mcap cbor2 numpy scipy matplotlib     # once

python3 analysis/sysid_mcap.py logs/flight_0007.mcap              # all fits, interactive
python3 analysis/sysid_mcap.py f.mcap --fit actuator              # one fit only
python3 analysis/sysid_mcap.py f.mcap --save out --no-show        # out_thrust_drag.png,
                                                                  # out_actuator.png,
                                                                  # out_moments.png
python3 analysis/sysid_mcap.py f.mcap --t0 12 --t1 45             # fit a sub-window
python3 analysis/sysid_mcap.py f.mcap --no-crop                   # keep pre-ARM data
python3 analysis/sysid_mcap.py f.mcap --raw                       # /imu1_raw instead of /imu1
python3 analysis/sysid_mcap.py f.mcap --yaw-signs=1,-1,-1,1       # non-QuadX spin layout
python3 analysis/sysid_mcap.py f.mcap --cutoff 32                 # gentler ω̇-estimate LP
python3 analysis/sysid_mcap.py f.mcap --vehicle vehicles/sakura_vicon.yaml
                                                                  # + print paste-ready YAML (§4.1)
```

Flags:

| Flag | Meaning |
|---|---|
| `--fit {thrust,actuator,moments,all}` | which model(s); default all |
| `--t0 S` / `--t1 S` | fit window, seconds from the first sample — crop to the maneuver of interest |
| `--no-crop` | disable the default crop to the [first ARM .. last DISARM] window |
| `--raw` | use `/imu1_raw` (pre-biquad-LP, Large/sysid tier) — see §6 |
| `--yaw-signs` | per-motor spin signs for the yaw fit, +1 = CW from above, mixer order M0..M3. Default `1,-1,-1,1` = the QuadX layout in [motor_mixing.md](motor_mixing.md) |
| `--cutoff HZ` | zero-phase low-pass on the gyro derivative (default 64 Hz, auto-reduced on 100 Hz logs) |
| `--save PREFIX` / `--no-show` | write PNGs / skip the interactive windows |
| `--vehicle PATH` | read `mass_kg`, motor `pos_m` / `spin` / `torque_coeff_m` from a vehicle YAML; sets the yaw signs from it and prints the §4.1 YAML block — including the identified `inertia_kg_m2` line (needs `pyyaml`) |
| `--rotor-inertia J` | with `--vehicle`: prop+bell polar inertia [kg·m²], adds the `Izz = J_r/k_rd` cross-check (§4.2) |
| `--g1-override` | with `--vehicle`: also print the inertia-free 24-key `g1_*` block (§4.2b) |

The script resamples every topic onto one uniform grid at the median
`/motor_state` rate (linear interpolation) and prints the resolved rate,
sample count, and crop window before fitting.

---

## 4. Interpreting the output

Each figure overlays measurement vs. model prediction — **judge the fit
by eye before trusting the numbers**. A coefficient from a fit whose
prediction visibly doesn't track the data is a random number.

Console output ends with a copy-pasteable block:

```
── fitted parameters ──
  k_w = 1.9000e-06, k_x = -2.0010e-04, k_y = -2.3000e-04
  w_min = 300.00, w_max = 2800.00, k = 0.600, tau = 0.0200 s
  k_p1..4 = -2.000e-05, 2.100e-05, -1.900e-05, 2.200e-05
  k_q1..4 = 1.800e-05, -2.000e-05, 2.050e-05, -1.850e-05
  k_r = 3.820e-04, k_rd = 1.001e-04
```

Sanity checks against the QuadX geometry
([motor_mixing.md](motor_mixing.md): M0=RR/CW, M1=FR/CCW, M2=RL/CCW,
M3=FL/CW; FLU: +x forward, +y left, +z up):

- **Hover balance**: `k_w · Σω²_hover ≈ 9.81 m/s²`. If it's far off,
  the thrust fit or the eRPM pole-pair config is wrong.
- **`k_p` (roll, +x = left side rises)**: left motors positive, right
  negative → signs `(−, −, +, +)` for M0..M3.
- **`k_q` (pitch, +y = nose rises)**: rear motors positive, front
  negative → signs `(+, −, +, −)`.
- **`k_r`, `k_rd`**: both positive with the default `--yaw-signs`.
  Both negative usually means your props/spin layout is mirrored —
  refit with the signs flipped rather than accepting negative values.
- **Symmetry**: the four `|k_p|` should match within ~10–20% (same for
  `|k_q|`); a per-motor τ far from the other three suggests a damaged
  prop or dying motor, not model error.
- **Actuator**: per-motor fits (figure titles) vs. the pooled "Total
  fit" (suptitle + summary) should agree; large spread → check that the
  log is sysid tier (≥500 Hz), not 100 Hz.

### 4.1 Filling the vehicle YAML

Every identified value has a home in `vehicles/<vehicle>.yaml`
([sakura_vicon.yaml](../vehicles/sakura_vicon.yaml) is the reference
layout) — either under `airframe:` / `tuning:` (baked into the
firmware, read by INDI / the RPM estimator / the MPC) or under `sim:`
(read only by the host simulator's plant). **No new parameters are
needed** to consume the identified result — see the "why" after the
table. `sysid_mcap.py --vehicle vehicles/<v>.yaml` reads `mass_kg`
and each motor's `pos_m` / `spin` / `torque_coeff_m` from the YAML and
prints the block below with the numbers filled in — including the
identified `inertia_kg_m2` line (§4.2). It also takes the yaw-sign
pattern from the `spin:` entries, so `--yaw-signs` is not needed. The
step-by-step paste procedure is §4.3.

Conversions use `m = airframe.mass_kg` (accurate — weigh it), the
motor positions `pos_m` (measurable), `sᵢ` = +1 for `spin: cw`, −1 for
`ccw`, and `ω̄` = the mean rotor speed over the fit window (printed by
the script). **The inertia diagonal is deliberately *not* an input**:
it is the least-known number in the YAML, so §4.2 treats it as an
output of the moments fit instead.

| Identified | YAML key | Value | Consumer |
|---|---|---|---|
| `τ` (per motor i) | `tuning.mi_tau` | `τᵢ` from the *per-motor* fit (figure titles), not the pooled value — asymmetry is real | INDI G2 scaler `ω_max²/(2τ)`, actuator PT1, RPM-KF |
| `w_max` (per motor i) | `tuning.mi_omega_max` | `w_maxᵢ` per-motor | INDI G2 scaler, RPM-KF `c_m` seed |
| `k` | `tuning.mi_nonlin` (all four) | pooled `k` | INDI thrust linearization — the controller's `u = k·d² + (1−k)·d` is exactly the actuator fit's ω² curve (§5.3) with ω_min ≈ 0 |
| `k` (analytic thrust model) | `airframe.thrust_model: { type: quadratic, k: <k> }` | same `k` | Only if the vehicle uses an analytic model; a `table` model keeps its bench CSV and `mi_nonlin` becomes the fallback |
| `k_rd` | `tuning.mi_g2_ry` | `sᵢ · k_rd` (CW +, CCW −) | INDI G2 yaw term — `g2_ry` is *defined* as angular accel per rotor angular accel, i.e. `J_r/Izz` with the spin sign |
| `k_w` + `w_max` | `airframe.motors[i].max_thrust_n` | `m · k_w · w_maxᵢ²` | G1 collective/roll/pitch columns (`Tᵢ`); cross-check against the bench thrust table |
| `k_p`, `k_q`, `k_r` + `k_w` | `airframe.inertia_kg_m2` = `[Ixx, 0, 0, 0, Iyy, 0, 0, 0, Izz]` | `Ixx = m·k_w·pyᵢ/k_pᵢ`, `Iyy = −m·k_w·pxᵢ/k_qᵢ` (mean over the 4 motors); `Izz = torque_coeff_m·m·k_w·ω̄/k_r` | **The moments result, in the existing structure** (§4.2): these are the inertia values for which the firmware's geometric G1 (`pos_m`, `max_thrust_n`, `torque_coeff_m`, `I⁻¹`) reproduces the identified angular-acceleration effectiveness. Every consumer of `inertia_kg_m2` — INDI's geometric G1, mpc_full, the geometric cascade, the planner — is served by this one line |
| *(same)* | `airframe.motors[i].torque_coeff_m` | **unchanged** — kept from the YAML | Izz is closed through it, so it is an *input* here. Re-run the script if you ever change it; Izz moves proportionally |
| *(alternative)* | `tuning.g1_*` (24 keys, `--g1-override`) | `k_pᵢ·w_maxᵢ²`, `k_qᵢ·w_maxᵢ²`, `sᵢ·k_r·w_maxᵢ²/ω̄`, `k_w·w_maxᵢ²` | Inertia-free G1 override for INDI only (§4.2b). Not needed if you use the inertia line above; bypasses `inertia_kg_m2` for INDI but not for mpc_full/cascade |
| `k_x`, `k_y` | `sim.aero_drag: [−m·k_x, −m·k_y, 0.0]` | N·s²/(m·rad); sign flips because the sim writes `F = −c·Σω·v` and the fit returns a negative `k` | **Sim plant only** — no controller models drag |
| `w_min` | `sim.rotor_omega_min_rad_s` | pooled `w_min` | **Sim plant only** — the firmware's `indi_idle_norm` is a *command* floor, not a speed |
| `k` | `sim.rotor_throttle_curve_k` | pooled `k` | Sim plant (kept separate from `mi_nonlin` on purpose — the plant may be more curved than the controller's clamped inverse) |
| `k_rd` | `sim.rotor_inertia_kg_m2` | `Izz · k_rd` (the identified Izz) | Sim yaw reaction torque + rotor gyroscopic term |

What the script prints for `--vehicle vehicles/sakura_vicon.yaml`
(synthetic ground-truth run: `k_w = 1.9e-6`, `k = 0.6`, `τ = 0.02`,
`w_max = 2800`, `k_rd = 1.0e-4`, and `k_p`/`k_q`/`k_r` generated from
the file's own geometry with Ixx = 0.0021, Iyy = 0.0018, Izz = 0.003 —
i.e. the block below is what a perfect flight on the file's current
values would reproduce):

```
── vehicle YAML (see docs/system_id.md section 4.1) ──
# identified from blackbox; mean rotor speed 1800 rad/s
airframe:
  # Ixx per motor: [0.0021, 0.0021, 0.0021, 0.0021]   YAML: 0.0021
  # Iyy per motor: [0.0018, 0.0018, 0.0018, 0.0018]   YAML: 0.0018
  # Izz = torque_coeff_m=0.022 (YAML) * m*k_w*omega_mean/k_r   YAML: 0.003
  #     (re-derive if torque_coeff_m changes; cross-checks: Ixx+Iyy = 0.0039, J_r/k_rd = 0.003)
  inertia_kg_m2: [0.0021, 0.0, 0.0, 0.0, 0.0018, 0.0, 0.0, 0.0, 0.003]
  motors:   # max_thrust_n = m*k_w*w_max^2 ; torque_coeff_m kept from YAML (Izz above absorbs the yaw fit)
    - { pos_m: [-0.09, -0.086], spin: cw, max_thrust_n: 8.938, torque_coeff_m: 0.022 }
    - { pos_m: [0.054, -0.086], spin: ccw, max_thrust_n: 8.938, torque_coeff_m: 0.022 }
    - { pos_m: [-0.09, 0.086], spin: ccw, max_thrust_n: 8.938, torque_coeff_m: 0.022 }
    - { pos_m: [0.054, 0.086], spin: cw, max_thrust_n: 8.938, torque_coeff_m: 0.022 }
tuning:
  m0_tau: 0.0200          # ← per-motor fits, one value each
  m1_tau: 0.0200
  m2_tau: 0.0200
  m3_tau: 0.0200
  m0_omega_max: 2800
  m1_omega_max: 2800
  m2_omega_max: 2800
  m3_omega_max: 2800
  m0_nonlin: 0.600        # ← pooled k
  m1_nonlin: 0.600
  m2_nonlin: 0.600
  m3_nonlin: 0.600
  m0_g2_ry: 1.0000e-04    # ← s_i · k_rd
  m1_g2_ry: -1.0000e-04
  m2_g2_ry: -1.0000e-04
  m3_g2_ry: 1.0000e-04
sim:
  aero_drag: [1.2006e-04, 1.3800e-04, 0.0]
  rotor_omega_min_rad_s: 300.0
  rotor_throttle_curve_k: 0.600
  rotor_inertia_kg_m2: 3.0000e-07
```

The `#` comment lines under `airframe:` are diagnostics, not YAML to
keep: the four per-motor Ixx/Iyy estimates should agree (spread is a
fit/geometry problem, §4.2), and the Izz cross-checks should bracket
the printed value. `g1_*` keys are deliberately absent — they stay at
the all-zero "derive geometrically" sentinel, and the `inertia_kg_m2`
line is what carries the moments result.

**Why no new parameters.** Everything a controller in this firmware
actually models already has a registry key: the INDI inner loop
consumes `τ`, `ω_max`, `k`, and `G2_yaw`; the allocator consumes per-motor
thrust and torque-per-thrust; the RPM estimator consumes `τ` and
`ω_max`. The three identified quantities *without* a firmware sink —
`k_x`, `k_y` (drag) and `w_min` (idle floor) — are absent because no
controller models them: `mpc/quad_model.rs` has no drag term, and the
INDI linearization absorbs the idle offset into `nonlin`. They are
therefore plant-only (`sim:`), which is exactly what the `SimYaml`
docs call "the deliberately unmodelled part of the vehicle". Adding a
drag feed-forward to the MPC model or an explicit `ω_min` to the INDI
actuator model would be a controller-model change first; the param
(`mpc_drag_x`, `m*_omega_min`) would come with that change, not ahead
of it. The one signal that such a change is worth making: an
identified `w_min/w_max` above ~10%, or drag residuals in the thrust
figure that grow with speed.

### 4.2 Moments → `inertia_kg_m2` (the inertia is an output, not an input)

The moments fit is the only place inertia enters, and
`airframe.inertia_kg_m2` is the least reliable number in the YAML (a
CAD guess, or a bifilar-pendulum test nobody redoes after a rebuild).
The firmware's *structure* is kept as-is — geometric G1 from `pos_m`,
`max_thrust_n`, `torque_coeff_m` and `I⁻¹`, and the same `I` in
mpc_full / the cascade / the planner — and the sysid result is the
inertia triple for which that structure reproduces the flight:

**Ixx, Iyy.** `effectiveness.rs` builds the roll column of motor i as
`pyᵢ·Tᵢ/Ixx`; the fit measured it directly as `k_pᵢ·w_maxᵢ²`. With
`Tᵢ = m·k_w·w_maxᵢ²` (mass accurate, `k_w` from the thrust fit) the
`w_max` cancels and

```
Ixx = m · k_w · pyᵢ / k_pᵢ         one estimate per motor — the four should agree within ~10 %
Iyy = −m · k_w · pxᵢ / k_qᵢ
```

Nothing on the bench is needed, and the spread between the four
per-motor estimates is a direct check of the fit, of `pos_m`, and of
the IMU-to-body alignment. The script prints all four and writes the
mean.

**Izz.** Yaw is only observable as the ratio `c_m/Izz` (drag torque)
and `J_r/Izz` (spin-up reaction) — a property of the physics, not of
the script. The existing parameter structure already carries the extra
constraint that closes it: `torque_coeff_m` (`c_m`, yaw torque per
newton). Matching the geometric yaw column `sᵢ·c_m·Tᵢ/Izz` to the
identified `sᵢ·k_r·w_maxᵢ²/ω̄` (the linear-in-ω fit re-quadratized
about the mean speed, §5.6) gives

```
Izz = c_m · m · k_w · ω̄ / k_r
```

So `torque_coeff_m` stays what it is in the YAML and Izz absorbs the
yaw fit. Two consequences to keep in mind:

- Izz is *defined relative to* `torque_coeff_m`. Change one and the
  other must follow — re-run the script; it recomputes Izz from the
  current YAML value. The yaw dynamics every consumer sees depend only
  on the ratio, which is what the flight pinned down.
- The absolute Izz still matters, weakly, through the gyroscopic
  coupling `(Iyy − Izz)·p·r/Ixx` in mpc_full and the cascade
  ([gyroscopic_term_analysis.md](gyroscopic_term_analysis.md)). The
  script therefore prints two independent cross-checks next to it:
  the perpendicular-axis estimate `Ixx + Iyy` (exact for a planar mass
  distribution, typically 10–20 % high for a real quad), and
  `J_r/k_rd` if you pass `--rotor-inertia <prop+bell polar inertia>`.
  If the three disagree by more than ~30 %, `torque_coeff_m` is the
  suspect (0.022 in the reference YAML is a "typical 5-inch" number,
  not a measurement) — then either measure `c_m` on a thrust stand
  (yaw torque / thrust) or, if you trust `J_r`, set
  `torque_coeff_m = J_r·k_r/(k_rd·m·k_w·ω̄)` and let Izz = `J_r/k_rd`.

`sim.rotor_inertia_kg_m2 = Izz·k_rd` uses the same Izz, so the sim's
yaw reaction torque matches the flight too.

#### 4.2b Alternative: inertia-free G1 override (INDI only)

Because INDI's G1 rows are stored as angular acceleration per unit
throttle (`I⁻¹·τ`, `effectiveness.rs`) and the `g1_*` override params
write straight into those rows, INDI can also take the moments result
without any inertia: `--g1-override` prints the 24-key block
(`g1_rr_mi = k_pᵢ·w_maxᵢ²`, `g1_rp_mi = k_qᵢ·w_maxᵢ²`,
`g1_ry_mi = sᵢ·k_r·w_maxᵢ²/ω̄`, `g1_fz_mi = k_w·w_maxᵢ²`, `fx`/`fy` = 0).
This bypasses `inertia_kg_m2` for the INDI inner loop only —
mpc_full, the cascade and the planner still read the inertia line —
so with the identified inertia in place it buys nothing; it is there
for the case where you want INDI pinned to flight data while the
inertia is still under dispute.

### 4.3 Drop-in procedure

1. **Fit with the vehicle file you fly:**
   ```sh
   python3 analysis/sysid_mcap.py logs/flight_NNNN.mcap \
       --vehicle vehicles/sakura_vicon.yaml --save sysid_NNNN --no-show
   ```
   Look at the three PNGs first (§4); only paste numbers from fits
   whose prediction tracks the data. `--fit` lets you re-run one model
   on a cropped window (`--t0/--t1`) if e.g. only the yaw steps need
   re-fitting.

2. **`airframe:` block — apply to *every* YAML that claims the same
   `airframe.name`.** `airframe.name` identifies the physical drone,
   and `crates/vehicle_yaml/tests/airframe_identity.rs` requires the
   whole `airframe:` block (mass, inertia, thrust_model, motors) to be
   identical across all files with that name. The reference airframe is
   `sakura_01`; `rg -l "name: sakura_01" vehicles/*.yaml` lists its configurations. Edit in each:
   - `inertia_kg_m2:` → the printed line (Ixx, Iyy identified; Izz
     derived through the file's `torque_coeff_m`, §4.2).
   - `motors[i].max_thrust_n:` → the printed values. **If the file uses
     `thrust_model: { type: table, … }`**, `max_thrust_n` is also the
     scale that maps `u ∈ [0,1]` onto the bench table, so compare the
     identified value with the table's full-throttle thrust before
     overwriting; a large difference means bench and flight disagree
     (pack voltage, prop wear) and is worth understanding rather than
     silently rescaling the table.
   - `torque_coeff_m`, `pos_m`, `spin`: unchanged (inputs to the fit).
   - `thrust_model:` — only for an analytic model, set
     `{ type: quadratic, k: <k> }`; leave a `table` model alone.
   `mass_kg` is not touched — weigh the vehicle; the fit assumes it.

3. **`tuning:` keys — per file, but copy them to all files of the same
   airframe anyway** (they are hardware facts; the identity test does
   not enforce `tuning:`, the pin-what-you-fly policy in the YAML
   comments does). Replace the existing `m0..m3_tau`, `m0..m3_omega_max`,
   `m0..m3_g2_ry` values and add `m0..m3_nonlin` (currently unpinned in
   `sakura_bench.yaml` because it was an all-zero sentinel — once
   identified it is a real value and belongs in the file). Leave
   `g1_*` absent/zero.

4. **`sim:` block — only into `vehicles/sim_baseline.yaml`, and only
   deliberately.** The firmware bake ignores `sim:`; the host simulator
   reads it from the *frozen* baseline, which by design lags the flight
   tune (CLAUDE.md, "Sim regression snapshot"). Updating it changes the
   plant, so pair it with `just sim-snapshot`, review the numeric diff,
   and commit snapshot + YAML together. Don't paste `sim:` into the
   flight vehicle files — it is parsed but unused there.

5. **Follow-on tunables now that the motor model is identified**
   (comments in `sakura_bench.yaml` already anticipate this):
   `indi_omega_kf: 1` (RPM-KF estimate instead of the held DShot value —
   safe once `m*_tau`/`m*_omega_max` are measured, not defaults) and
   `rpm_est_thr_psd` from `1.25e-6` toward `1.25e-8` (the KF's process
   noise was sized for ~25 % model error).

6. **Verify and build:**
   ```sh
   just test                         # airframe_identity across the same-name files
   just print-features <vehicle>     # bake dry run — warns on unpinned keys
   just build <vehicle>              # or just flash <vehicle>
   ```
   After flashing, `param diff` over USB should be empty against the
   new bake (no stale flash overrides of the keys you just changed —
   `param reset <key>` any that linger, or the flash value wins).

7. **Live-tuning alternative for the `tuning:` keys.** `m*_tau`,
   `m*_omega_max`, `m*_nonlin`, `m*_g2_ry` are `live` params:
   `param set m0_tau 0.018` … `param save` on the bench, then
   `just param-sync <vehicle>` writes the flash state back into the
   YAML. `airframe.*` (inertia, motors) is `reboot`-class and only
   changes through the YAML + rebuild.

8. **Close the loop.** Fly the same excitation once more on the new
   bake and re-fit: `k_p`/`k_q`/`k_r` should come back within a few
   percent (they are properties of the airframe, not of the tune), and
   the per-motor Ixx/Iyy should now print equal to the YAML values in
   the diagnostic comments. That is the regression test for the whole
   procedure.

---

## 5. Derivations

### 5.0 Frames, notation, measurements

FLU body frame (x forward, y left, z up), ENU world frame, per
[motor_mixing.md](motor_mixing.md). `R(q)` is the body→world rotation
from the `/odometry` quaternion `[w, i, j, k]`. Body rates
`(p, q, r)` = gyro x/y/z. Rotor speeds `ωᵢ` (rad/s) and accelerations
`ω̇ᵢ` come from `/motor_state` (KF-fused from bidir-DShot eRPM).
Commands `uᵢ ∈ [0,1]` come from `/motors`.

The **accelerometer measures specific force**, not coordinate
acceleration:

```
f_body = R(q)ᵀ · (a_world − g_world),      g_world = (0, 0, −9.81) m/s²
```

i.e. gravity never appears in the accelerometer output directly — at
rest `f_body = (0, 0, +9.81)`. This is why the thrust/drag models below
regress on raw accelerometer axes with no gravity-compensation step:
the non-gravitational (aerodynamic + propulsive) forces are exactly
what the sensor reads, divided by mass.

### 5.1 Thrust model

Momentum/blade-element theory gives a single rotor's thrust as

```
Tᵢ = c_T · ρ · A · r² · ωᵢ²  ≡  k_T · ωᵢ²
```

with `c_T` the thrust coefficient, `ρ` air density, `A` disk area, `r`
rotor radius. Summing four rotors, all thrusting along body +z, and
dividing by mass m (the accelerometer sees force/mass):

```
f_z = (1/m) Σ Tᵢ = (k_T/m) Σωᵢ²  ≡  k_w Σωᵢ²
```

So `k_w = k_T/m`, one scalar assuming four identical rotors. This is a
**linear least-squares problem** in `k_w`:

```
minimize over k_w:   Σₜ ( az(t) − k_w · Σᵢωᵢ²(t) )²
```

solved in closed form (`np.linalg.lstsq` on the N×1 regressor matrix).
Sign note: in the FRD original `az ≈ −9.81` at hover and `k_w < 0`;
in cybflight's FLU frame both are positive.

### 5.2 Drag model

Following Eq. 2 of the lumped-drag model the original script cites
(https://doi.org/10.1016/j.robot.2023.104588), the dominant horizontal drag on a rotorcraft is **induced drag from blade
flapping**, proportional to the product of body-frame airspeed and
total rotor speed (not airspeed squared — at quadrotor scales the
rotor-induced term dominates classical `v²` fuselage drag):

```
f_x = −(k_d/m) · v_bx · Σωᵢ  ≡  k_x · v_bx · Σωᵢ
f_y = −(k_d/m) · v_by · Σωᵢ  ≡  k_y · v_by · Σωᵢ
```

fitted independently per axis (k_x ≠ k_y in general — the frame is not
rotationally symmetric). Again linear least squares, one scalar each.

The body-frame velocity is not logged directly; it is reconstructed
from `/odometry` (world velocity `v_w` + attitude quaternion):

```
v_b = R(q)ᵀ · v_w
```

with the quaternion normalized first (interpolating quaternions
component-wise slightly denormalizes them; for the small inter-sample
rotations at 1 kHz odometry this renormalization is sufficient — no
slerp needed). Air velocity is approximated by ground velocity, i.e.
**zero wind is assumed** — fly sysid indoors or on a calm day, or fly
symmetric passes in both directions so wind averages out of the fit.

Thrust coupling caveat: `f_x`/`f_y` also contain small projections of
thrust when the rotor plane is tilted relative to the body (rotor
flapping), which this model lumps into `k_x`/`k_y`. That is intended —
the same lumped term is what a controller/simulator using this model
will apply.

### 5.3 Actuator model

Two parts: a static command→steady-state-speed map, and first-order
lag dynamics.

**Static map.** ESCs approximately linearize *thrust* against throttle
`u`. Since `T ∝ ω²`, a thrust-linear ESC would give `ω_ss ∝ √u`; a
speed-linear ESC would give `ω_ss ∝ u`, i.e. `T ∝ u²`. Real
ESC/motor/prop combinations sit between the two, so both regimes are
blended under one square root with a shape parameter `k ∈ [0,1]`:

```
ω_c(u) = (w_max − w_min) · √( k·u² + (1−k)·u ) + w_min
```

- `k = 0`: `ω_c ∝ √u` — thrust linear in throttle.
- `k = 1`: `ω_c ∝ u` — speed linear in throttle.
- Endpoints are exact by construction: `ω_c(0) = w_min` (idle speed,
  DShot idle throttle), `ω_c(1) = w_max`.

**Dynamics.** The motor+ESC speed loop is modeled as a first-order lag
toward the commanded steady state:

```
ω̇ = (ω_c(u) − ω) / τ
```

**Discretization + fitting.** On the uniform grid (step dt) the ODE is
forward-Euler discretized:

```
ω̂[n] = ω̂[n−1] + ( ω_c(u[n]) − ω̂[n−1] ) · dt/τ
```

which is a first-order IIR filter — the script evaluates it with
`scipy.signal.lfilter(b=[a], a=[1, a−1])`, `a = dt/τ`, seeded so
`ω̂[0]` equals the measured `ω[0]`. This is **output-error (simulation
error) estimation**: the model is rolled forward from the initial
condition only, never re-anchored to measurements, and

```
minimize over (w_min, w_max, k, τ⁻¹):   ‖ ω̂(θ; u) − ω ‖₂
```

is solved with `scipy.optimize.minimize` (bounded, from a physically
plausible initial guess). Output-error is deliberately chosen over
one-step-ahead prediction error: one-step fits on smooth data are
dominated by the trivial `ω̂[n] ≈ ω[n−1]` solution and underestimate τ,
while simulation error forces the dynamics to explain the whole
transient. Fitting `τ⁻¹` rather than τ keeps the bounded search
well-conditioned near fast motors. Five fits run: one per motor
(catching per-motor asymmetry) plus one pooled fit over all four (the
headline numbers).

### 5.4 Moments model

Euler's rotational equation for a rigid body with diagonal inertia
`I = diag(Ixx, Iyy, Izz)`:

```
Ixx·ṗ = (Iyy − Izz)·q·r + Mx
Iyy·q̇ = (Izz − Ixx)·p·r + My
Izz·ṙ = (Ixx − Iyy)·p·q + Mz
```

The gyroscopic cross terms `(I−I)·q·r` are dropped (near-symmetric
quads have `Ixx ≈ Iyy` and the products of rates are small relative to
motor torques at sysid excitation levels; the original fits behaved
identically with them in or out — see also
[gyroscopic_term_analysis.md](gyroscopic_term_analysis.md)). What
remains is regressed inertia-normalized:

**Roll / pitch.** Motor thrust `k_T·ωᵢ²` at lever arm `(pxᵢ, pyᵢ)`
produces `Mx = Σ pyᵢ·k_T·ωᵢ²` and `My = −Σ pxᵢ·k_T·ωᵢ²`. Rather than
assuming the geometry, the per-motor lumped coefficients are fitted
directly:

```
ṗ = k_p1·ω₁² + k_p2·ω₂² + k_p3·ω₃² + k_p4·ω₄²      k_pᵢ = pyᵢ·k_T/Ixx
q̇ = k_q1·ω₁² + ... + k_q4·ω₄²                       k_qᵢ = −pxᵢ·k_T/Iyy
```

Linear least squares, 4 coefficients each; the recovered signs should
reproduce the motor layout (§4) — a free consistency check.

**Yaw.** Two mechanisms, both linear in the fitted quantities:

1. *Rotor drag (reaction) torque*: the motor torque that sustains each
   rotor against aerodynamic drag reacts on the airframe opposite the
   rotor's spin. Aerodynamically this torque is `∝ ω²`, but over the
   flight envelope's ω range a linear approximation `∝ ω` fits as well
   (the range is far from ω = 0) and matches the reference model — so
   the term is `k_r · Σ sᵢωᵢ` with sᵢ the spin sign (+1 = CW from
   above, which in FLU produces **positive** body yaw — see
   motor_mixing.md "spin_sign").
2. *Rotor inertia reaction*: accelerating a rotor with inertia `J_r`
   requires torque `J_r·ω̇ᵢ`, whose reaction is exactly linear:
   `k_rd · Σ sᵢω̇ᵢ` with `k_rd = J_r/Izz`. This term is what makes yaw
   respond *instantly* to motor commands (before the drag torque
   builds), and is why the fit uses `/motor_state`'s KF-fused `ω̇`
   directly instead of numerically differentiating eRPM.

```
ṙ = k_r · Σᵢ sᵢ·ωᵢ  +  k_rd · Σᵢ sᵢ·ω̇ᵢ
```

Linear least squares, 2 coefficients. Note the reduction: fitting
8 free per-motor coefficients instead would let the regression exploit
collinearity between motors; pinning the known spin pattern into `s`
makes the two physical mechanisms identifiable with far less yaw
excitation.

**Angular acceleration** `(ṗ, q̇, ṙ)` is not logged; it is estimated as
the time-gradient of the gyro rates followed by a zero-phase (forward–
backward, `sosfiltfilt`) 2nd-order Butterworth low-pass, default 64 Hz.
Differentiation amplifies noise ∝ frequency, so the low-pass sets the
bias/variance trade-off: too low smears the flick transients that carry
the information; too high lets D-term noise dominate the regression.
Zero-phase filtering matters — a causal filter would delay `ṙ` relative
to the (unfiltered) regressors and bias every coefficient toward zero.
On 100 Hz logs the cutoff is clamped to 0.45·fs to stay below Nyquist.

### 5.5 Estimation machinery (common)

- **Least-squares fits** solve `min‖Xᵀθ − y‖₂` via `np.linalg.lstsq`
  (SVD — no explicit normal equations, so rank-deficient excitation
  degrades gracefully to minimum-norm rather than exploding).
- **Resampling**: topics log at different rates (`/imu1` ~1 kHz,
  `/motor_state` 100–500 Hz, `/odometry` ~1 kHz) with independent
  jitter. Everything is linearly interpolated onto one uniform grid at
  the median `/motor_state` rate — the rotor-speed channels are the
  bandwidth bottleneck, so interpolating the faster channels *down*
  onto their grid loses nothing the fits could use, and a uniform grid
  is required by both the IIR actuator propagation and `sosfiltfilt`.
- **Windowing**: fitting only inside [ARM .. DISARM] excludes the
  on-bench idle before arming (motors at zero — a large block of
  zero-excitation samples would dilute every regression) and any
  post-disarm tail.

### 5.6 Recovering physical parameters

With mass m and inertia diag(Ixx, Iyy, Izz) measured separately:

```
k_T  = m · k_w                      [N/(rad/s)²]  per-rotor thrust coeff (Σω² already sums the rotors)
lever·k_T = Ixx · k_pᵢ              [N·m/(rad/s)²] rotor i roll moment coeff  (= pyᵢ·k_T)
             Iyy · k_qᵢ             [N·m/(rad/s)²] rotor i pitch moment coeff (= −pxᵢ·k_T)
c_Q  ≈ Izz · k_r / ω̄               yaw drag torque coeff, if re-quadratized about mean speed ω̄
J_r  = Izz · k_rd                   [kg·m²] rotor+prop inertia
```

Since mass and motor positions are measurable while inertia is not,
the useful direction is the reverse for the roll/pitch axes:
`Ixx = m·k_w·pyᵢ/k_pᵢ`, `Iyy = −m·k_w·pxᵢ/k_qᵢ` (§4.2). Izz is
identifiable only up to the ratios `c_m/Izz` and `J_r/Izz`.

For the INDI effectiveness matrix the conversion is not needed at all:
the firmware stores G1 torque rows as angular acceleration per unit
throttle (`I⁻¹·τ`), so `g1_rr_mi = k_pᵢ·w_maxᵢ²` etc. can be pinned
directly ([motor_mixing.md](motor_mixing.md), §4.2).

---

## 6. Limitations and assumptions

- **Zero wind** (§5.2): body airspeed ≈ ground velocity from odometry.
- **Rigid body, diagonal inertia, gyroscopic terms dropped** (§5.4).
- **Identical rotors** for thrust/drag (`k_w`, `k_x`, `k_y` are
  totals); the actuator and roll/pitch fits are per-motor and catch
  asymmetry.
- **Post-filter IMU by default**: `/imu1` is post-biquad-LP, so the
  firmware's IMU filter shapes what the thrust/drag/moments fits see.
  For sysid the filter delay is common-mode (regressand and regressors
  are both low-passed or slow), but if you suspect it, record Large/
  sysid tier and refit with `--raw` (`/imu1_raw`, pre-filter) — the
  coefficients should move only marginally.
- **eRPM correctness**: `ω` is only as good as the bidir-DShot
  pole-pair configuration. The `k_w·Σω²_hover ≈ 9.81` check (§4)
  catches a wrong pole count (it scales all speed-dependent
  coefficients by the corresponding power of the scale error).
- **Output-error actuator fit is initial-condition sensitive** on very
  long windows (drift accumulates); crop with `--t0/--t1` to the
  chirp/step section if the pooled fit looks biased.
- **No confidence intervals**: the fits report point estimates only.
  The practical substitute: fit two different flights (or two windows
  of one flight) and compare — parameters that move more than a few
  percent aren't identified by your excitation.

```python
python3 analysis/sysid_mcap.py analysis/datasets/indoor_exp_*/*.mcap \
    --fit thrust --from-mission --no-show --vehicle vehicles/simulation/research_1khz.yaml

python3 analysis/sysid_mcap.py analysis/datasets/indoor_exp_*/*.mcap  --from-mission --until-idle --vehicle vehicles/simulation/research_1khz.yaml

# for drag
python3 analysis/sysid_mcap.py analysis/datasets/indoor_exp_*/*.mcap --fit thrust --from-mission --until-idle --vehicle vehicles/simulation/research_1khz.yaml



```