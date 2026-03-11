# IMU Temperature Compensation

## Why It Matters

MEMS gyroscopes and accelerometers have temperature-dependent bias and
scale factor drift. This is the dominant error source in consumer IMUs
like the ICM42688P, IIM42652, and MPU6000/6500. Uncorrected, gyro bias
can drift 1-5 deg/s and accel bias 20-80 mg across a typical operating
temperature range (0-60°C).

On a drone, the board temperature can rise 20-30°C in the first few
minutes of flight (motor heat, voltage regulators). Without compensation,
this causes progressive attitude estimation drift.

## The Compensation Model

### Polynomial Bias Correction

The standard model used by ArduPilot (`INS_TCAL`), PX4 (`TC_*`), and
industrial INS systems is a per-axis polynomial of temperature:

```
bias(T) = c0 + c1*(T - Tref) + c2*(T - Tref)^2 + c3*(T - Tref)^3
```

Where:
- `T` = current IMU temperature reading (°C)
- `Tref` = reference temperature at which the polynomial is centered,
  typically 25°C or the midpoint of the calibration range
- `c0..c3` = calibration coefficients fit from sweep data

A 3rd-order polynomial is standard. Higher orders risk overfitting;
lower orders miss inflection points common in MEMS thermal behavior.

### What Gets Corrected

Each IMU has **12 polynomial coefficient sets** (3 axes × {gyro, accel} × {bias}):

| Sensor | Axes | Coefficients | Total floats |
|---|---|---|---|
| Gyro bias | X, Y, Z | c0, c1, c2, c3 each | 12 |
| Accel bias | X, Y, Z | c0, c1, c2, c3 each | 12 |

Scale factor compensation (an additional multiplicative correction) is
possible but usually secondary — bias drift dominates in practice.

### Runtime Application

Applied per sample, before any filtering or EKF ingestion:

```
gyro_corrected[axis] = gyro_raw[axis] - gyro_bias_poly(temp_c, axis)
accel_corrected[axis] = accel_raw[axis] - accel_bias_poly(temp_c, axis)
```

Cost: ~24 multiply-adds per IMU sample. Negligible at 8 kHz.

## Calibration Procedure

### Equipment

- The board under test
- A way to vary temperature: cold soak (freezer, outdoors in winter) then
  passive warm-up, or a temperature chamber for controlled sweeps
- Logging capability (shell `stream imu` or flash logging)
- A **perfectly stationary** mounting — any vibration corrupts the data

### Steps

1. **Cold soak**: bring the board to the low end of your expected operating
   range (0-10°C). Freezer works — seal in a bag to avoid condensation.

2. **Mount stationary**: place the board on a stable surface, level,
   with no vibration sources nearby. Orientation must remain constant
   throughout the sweep.

3. **Log continuously**: record raw gyro (rad/s), raw accel (m/s²), and
   temperature (°C) as the board warms to ambient and beyond. A full
   sweep from cold to hot takes 20-60 minutes depending on thermal mass.
   Logging at 100 Hz is sufficient (temperature changes slowly).

4. **Optional: heat phase**: after reaching ambient, a heat source
   (hair dryer at distance, warm enclosure, or just running the motors
   at low throttle) extends the calibration range to in-flight temperatures.

5. **Extract bias vs temperature**: at each temperature point, the mean
   of the stationary readings is the bias. Window-average over ~1s blocks,
   paired with the mean temperature of each block.

6. **Fit polynomials**: least-squares fit of 3rd-order polynomial to
   `bias[axis]` vs `temperature` for each of the 6 channels (3 gyro + 3 accel).
   Any curve-fitting tool works (numpy polyfit, MATLAB, etc.).

7. **Store coefficients**: the resulting 24 floats per IMU are stored as
   calibration constants (flash, const array, or runtime parameter).

### Validation

After applying the correction, repeat the temperature sweep and verify:
- Gyro residual bias < 0.1 deg/s across the full range
- Accel residual bias < 5 mg across the full range
- No polynomial divergence at the edges of the calibration range

## Relationship to EKF Bias Estimation

A well-formulated navigation EKF includes gyro bias (and often accel bias)
as state variables. These serve a different purpose than temperature
compensation, and **both should be used together**.

### What Each Handles

| Error source | Temp comp | EKF bias state |
|---|---|---|
| Deterministic thermal drift | Yes | Poorly — lags behind |
| Turn-on bias (varies per power cycle) | No | Yes |
| Aging / long-term drift | No | Yes |
| Calibration imperfection | No | Yes |
| Vibration-induced bias | No | Partially |

### Why the EKF Alone Is Not Enough

The EKF bias state is modeled as a **random walk** (or first-order Markov
process) with low process noise. This assumes bias changes slowly and
smoothly. Temperature-induced drift violates this assumption:

- A 20°C warm-up over 5 minutes causes rapid, deterministic bias change
- The EKF must choose between low process noise (stable but lags thermal
  drift) and high process noise (tracks drift but absorbs real rotation
  into the bias estimate, degrading attitude accuracy)
- During aggressive maneuvers the observability of bias decreases, so the
  EKF cannot distinguish dynamics from drift

### Why Temp Comp Alone Is Not Enough

- Turn-on bias varies each power cycle (MEMS mechanical settling)
- The polynomial fit is imperfect — residual errors of 0.05-0.2 deg/s
  are typical even with good calibration
- The IMU ages: bias curves shift over months/years
- Board-level stress (mounting, thermal expansion) adds bias not captured
  in the polynomial

### Combined Pipeline

```
                    deterministic          stochastic
                    known f(T)             estimated online
                        │                      │
gyro_raw ───► [- bias_poly(T)] ───► EKF ───► [- bias_state] ───► navigation
```

Temperature compensation removes the large, predictable component so the
EKF bias state only needs to track a small, slowly-varying residual. This
allows:

- Lower bias process noise (Q) → more stable bias estimate
- Faster convergence after startup
- Better performance during thermal transients (motor spin-up, altitude
  changes, sun exposure)

### Practical Impact

| Configuration | Typical gyro error during 20°C warm-up |
|---|---|
| No compensation, no EKF bias | 1-5 deg/s drift |
| EKF bias only (tuned aggressive) | 0.2-1 deg/s, noisy attitude |
| EKF bias only (tuned conservative) | 0.5-2 deg/s, lags behind |
| Temp comp only | 0.05-0.2 deg/s residual |
| **Temp comp + EKF bias** | **< 0.05 deg/s** |

## Cybflight Implementation Status

Both IMU drivers already read temperature in the burst read:
- `icm426xx.rs`: `ImuReading.temp_c` from registers in the 14-byte burst
- `mpu6x00.rs`: `ImuReading.temp_c` from the 2-byte temp field between
  accel and gyro

Temperature flows through the IMU channel but is not currently used for
compensation. Implementation would require:

1. **Calibration infrastructure**: a shell command or logging mode to
   capture stationary temperature sweep data
2. **Coefficient storage**: 24 floats per IMU, stored in flash or as
   per-board const arrays
3. **Runtime correction**: polynomial evaluation in the sensor task,
   between raw read and channel publish
