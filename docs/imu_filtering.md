# IMU Filtering Architecture

## The chain

There is no decimation anywhere in the IMU path. `ImuReader` filters at the
IMU's own sample rate and publishes **every** sample to `IMU_1`; the INDI
inner loop runs at that same rate (`rates::IMU_ODR_HZ`), and the ESKF
propagates on every message too. So the software biquads are true
sample-rate filters, not post-decimation smoothers.

```
MEMS → analog AAF → ADC → digital UI filter → SPI/DMA
     → ImuReader: align → publish IMU_1_RAW → 2nd-order biquad → publish IMU_1
     → INDI (RPM notches → control law) / ESKF
```

`IMU_1_RAW` is the pre-biquad mirror, published for blackbox sysid — RPM-notch
fits and vibration spectra need pre-filter samples.

## What the hardware already does

Per `docs/imu_aaf.md`, the ICM426xx contributes very different filtering in the
two modes the `imu_rate` build knob selects:

| Mode | Gyro AAF | UI filter | Accel AAF |
|---|---|---|---|
| 8 kHz (default) | ~997 Hz, 1st order | pinned wide open (~2 kHz) | ~250 Hz, 1st order |
| 1 kHz (`imu_1khz`) | ~258 Hz, 1st order | 227 Hz, 2nd order | ~249 Hz, 1st order |

The BMI270 board (micoair743v2, 3.2 kHz) has a ~751 Hz normal-mode gyro filter
and a 1600 Hz OSR4 accel.

## Why the software filter exists

At 8 kHz it is the only software band-limiter in the loop, and it is
load-bearing for one specific path: `rate_err = rate_sp − gyro` in
`IndiController::step` feeds the gyro into `dv` with gain `indi_rate_*` and
no filtering of its own. Every other INDI signal (`spf_fs`, `rate_dot_fs`,
`u_state_fs`, `omega_fs`) goes through the `indi_sync_hz` biquad; this one
does not. The RPM notch bank would normally cover the motor harmonics there,
but `rpm_notch_en` is currently 0 on every vehicle.

Measured end-to-end (hardware chain × software biquad, RBJ lowpass Q=1/√2,
prewarped — so a cutoff in Hz means the same thing at every sample rate):

| chain | ENBW | delay @10 Hz | \|H\| @400 Hz |
|---|---|---|---|
| gyro 8 kHz, hw only | 1322 Hz | 0.16 ms | −0.7 dB |
| gyro 8 kHz + sw 200 Hz | 215 Hz | 1.28 ms | −13.1 dB |
| gyro 1 kHz, hw only | 186 Hz | 1.61 ms | −15.6 dB |
| gyro 1 kHz + sw 200 Hz | 151 Hz | 2.58 ms | −40.7 dB |
| gyro 1 kHz + sw 380 Hz | 183 Hz | 1.89 ms | −24.1 dB |
| accel 8 kHz, hw only | 377 Hz | 0.64 ms | −5.5 dB |
| accel 8 kHz + sw 80 Hz | 83 Hz | 3.46 ms | −33.6 dB |
| accel 1 kHz + sw 80 Hz | 80 Hz | 4.40 ms | −59.0 dB |

The consequence: **the same cutoff is not the same decision at both rates.**
At 8 kHz the software filter does all the shaping. At 1 kHz the hardware
already delivers roughly what the 8 kHz build gets *after* its software
filter, so a 200 Hz software cutoff there is mostly added delay — which is
why `vehicles/sakura_bench.yaml` (the one `imu_rate: 1khz` vehicle) pins the
gyro at 380 Hz instead of inheriting the default.

Accel is the easier call: `spf_fs` sits behind a 12 Hz `indi_sync_hz` filter
(18.8 ms), so 40 vs 80 vs 150 Hz is invisible to INDI's thrust increment. The
cutoff is chosen for the ESKF, where band-limiting reduces the vibration
amplitude entering the nonlinear body→world rotation.

## Nyquist constraint

`ImuReader::new` clamps both cutoffs through `clamp_imu_cutoff_hz` (0.4·fs,
shared with `cybflight_core::indi::clamp_cutoff_hz`) and logs when it binds.
This is not decoration: `imu_accel_lpf_hz` / `imu_gyro_lpf_hz` are
schema-bounded to 2000 Hz and reboot-flagged, so without the clamp a single
`param set` + `save` of, say, 600 Hz on an `imu_1khz` build would panic the
biquad builder at startup — a flash-persistent boot loop.

Effective ceilings after the clamp: 3200 Hz at 8 kHz ODR, 1280 Hz on the
3.2 kHz BMI270 board, **400 Hz on an `imu_1khz` build**.

## Not covered by the sim

`cybflight_sim` models IMU white noise and bias but neither the `ImuReader`
biquads nor vibration, so `just sim-check` cannot see a cutoff change. Tuning
these is a flight-test decision, informed by the `IMU_1_RAW` blackbox
spectrum — in particular by where hover 1P actually lands relative to
`imu_gyro_lpf_hz` and `rpm_notch_min_hz`.
