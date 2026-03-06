# IMU Filtering Architecture

## Hardware context

The IMU can be polled at up to 8 kHz over SPI+DMA.

## Problem with naive decimation

The current `imu_task` applies a Butterworth LPF at the *output* rate, not the
input rate. Samples are discarded before being filtered:

```
8 kHz IMU → discard 79/80 samples → Butterworth (fs=100 Hz) → channel
```

This provides no anti-aliasing. The filter only smooths the already-decimated
output; aliased energy from 4–8 kHz folds into the passband unchecked.

## Correct decimation structure

The LPF must be designed for and run at the input sample rate, with decimation
applied to the filter output:

```
8 kHz IMU → LPF (fs=8 kHz, fc=40 Hz) → pick every 80th output → channel
```

## Gyroscope vs accelerometer requirements

| Signal | Requirement | Rationale |
|--------|-------------|-----------|
| Gyro   | Integrate at full 8 kHz | Every skipped sample is lost integration area → attitude drift |
| Accel  | LPF 20–40 Hz, decimate to 100 Hz | Dominated by vibration; 100 Hz more than sufficient for attitude correction |

## Recommended approach: Mahony / Madgwick AHRS

For navigation attitude estimation, a complementary AHRS filter (Mahony or
Madgwick) is the standard embedded approach:

- Runs on every raw gyro + accel sample at 8 kHz
- Integrates gyro continuously (minimises drift)
- Uses accel at low gain to correct attitude
- Outputs a quaternion — read at any desired nav rate (e.g. 100 Hz)
- No matrix inversion; trivially `no_std`; fits naturally in `cybflight_core`

### Suggested `imu_task` structure

```
loop {
    let reading = imu.read().await;          // every sample at 8 kHz
    ahrs.update(reading.gyro, reading.accel, dt);  // integrate at full rate

    if elapsed >= nav_period {
        let attitude = ahrs.attitude();       // quaternion output
        NAV_CHANNEL.try_send(attitude);
        reset_timer();
    }
}
```

## Butterworth filter validity constraint

`ButterworthFilter::new(cutoff_hz, sampling_hz)` returns `None` when
`cutoff_hz >= sampling_hz / 2` (Nyquist violation). Calling `.unwrap()` on the
result panics at startup. Ensure `cutoff < fs / 2` at all call sites.
