# Dual IMU Fusion Design

Status: **Draft / Discussion needed**

## Background

SAKURAH743 has two IMUs (ICM42688P + IIM42652). Currently only IMU1 is
initialized; IMU2 is skipped with a TODO. This document captures the design
for fusing both IMUs before implementation begins.

## How Betaflight Does It

Betaflight's dual-gyro handling is simple:

- Config option `gyro_to_use`: `FIRST`, `SECOND`, or `BOTH`.
- When `BOTH`: straight 50/50 arithmetic average of both gyros' filtered outputs.
- Each gyro runs its own independent filter chain (lowpass, notch, dynamic notch)
  before averaging.
- Accelerometer is always taken from gyro 1 only.
- Both gyros are read on the same scheduler tick (synchronous sampling).
- No weighting, Kalman filtering, or outlier rejection.

The benefit is purely noise reduction from averaging co-located sensors
(~sqrt(2) improvement in noise floor).

## Proposed Channel Layout

```
IMU1 reader --> RAW_IMU1 --\
                             --> imu_fusion_task --> FUSED_IMU --> attitude, PID, NMPC
IMU2 reader --> RAW_IMU2 --/

(single-IMU boards: reader --> FUSED_IMU directly, no fusion task)
```

### Channels

| Channel     | Publishers | Subscribers | Purpose |
|-------------|-----------|-------------|---------|
| `RAW_IMU1`  | 1 (reader task) | 1 (fusion) + diagnostics | Per-sensor filtered output |
| `RAW_IMU2`  | 1 (reader task) | 1 (fusion) + diagnostics | Per-sensor filtered output |
| `FUSED_IMU` | 1 (fusion or passthrough) | attitude, PID, NMPC, telemetry | Board-agnostic fused stream |

On single-IMU boards (e.g. FOXEERH743), the reader publishes directly to
`FUSED_IMU`. `RAW_IMU1`, `RAW_IMU2`, and the fusion task are never referenced
and get stripped by the linker.

### Why separate raw channels instead of one tagged channel

- Clean ownership: each reader has its own publisher, no index tagging needed.
- The fusion task knows exactly which source each sample came from.
- Diagnostics can subscribe to a specific IMU without filtering.

## Fusion Task Sketch

```rust
#[embassy_executor::task]
async fn imu_fusion_task() {
    let mut sub1 = RAW_IMU1.subscriber().unwrap();
    let mut sub2 = RAW_IMU2.subscriber().unwrap();
    let pub_fused = FUSED_IMU.immediate_publisher();

    loop {
        // Wait for IMU1 (primary), grab latest from IMU2
        let s1 = sub1.next_message_pure().await;
        let s2 = sub2.try_next_message_pure(); // latest, non-blocking

        let fused = match s2 {
            Some(s2) => msgs::Imu {
                gyro_rad_s: average(s1.gyro_rad_s, s2.gyro_rad_s),
                accel_m_s2: s1.accel_m_s2,  // accel from IMU1 only
                temp_c: s1.temp_c,
                timestamp: s1.timestamp,
            },
            None => s1,  // IMU2 not ready yet, pass through IMU1
        };

        pub_fused.publish_immediate(fused);
    }
}
```

## Open Questions

1. **Synchronization strategy**: Wait for both readings (blocking), or use
   latest-available from IMU2 (non-blocking)? Blocking is cleaner but adds
   latency if one IMU is slightly slower. Non-blocking is what the sketch above
   does, matching Betaflight's approach.

2. **Accelerometer**: Betaflight only uses accel from gyro 1. Should we average
   both accels, or is there a reason to prefer one?

3. **Failure handling**: If IMU2 fails or returns errors, should the fusion task
   fall back to IMU1-only permanently, or keep retrying? Betaflight doesn't
   handle this — it just averages whatever it gets.

4. **Configurable weights**: Start with 50/50 like Betaflight, but should we
   support configurable weights (e.g. for sensors with different noise
   characteristics)?

5. **Filter placement**: Currently each `ImuReader` runs its own Butterworth
   filter. With fusion, should filtering happen before fusion (current design,
   matches Betaflight), after fusion, or both?

6. **Diagnostics**: Should we expose per-IMU data over telemetry/shell for
   debugging? The separate raw channels make this easy but we'd need subscriber
   slots.
