# Sensor Bus Sharing & Configuration

## Overview

Multiple sensors can share a single I2C or SPI bus. On a cooperative (non-preemptive)
executor like Embassy, sensor tasks sharing a bus must be carefully scheduled to avoid
starvation. This document covers bus topology, sensor configuration, and the
scheduling strategy.

## Bus Topology

### SAKURAH743

| Bus | Speed | Sensors |
|-----|-------|---------|
| SPI4 | 1 MHz, MODE_3 | ICM42688P (IMU1) |
| SPI1 | 1 MHz, MODE_3 | DPS310 (baro2), IIM42652 (IMU2, future) |
| I2C1 | 400 kHz | ICP20100 (baro1, addr 0x63), IST8310 (magint, addr 0x0E) |
| I2C2 | 400 kHz | QMC5883L (magext, addr 0x0D) |

### FOXEERH743

| Bus | Speed | Sensors |
|-----|-------|---------|
| SPI2 | 1 MHz, MODE_3 | ICM42688P / MPU6000 / MPU6500 (IMU1) |
| I2C1 | 400 kHz | DPS310 (baro1, addr 0x76), QMC5883L (magext, addr 0x0D) |

## Cooperative Scheduling on Shared Buses

Embassy uses a cooperative executor — tasks are never preempted. When two sensor
tasks share an I2C bus via `Mutex<NoopRawMutex, _>`, a task that continuously
re-acquires the mutex in a tight loop will starve other tasks on the same bus.

### The problem

Both ICP20100 and IST8310 on SAKURAH743's I2C1 need to poll for data readiness:
- ICP20100 polls `FIFO_FILL` register until packets are available
- IST8310 polls `STAT1` register until `DRDY` is set

A tight polling loop like this starves the other task:
```rust
// BAD: monopolizes the I2C bus mutex
for _ in 0..200 {
    let status = i2c.write_read(addr, &[REG_STATUS], &mut buf).await?;
    if ready { break; }
    // No yield — immediately re-acquires mutex on next iteration
}
```

### The solution: Timer-based pacing + yield safety net

Two complementary mechanisms prevent bus starvation:

**1. Timer pacing in task loops (primary)**

Each sensor task sleeps for one sensor period between reads. When the task wakes,
data is already ready — the driver's poll loop completes in 1-2 iterations instead
of hundreds.

```rust
// sensors/baro.rs and sensors/mag.rs
let period_ms = (1000.0 / rate_hz) as u64;
loop {
    Timer::after_millis(period_ms).await;  // bus is free during sleep
    match self.sensor.read().await { ... }
}
```

This ensures the I2C bus is idle ~95% of the time. Both tasks only touch it briefly
when data is guaranteed to be ready.

**2. yield_now() in driver poll loops (safety net)**

If data isn't immediately ready (timing jitter, recovery), the driver poll loops
yield between iterations so other tasks can use the bus:

```rust
// In driver read() functions
for _ in 0..200 {
    let status = read_reg(&mut self.i2c, self.addr, REG_STATUS).await?;
    if ready { break; }
    crate::yield_now().await;  // let other tasks run
}
```

`yield_now()` is a core-only async yield (no embassy dependency in the drivers
crate). It returns `Pending` once, wakes itself, and returns `Ready` on the next
poll — giving the executor exactly one chance to switch tasks.

### Why yield_now alone is not enough

With only yield_now (no Timer pacing), both tasks run hot poll loops 100% of the
time. They interleave via yields, but the bus is saturated with poll traffic. In
practice this creates fragile timing where one task can still starve the other,
depending on executor scheduling order and I2C transaction durations.

Timer pacing is the primary mechanism — it matches task wake-ups to the actual
sensor output rate. yield_now is the safety net for edge cases.

## Sensor Configuration Reference

### DPS310 (Barometric Pressure)

Connection: SPI (SAKURAH743 baro2) or I2C (FOXEERH743 baro1).

**Current config: 32 Hz, 16x oversampling (PRS_CFG = 0x54)**

| Rate | Oversampling | PRS_CFG | Noise (Pa) | Alt. noise |
|------|-------------|---------|------------|------------|
| 128 Hz | 1x | `0x70` | ~12 Pa | ~1 m |
| 128 Hz | 2x | `0x71` | ~9 Pa | ~0.75 m |
| 64 Hz | 8x | `0x63` | ~1.5 Pa | ~12 cm |
| **32 Hz** | **16x** | **`0x54`** | **~0.8 Pa** | **~7 cm** |
| 16 Hz | 32x | `0x45` | ~0.5 Pa | ~4 cm |
| 2 Hz | 128x | `0x17` | ~0.2 Pa | ~2 cm |

32 Hz / 16x is the standard flight controller setting (ArduPilot/Betaflight default).
On SPI there is no bus contention — rate is limited only by the noise tradeoff.

### ICP20100 (Barometric Pressure)

Connection: I2C only (SAKURAH743 baro1, addr 0x63).

**Current config: Mode 1, continuous P+T (MODE_SELECT = 0x28)**

| Mode | ODR | Bandwidth | MODE_SELECT | Noise |
|------|-----|-----------|-------------|-------|
| Mode 0 | ~25 Hz | Low | `0x08` | Lowest |
| **Mode 1** | **~120 Hz** | **~30 Hz** | **`0x28`** | **Low** |
| Mode 2 | ~200 Hz | ~50 Hz | `0x48` | Medium |
| Mode 3 | ~400 Hz | ~100 Hz | `0x68` | Higher |
| Mode 4 | ~800 Hz | ~200 Hz | `0x88` | Highest |

Mode 1 provides better noise than DPS310 at higher update rate. Going above Mode 1
adds noise without benefit — the altitude filter doesn't need >120 Hz input.

**ICP20100 quirk**: requires a dummy register read (reg 0x00) after every I2C
transaction. The `read_reg` and `write_reg` helpers handle this automatically.

### IST8310 (Magnetometer)

Connection: I2C only (SAKURAH743 magint, addr 0x0E).

**Current config: single-measurement mode, 16x averaging (AVGCNTL = 0x24)**

| Averaging | AVGCNTL | Meas. time | Max rate | Noise |
|-----------|---------|------------|----------|-------|
| None | `0x00` | ~0.5 ms | ~2000 Hz | Highest |
| 4x | `0x09` | ~2 ms | ~500 Hz | Medium |
| **16x** | **`0x24`** | **~6 ms** | **~166 Hz** | **Lowest** |

IST8310 has no continuous mode — firmware triggers each measurement individually.
The ~100 Hz effective rate comes from the 10ms Timer pacing in the task loop
(slightly conservative vs. the ~6ms measurement time to ensure DRDY is always set).

16x averaging is the right choice for compass heading — accuracy matters far more
than speed. ArduPilot and Betaflight use the same setting.

**IST8310 quirk**: the WHO_AM_I register (0x00) is writable and can be corrupted by
bus noise. The driver resets the sensor before reading WHO_AM_I during probe.

### QMC5883L (Magnetometer)

Connection: I2C (SAKURAH743 I2C2, FOXEERH743 I2C1, addr 0x0D).

Continuous mode at 200 Hz, 512x oversampling, ±8 Gauss range. Configuration is
fixed in the driver init — no configurable tradeoffs.

### I2C Bus Bandwidth

At 400 kHz I2C, each small transaction (2-3 bytes) takes ~50-100 µs.

For SAKURAH743 I2C1 (ICP20100 + IST8310):
- ICP20100 read cycle: ~3 transactions × 100 µs = ~300 µs every 8 ms
- IST8310 read cycle: ~3 transactions × 100 µs = ~300 µs every 10 ms
- Combined: ~600 µs per ~8 ms = **~8% bus utilization**

Bus bandwidth is not the constraint. The Timer-based pacing is purely about
cooperative task scheduling.

## Board Init Pattern for Shared Buses

When multiple sensors share a bus, all devices must be initialized before any
sensor tasks are spawned. This prevents a running task from monopolizing the bus
while other devices are still being configured.

```rust
// Init all I2C1 devices first (no tasks running yet)
let mut baro_driver = None;
let mut mag_driver = None;

match Icp20100::new(I2cDevice::new(bus), addr, &mut delay).await {
    Ok(baro) => baro_driver = Some(baro),
    Err(e) => defmt::warn!("ICP20100 init failed: {}", e),
}

match Ist8310::new(I2cDevice::new(bus), addr, &mut delay).await {
    Ok(mag) => mag_driver = Some(mag),
    Err(e) => defmt::warn!("IST8310 init failed: {}", e),
}

// Spawn tasks only after all devices are initialized
if let Some(baro) = baro_driver {
    spawner.spawn(icp20100_baro_task(BaroReader::new(baro), &BARO_1)).unwrap();
}
if let Some(mag) = mag_driver {
    spawner.spawn(ist8310_mag_task(MagReader::new(mag, align))).unwrap();
}
```
