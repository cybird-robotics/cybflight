# Executor Priority & Motor Safety Architecture

## Overview

Cybflight uses two Embassy executors at different NVIC priority levels to
guarantee that safety-critical motor output (DShot) runs on time regardless
of other firmware load.

## Executor Layout

```
Priority   Executor / Handler          Tasks
--------   -------------------------   ----------------------------------
P0–P5      HAL-managed interrupts      DMA completion, SPI, I2C, EXTI
P6         InterruptExecutor (CRS)     DShot motor output
Thread     Thread executor (main)      IMU, CRSF/GHST, USB, LED, IWDG feed
```

- **P0–P5**: Hardware interrupt handlers managed by embassy-stm32. These must
  remain highest priority so that DMA completion events can wake the DShot
  task's `join4` await.
- **P6**: The DShot task runs on an `InterruptExecutor` bound to the CRS
  (Clock Recovery System) NVIC slot. CRS is unused by this firmware — it just
  provides an interrupt vector. Any task on this executor preempts the thread
  executor.
- **Thread**: The default `#[embassy_executor::main]` executor. All non-safety
  tasks run here. It only executes when no higher-priority interrupt is pending.

### Why CRS?

Any unused interrupt vector works. CRS was chosen because:
- Not connected to any peripheral on either SAKURAH743 or FOXEERH743.
- Unlikely to be needed in future (CRS is for HSI48 trimming via USB SOF,
  which this firmware doesn't use).
- If a board needs CRS, pick another unused vector (UART7, UART8, LPTIM2, etc.)
  and update the `#[interrupt]` handler in `main.rs`.

## DShot Frame Rate

### Current behavior

DShot runs at approximately **6553 Hz** (152.6 µs per frame). This is determined
by the embassy timer tick resolution, not by executor scheduling.

Embassy's time driver runs at 32,768 Hz (one tick = 30.518 µs). Each DShot
iteration consists of:

| Phase             | Duration    | Ticks |
|-------------------|-------------|-------|
| DMA transfer      | ~30 µs      | 1     |
| `after_micros(95)`| rounds up   | 4     |
| **Total**         | **152.6 µs**| **5** |

`Timer::after_micros(95)` maps to 95 / 30.518 = 3.11 → rounds to 4 ticks.
Combined with 1 tick for the DMA await, each frame takes exactly 5 ticks.

### Is 6553 Hz acceptable?

Yes. DShot600 specifies the **bit rate** (600 kbit/s = 1.67 µs/bit), not the
frame repetition rate. ESCs accept frames at any rate from ~1 Hz to ~32 kHz.
Betaflight defaults to 4 kHz or 8 kHz, matching the PID loop rate. 6553 Hz
exceeds the common 4 kHz default and is well within ESC tolerances.

The frame rate only matters relative to the control loop rate — sending faster
than PID updates is pointless (same command repeated), and sending slower adds
latency. As long as DShot rate >= PID rate, timing is correct.

### How to change it

**Option 1: Adjust the inter-frame delay** (simplest)

In `motors/dshot.rs`, change `Timer::after_micros(95)`:

| Delay value | Ticks (at 32768 Hz) | Total frame period | Effective rate |
|-------------|---------------------|--------------------|----------------|
| 95 µs       | 4                   | 5 ticks = 152.6 µs | 6553 Hz        |
| 65 µs       | 3                   | 4 ticks = 122.1 µs | 8192 Hz        |
| 30 µs       | 1                   | 2 ticks = 61.0 µs  | 16384 Hz       |

**Option 2: Increase embassy tick rate**

In `crates/cybflight/Cargo.toml`, change the tick-hz feature:

```toml
# Current: 32,768 Hz (30.5 µs resolution)
embassy-time = { version = "0.5.0", features = ["tick-hz-32_768"] }

# Alternative: 1 MHz (1 µs resolution)
embassy-time = { version = "0.5.0", features = ["tick-hz-1_000_000"] }
```

Higher tick rates give finer delay control but increase timer interrupt
overhead system-wide. Generally unnecessary for DShot.

**Option 3 (Phase 3): Event-driven, no fixed delay**

When the motor command channel is wired up, replace the fixed timer delay
with a signal wait:

```rust
loop {
    let cmd = MOTOR_CMD.wait().await;
    // encode & DMA
}
```

This locks DShot to the PID loop rate automatically — one frame per control
update, zero wasted frames, minimum latency. This is the intended production
architecture.

### Why we leave it as-is

- 6553 Hz is above typical PID rates (4 kHz) and well within ESC tolerance.
- The fixed delay is a Phase 2 placeholder — it will be replaced by
  event-driven output in Phase 3 when the control loop is wired.
- Changing tick-hz affects all embassy timing system-wide for marginal gain.
- The important problem (192 Hz from executor starvation) is solved.

## IWDG Watchdog

The Independent Watchdog (IWDG1) provides a last-resort system reset if the
firmware hangs completely.

- Clock: LSI (~32 kHz), independent of main oscillator
- Prescaler: /32 → 1 kHz effective tick
- Reload: 500 → ~500 ms timeout
- Fed every 200 ms by `iwdg_feed_task` on the thread executor

If the thread executor stalls for >500 ms (hard fault, deadlock, infinite
loop), the IWDG resets the MCU. ESCs will also independently disarm after
~1 second without valid DShot frames.

IWDG cannot be stopped once started — this is by design.

### Safety layers summary

| Layer              | Mechanism                              | Failure mode covered          |
|--------------------|----------------------------------------|-------------------------------|
| InterruptExecutor  | DShot preempts all async tasks         | Executor starvation           |
| Motor cmd timeout  | DShot sends disarm if control stalls   | Stale throttle (Phase 3)      |
| IWDG               | MCU reset after 500 ms hang            | Hard faults, deadlocks        |
| ESC failsafe       | ESC disarms after ~1 s without DShot   | Total firmware death          |
