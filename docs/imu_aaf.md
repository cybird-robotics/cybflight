# IMU Hardware Anti-Alias Filter (AAF)

## Theory

The AAF is a continuous-time RC-ladder low-pass filter built into the ICM-426xx
silicon, sitting **between the MEMS sensing element and the ADC**. Its job is
purely analog: attenuate high-frequency energy before sampling so that
frequencies above the Nyquist limit (ODR/2) cannot fold back into the digital
signal. At 8 kHz ODR, anything above 4 kHz that reaches the ADC will alias to a
lower frequency indistinguishable from real signal — the AAF prevents that. No
digital filter applied afterward can undo aliasing.

The filter is a 1st-order IIR in hardware, parameterized by three values that
set its pole location:

| Field      | Description |
|------------|-------------|
| `DELT`     | Primary tuning code (1–63); higher = wider bandwidth |
| `DELTSQR`  | DELT² (with minor rounding), used in the hardware coefficient calculation |
| `BITSHIFT` | Controls binary scaling of the coefficient; decreases as DELT grows |

## Which Datasheet Table Matters

**Section 5.3, page 28** of DS-000347 (ICM-42688-P datasheet). It maps each
(DELT, DELTSQR, BITSHIFT) triplet to a 3 dB bandwidth in Hz, spanning 42 Hz to
3979 Hz in 63 steps. The values are never computed — look up the row closest to
the target frequency and copy the three numbers verbatim.

The UI Filter tables in **section 5.5** are separate: those are *digital*
post-ADC filters with configurable order and bandwidth, applied after sampling.
They reduce noise further but cannot remove aliased energy introduced before the
ADC.

## Code Strategy

Three decisions are made in `icm426xx.rs`:

### 1. Target frequencies (per ODR mode)

The driver takes an `OutputDataRate` argument (selected by the `imu_1khz`
build knob, see `crates/cybflight/src/rates.rs`):

- **8 kHz mode** (default): gyro ~1 kHz — passes all flight-relevant
  dynamics while rejecting high motor harmonics
- **1 kHz low-noise mode**: gyro ~258 Hz — Nyquist drops to 500 Hz, so the
  8 kHz-mode ~1 kHz gyro AAF would sit *above* Nyquist and stop doing its
  job; the narrower pole restores real anti-aliasing
- **Accel ~250 Hz in both modes** — sufficient for attitude/gravity
  correction and already below either Nyquist; narrower bandwidth reduces
  vibration noise in the gravity estimate

In 1 kHz mode the gyro simply reuses the accel triplet — same LUT row.

Related but separate: in 1 kHz mode the driver also programs the *digital*
UI filter (section 5.5) to BW code 1 = ODR/4 ≈ 227 Hz, 2nd order. The
selectable UI bandwidths only take effect at ODR ≤ 1 kHz — at 8 kHz the
hardware pins the UI filter wide open (~2 kHz, "low latency"), which is
why the 8 kHz mode writes `GYRO_ACCEL_CONFIG0 = 0xFF` and the datasheet
calls 1 kHz the best-noise operating point.

### 2. Per-family register values

The AAF pole frequency scales linearly with the chip's internal AAF clock:

| Family                          | Internal clock | Gyro 8 kHz mode | DELT | Gyro 1 kHz mode / Accel | DELT |
|---------------------------------|---------------|-----------------|------|--------------------------|------|
| ICM-42688P / ICM-42622P         | 32 MHz        | ~997 Hz         | 21   | ~258 Hz                  | 6    |
| ICM-42605 / IIM-42652 / IIM-42653 | 8 MHz       | ~995 Hz         | 63   | ~249 Hz                  | 21   |

The 8 MHz family runs 4× slower, so it needs 4× larger DELT values to hit the
same real-world cutoff. The table entry for delt=63 at 32 MHz is 3979 Hz;
scaled to 8 MHz: 3979 × (8/32) ≈ 995 Hz.

### 3. Register encoding

| Signal | Bank | Register | Encoding |
|--------|------|----------|----------|
| Gyro DELT     | 1 | `0x0C` | Raw value |
| Gyro DELTSQR  | 1 | `0x0D` (lo) / `0x0E` (hi nibble) | Split: `[7:0]` in LO, `[11:8]` packed with BITSHIFT in HI |
| Gyro BITSHIFT | 1 | `0x0E` bits `[7:4]` | `(bitshift << 4) \| (deltsqr >> 8)` |
| Accel DELT    | 2 | `0x03` bits `[6:1]` | Written as `delt << 1` (field is not at bit 0) |
| Accel DELTSQR | 2 | `0x04` (lo) / `0x05` (hi nibble) | Same split encoding as gyro |
| Accel BITSHIFT| 2 | `0x05` bits `[7:4]` | Same as gyro |
