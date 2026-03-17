# Time Synchronization

This document describes the time synchronization architecture across the
cybflight ecosystem: how VICON motion capture and IMU data are aligned to a
common UTC timebase for ESKF sensor fusion and telemetry.

## Problem

VICON and IMU run on different devices with unrelated clocks. Without
synchronization there is no way to compute VICON measurement age (how stale
a pose is when the ESKF processes it) or correlate telemetry across devices.

## Architecture Overview

```
                    ┌──────────┐
                    │NTP Server│  (WAN or local)
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
         ┌────▼───┐ ┌───▼────┐ ┌──▼───────┐
         │VICON PC│ │ GS PC  │ │ ESP32-C6 │
         │ (NTP)  │ │(chrony)│ │ (SNTP)   │
         └───┬────┘ └───┬────┘ └──┬───────┘
             │          │         │ UART (deterministic)
             │ Ethernet │ WiFi    │
             ▼          ▼         ▼
         ViconSDK ─► vicon-bridge ─► cybesp-bridge ─► STM32H743
                        GS PC          ESP32            cybflight
```

The GS PC acts as the NTP server for the local network. The ESP32 syncs via
SNTP every 3 seconds and forwards time to the STM32 using PTP-style
follow-up messages over the deterministic UART link.

## Latency Path Decomposition

Every VICON pose frame traverses three segments before reaching the ESKF:

```
VICON cameras ──[A]──► GS PC ──[B]──► ESP32 ──[C]──► STM32
```

| Segment | What | Measurement |
|---------|------|-------------|
| **[A] VICON pipeline** | Camera capture → SDK delivery | Per-frame: `GetLatencyTotal()` from ViconDataStreamSDK |
| **[B] GS processing** | SDK return → UDP send | Per-frame: `gs_send_time - (gs_recv_time - vicon_latency)` |
| **[C] Transport** | UDP → WiFi → UART → STM32 RX | Per-frame: `to_utc(rx_instant) - gs_send_time` via clock sync |

All three segments are measured per-frame and embedded in the wire message,
giving the ESKF a precise `measurement_age` for each VICON update.

## Clock Domains

| Device | Clock | Precision | Notes |
|--------|-------|-----------|-------|
| VICON cameras | Internal 135 MHz | Sub-µs between cameras | No wall-clock; SDK reports latency, not timestamps |
| VICON PC | NTP-synced system clock | ~0.1–1 ms to NTP | Provides reference for VICON capture time |
| GS PC | chrony (NTP server) | ~0.1–1 ms to NTP | **Time authority** — serves NTP to ESP32 |
| ESP32-C6 | SNTP to GS PC | ~0.1–1 ms to GS PC | Extrapolates between 3s SNTP polls |
| STM32H743 | Embassy `Instant` (HSE crystal) | ~50 ppm drift | Monotonic internally; UTC via PTP follow-up |

## Internal vs External Time

**Rule: all STM32 internal timestamps are monotonic `Instant`.**

The Mahony filter, ESKF, control loops, and dt calculations never see UTC.
UTC conversion happens only at the telemetry serialization boundary
(`esp_bridge_tx_task`) via `time_sync::to_utc_us(instant)`.

For VICON measurement age, the ESKF computes:
```
age = to_utc_us(Instant::now()) - msg.capture_time_utc_us
```

## PTP Follow-Up Protocol

The ESP32 sends `WireTimeSync` (msg_id 129) frames to the STM32 every 3
seconds. Each frame carries two NTP timestamps:

| Field | Source | Purpose |
|-------|--------|---------|
| `esp_send_ntp_us` | `get_ntp_time_us()` before `write_all()` | Primary offset estimate |
| `prev_tx_complete_ntp_us` | `get_ntp_time_us()` after previous `write_all()` returned | Refined PTP follow-up |

### Primary Offset (first sync or fallback)

```
offset = esp_send_ntp_us - rx_instant_us + UART_FRAME_DELAY_US
```

`UART_FRAME_DELAY_US ≈ 217 µs` (20 COBS bytes × 10 bits / 921600 baud).
Error: ESP32 task scheduling jitter between `now()` and DMA start (~µs
normally, ~ms if WiFi preempts).

### Refined Offset (PTP follow-up, steady state)

```
offset = prev_tx_complete_ntp_us - prev_rx_instant_us
```

After `write_all()` returns, the ESP32 captures NTP time. At that moment
the last byte has left the UART shift register, and the STM32's idle-line
detector fires almost immediately. So `tx_complete ≈ rx_instant` in the
shared timebase — **no UART delay estimation needed.**

This is sent in the *next* frame, and the STM32 uses the stored
`prev_rx_instant` for the calculation. Accuracy: ~µs (ISR latency only).

### Smoothing and Outlier Rejection

- EMA with α = 0.1 (26/256 in fixed-point)
- Reject any sample where |offset - smoothed| > 10 ms
- First sample accepted directly (no smoothing)

## Wire Protocol

### Uplink Messages (→ STM32)

| Message | msg_id | Size | Fields |
|---------|--------|------|--------|
| `WirePose` | 128 | 44 B | `capture_time_us: i64`, `gs_send_time_us: i64`, position, orientation |
| `WireTimeSync` | 129 | 16 B | `esp_send_ntp_us: i64`, `prev_tx_complete_ntp_us: i64` |
| `WirePingResp` | 130 | 28 B | `ping_id: u32`, `stm32_send_us: u64`, `gs_recv_time_us: i64`, `gs_send_time_us: i64` |

### Downlink Messages (→ GS)

| Message | msg_id | Size | Fields |
|---------|--------|------|--------|
| `WirePing` | 16 | 12 B | `ping_id: u32`, `stm32_send_us: u64` |
| `WireTimeSyncStatus` | 17 | 25 B | `synced: u8`, `offset_us: i64`, `ping_rtt_us: u64`, `ping_clock_err_us: i64` |

All existing downlink telemetry messages (IMU, attitude, RC, DShot, etc.)
retain their `timestamp_us: u64` field, now populated with UTC microseconds
via `time_sync::to_utc_us()` instead of boot-relative `Instant::as_micros()`.

When time sync is not established (e.g., debugging without a GS), timestamps
fall back to boot-relative microseconds so telemetry still works.

## Ping RTT Cross-Validation

The STM32 sends `WirePing` every 5 seconds (batched in the 100 Hz TX task).
The GS PC echoes it as `WirePingResp` with its own UTC timestamps. The
STM32 computes:

```
RTT = rx_instant - stm32_send_us
clock_error = gs_send_time - (to_utc(rx_instant) - RTT/2)
```

This provides an independent check that the PTP-derived UTC offset is
consistent with the actual network delay. If `clock_error` drifts beyond a
threshold, the time sync may be degraded (e.g., ESP32 lost WiFi and SNTP
is stale).

## Shell Commands

| Command | Description |
|---------|-------------|
| `timesync` | One-shot: print current sync status (synced, offset, ping RTT, clock error) |
| `stream timesync on` | Start 1 Hz time sync status stream |
| `stream timesync off` | Stop time sync status stream |

## Graceful Degradation

When the time sync chain is not running (e.g., debugging without GS PC or
ESP32 WiFi), the system degrades gracefully:

- All internal control (Mahony, ESKF, attitude control) runs unchanged on
  monotonic `Instant` — **zero impact from missing time sync**
- `to_utc_us()` falls back to boot-relative microseconds, so telemetry
  timestamps are still monotonic and usable (just not UTC-aligned)
- `WirePose` fields `capture_time_utc_us` and `gs_send_time_utc_us` are `0`
  when vicon-bridge is not running — consumers can detect this
- The `timesync` shell command shows `synced=false` when no sync is active

## NTP Server Setup

The GS PC runs chrony in server mode. See `vicon-bridge/README.md` for
configuration. Key points:

- `allow 192.168.50.0/24` — serve to UTADR network
- `local stratum 1` — serve time even without upstream NTP
- ESP32 SNTP queries `192.168.50.101:123` (GS PC IP)

## File Map

| File | Role |
|------|------|
| `cybflight-msgs/src/wire.rs` | Wire types: WirePose, WireTimeSync, WirePing, WirePingResp, WireTimeSyncStatus |
| `cybflight-msgs/src/lib.rs` | ViconPose channel type with UTC timing fields |
| `cybflight/src/comm/time_sync.rs` | UTC offset computation, EMA smoothing, to_utc_us(), status() |
| `cybflight/src/comm/esp_bridge.rs` | RX: dispatches TIME_SYNC/PING_RESP; TX: UTC conversion + ping + status telem |
| `cybflight/src/usb_serial.rs` | Shell commands: `timesync`, `stream timesync on/off` |
| `cybesp-bridge/src/bin/main.rs` | SNTP client, NTP state, PTP follow-up in uplink_task |
| `vicon-bridge/vicon-bridge/src/main.rs` | SystemTime timestamps, GetLatencyTotal, ping responder |
| `vicon-bridge/README.md` | NTP server (chrony) setup guide |

## Why Not Sync STM32 Directly to NTP?

The STM32 has no network stack — it communicates via UART to the ESP32.
Relaying raw NTP packets through the UART would introduce asymmetric delay
that the NTP algorithm can't account for. Instead, the ESP32 handles NTP
complexity (outlier rejection, clock discipline) and forwards the result
over the deterministic UART link where delay is constant and known.

## Why Not Just Use Ping RTT/2 for Everything?

Ping RTT/2 is an *average* across a sliding window — it cannot capture
per-frame WiFi jitter. The clock sync gives per-frame transport delay:
`to_utc(rx_instant) - gs_send_time`. The ping is kept as a health check
and fallback, not as the primary delay measurement.
