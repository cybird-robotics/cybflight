# RC Telemetry Integration

Telemetry is sent back to the RC transmitter in the inter-frame gap after each
received RC channel packet. Each protocol has its own set of supported telemetry
frame types. This document describes the current state and the planned
integration once additional sensor subsystems come online.

## How Telemetry Works

Both CRSF and GHST are half-duplex protocols. The receiver sends frames on a
fixed schedule (~4ms cycle for CRSF, similar for GHST). After receiving an RC
frame the flight controller has a brief window to send one telemetry frame back
before the next receiver packet arrives.

The runner tasks (`CrsfRunner`, `GhstRunner`) use a round-robin index to cycle
through telemetry slots, sending one frame type per received RC packet.

## Current State

### CRSF

| Slot | Frame | Data Source |
|------|-------|-------------|
| 0 | Heartbeat | Fixed (always sent) |
| 1 | Attitude (pitch/roll/yaw) | `VEHICLE_ATTITUDE` channel |
| 2 | Flight mode | Hardcoded `"ACRO"` |

### GHST

| Slot | Frame | Data Source |
|------|-------|-------------|
| 0 | Pack (battery) | Placeholder zeros |
| 1 | Magbaro (yaw/alt/vario) | Placeholder zeros |

## Planned Integration

Once the corresponding sensor/subsystem channels exist, each telemetry slot
should subscribe to the relevant `PubSubChannel` and send real data. The
round-robin schedule should also be expanded to include the new frame types.

### CRSF Target Schedule

| Slot | Frame | Required Channel |
|------|-------|------------------|
| 0 | Heartbeat | None (always sent) |
| 1 | Battery (`write_battery`) | `BATTERY` |
| 2 | Attitude (`write_attitude`) | `VEHICLE_ATTITUDE` |
| 3 | Flight mode (`write_flight_mode`) | Control mode state |
| 4 | GPS (`write_gps`) | `GPS` |
| 5 | Baro altitude + vario (`write_baro_altitude`) | `BARO` |

The driver also supports `write_vario()` and `write_device_info()` which can be
added to the schedule if the transmitter/receiver benefits from them.

### GHST Target Schedule

| Slot | Frame | Required Channel |
|------|-------|------------------|
| 0 | Pack (`write_pack`) | `BATTERY` |
| 1 | GPS primary (`write_gps_primary`) | `GPS` |
| 2 | GPS secondary (`write_gps_secondary`) | `GPS` |
| 3 | Magbaro (`write_magbaro`) | `VEHICLE_ATTITUDE` + `BARO` |

Note: GHST has no dedicated attitude frame. Yaw (heading) is sent via the
magbaro frame; pitch and roll are not available in the GHST protocol.

## Integration Steps

When adding a new telemetry data source:

1. Define the message type in `msgs.rs` (e.g. `BatteryStatus`).
2. Add a `PubSubChannel` in `sensors/mod.rs` (e.g. `BATTERY`).
3. In the runner's `send_telemetry()`:
   - Create a subscriber for the new channel.
   - Add a new round-robin slot that reads from the subscriber via
     `try_next_message_pure()` and calls the corresponding driver write method.
   - Update the modulo divisor to match the new slot count.
4. If the data source doesn't exist yet (e.g. no GPS module connected), the
   slot can either be skipped or send placeholder values — the transmitter
   handles missing telemetry gracefully.

## Driver Capabilities Reference

Available telemetry write methods in the drivers:

**CRSF** (`cybflight_drivers::rc::crsf::Crsf`):
- `write_heartbeat()`
- `write_battery(voltage_mv, current_10ma, mah_drawn, remaining_pct)`
- `write_attitude(pitch, roll, yaw)` — radians
- `write_flight_mode(mode: &[u8])` — null-terminated ASCII
- `write_gps(lat, lon, speed_kmh10, heading_deg100, alt_m_offset, num_sat)`
- `write_baro_altitude(alt_packed, vario_packed)`
- `write_vario(vspeed_cm_s)`
- `write_device_info(...)`

**GHST** (`cybflight_drivers::rc::ghst::Ghst`):
- `write_pack(voltage_10mv, current_10ma, mah_10, armed)`
- `write_gps_primary(lat, lon, alt_m)`
- `write_gps_secondary(speed, course, sats, dist_home, dir_home, flags)`
- `write_magbaro(yaw_decideg, alt_m, vario_cm_s, flags)`
