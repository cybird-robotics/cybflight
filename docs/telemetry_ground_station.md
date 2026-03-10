# Telemetry & Ground Station Architecture

Design decisions for offboard telemetry, ground station, and position estimate
uplink. Covers the full data path from STM32H743 to the operator's screen and
back.

## Data Flow Overview

```
                          ┌──────────────────────────────────┐
                          │  Ground Station (Python)         │
                          │                                  │
                          │  Vicon DataStream SDK            │
                          │    → position @ ~200 Hz          │
                          │    → UDP TX to ESP32             │
                          │                                  │
                          │  UDP RX from ESP32               │
                          │    → struct.unpack per msg_id    │
                          │    → rerun.log() (live + record) │
                          └──────────┬───────────────────────┘
                                     │ UDP over WiFi
                                     │ port 4210 (telem down)
                                     │ port 4211 (position up)
                          ┌──────────┴───────────────────────┐
                          │  ESP32 (Rust, bridge only)       │
                          │  WiFi AP: 192.168.4.1            │
                          │                                  │
                          │  UART RX → strip COBS → UDP TX  │
                          │  UDP RX → COBS encode → UART TX │
                          └──────────┬───────────────────────┘
                                     │ UART + COBS framing
                          ┌──────────┴───────────────────────┐
                          │  STM32H743 (cybflight)           │
                          │                                  │
                          │  telemetry_tx_task               │
                          │    subscribes to all channels    │
                          │    → serialize → COBS → UART TX  │
                          │                                  │
                          │  position_rx_task                │
                          │    UART RX → COBS decode         │
                          │    → deserialize                 │
                          │    → publish to VEHICLE_ODOMETRY │
                          └──────────────────────────────────┘
```

## Physical Layer: UART (not CAN)

UART is chosen over CAN for the STM32↔ESP32 link because:

- **Point-to-point**: Only two nodes. CAN's bus arbitration and multi-node
  features are wasted.
- **Bandwidth**: UART at 921600 baud handles all telemetry comfortably. CAN
  classic limits frames to 8 bytes, requiring multi-frame fragmentation for
  every message.
- **Simplicity**: No transceiver IC needed, direct 3.3V TX/RX connection.

SAKURAH743 has five free UARTs; FOXEERH743 has four. One will be dedicated to
the ESP32 bridge.

## Framing: COBS

Consistent Overhead Byte Stuffing (COBS) provides reliable message framing over
the UART byte stream:

- Delimiter: `0x00` marks frame boundaries.
- Overhead: 1 byte per 254 payload bytes (negligible).
- No escaping, constant overhead, simple encode/decode.

On the WiFi/UDP leg, COBS is stripped — UDP datagrams are inherently
message-oriented, so each datagram carries one raw message.

## Wire Format: Fixed-Size Packed Structs

Messages use `#[repr(C, packed)]` binary layout — not postcard, not protobuf.

Rationale:
- **Cross-language**: The ground station is Python. `struct.unpack` decodes
  packed C structs trivially. postcard uses varint encoding with no Python
  decoder; protobuf needs code generation.
- **Zero serialization cost on STM32**: Transmit the struct bytes directly.
- **Bandwidth is not a constraint**: At ~50 KB/s total telemetry throughput,
  the slight size increase vs. varint encoding is irrelevant.

### Frame Layout

```
COBS-encoded frame:
┌──────────┬────────┬─────────────────────────┐
│ msg_id   │ seq    │ packed struct payload    │
│ u8       │ u8     │ fixed size per msg_id    │
└──────────┴────────┴─────────────────────────┘
← delimited by 0x00 →

msg_id:  discriminates message type
seq:     wrapping counter (0–255) for drop detection
payload: repr(C, packed) struct, little-endian, fixed size per type
```

### Message Types

| msg_id | Type | Key Fields |
|--------|------|------------|
| 1 | Imu | timestamp, accel[3], gyro[3], temp |
| 2 | VehicleAttitude | timestamp, quaternion[4] |
| 3 | PowerStatus | timestamp, voltage_mv, current_ma, mah_drawn, cell_count |
| 4 | RcInput | timestamp, channels[16], channel_count |
| 5 | RcLinkStatus | timestamp, rssi_dbm, link_quality, snr, rf_mode |
| 6 | DshotTelemetry | timestamp, motor_values[4] |
| 7 | OcpSolverOutput | timestamp, command[4], iterations, converged, solve_time_us |
| 8 | VehicleOdometry | timestamp, position[3], orientation[4], linear_vel[3], angular_vel[3] |

Message IDs and struct layouts are defined in the shared `cybflight-msgs` crate.

## Shared Crate: `cybflight-msgs`

A new no_std crate at `crates/msgs/` defines all message types used by both
STM32 firmware and ESP32 bridge firmware. This is the single source of truth for
wire format.

```
crates/msgs/
  Cargo.toml          # no_std, no dependencies beyond defmt (optional feature)
  src/lib.rs          # Message structs with #[repr(C, packed)], msg_id consts
```

Both `cybflight` (STM32) and the ESP32 firmware depend on this crate. The
Python ground station uses a corresponding `struct.Struct` format string per
message type, derived from the same definitions.

## ESP32 Bridge

The ESP32 is a minimal, stateless bridge. It does not interpret message
contents.

**WiFi mode**: Access Point (AP). The ground station laptop connects directly
to the ESP32's network. No router or infrastructure needed in the field. Fixed
IP: ESP32 = `192.168.4.1`, ground station gets `192.168.4.2` via DHCP.

For lab use with Vicon on a wired LAN, the ESP32 can be switched to STA mode
to join the same network.

**Firmware**: Rust with `esp-hal` + `embassy`, sharing the `cybflight-msgs`
crate with the STM32 firmware.

**Data path**:
- **Downlink** (telemetry): UART RX → COBS decode → wrap payload in UDP
  datagram → send to ground station on port 4210.
- **Uplink** (position): UDP RX on port 4211 → COBS encode → UART TX.

## Ground Station

A single Python script (~200 lines) handles all ground station functions.

### Dependencies

| Package | Purpose |
|---------|---------|
| `rerun-sdk` | Visualization (3D + 2D time-series), recording |
| `vicon-datastream-sdk` | Position tracking from Vicon system |
| `asyncio` | Non-blocking UDP networking |

### Visualization: rerun

rerun provides both 2D time-series plots and 3D spatial visualization in a
single viewer. It replaces what would otherwise require Grafana (2D) + rviz
(3D) — two separate tools with separate infrastructure.

Capabilities used:
- **3D**: Drone pose (position + orientation), trajectory trace, coordinate
  frames.
- **2D time-series**: Voltage, current, mAh, solver iterations, solve time,
  RC link quality, task timing.
- **Recording**: `.rrd` files for post-flight review with full timeline
  scrubbing.

### Recording Strategy

The rerun viewer runs continuously as a live monitor. File recording is
triggered by arm/disarm events from the telemetry stream:

```python
rr.init("cybflight", spawn=True)  # viewer always on

recording = None

def on_arm():
    global recording
    recording = rr.new_recording("cybflight_flight", make_default=False)
    recording.save(f"flight_{timestamp}.rrd")

def on_disarm():
    global recording
    del recording
    recording = None

# Telemetry loop:
rr.log("power/voltage", rr.Scalar(v))            # always → live viewer
if recording:
    recording.log("power/voltage", rr.Scalar(v))  # only → file when armed
```

### Position Estimate Uplink

The ground station receives Vicon pose data via the DataStream SDK and
forwards it to the drone:

```
Vicon DataStream SDK (callback @ ~200 Hz)
  → pack as VehicleOdometry (msg_id=8)
  → UDP TX to ESP32 (port 4211)
  → ESP32 COBS-encodes, forwards over UART
  → STM32 position_rx_task deserializes
  → publishes to VEHICLE_ODOMETRY channel
```

Latency budget: Vicon processing (~2 ms) + WiFi (~1–3 ms) + UART (~0.5 ms) +
decode (~0.1 ms) ≈ **4–6 ms** end-to-end. Acceptable for position control at
100–200 Hz.

## Alternatives Considered

| Option | Why rejected |
|--------|-------------|
| **postcard serialization** | No Python decoder. Varint encoding is compact but opaque to `struct.unpack`. Only beneficial if ground station is also Rust. |
| **MAVLink** | C-oriented, poor no_std Rust support. Custom messages needed anyway for NMPC solver output and task timing. Ecosystem (QGroundControl) not useful for a custom research controller. |
| **CAN bus** (STM32↔ESP32) | 8-byte frame limit requires fragmentation for every message. Bus arbitration unnecessary for point-to-point. |
| **Grafana + InfluxDB** | Good for 2D telemetry but no 3D visualization. Requires running a database server and web server — unnecessary infrastructure when rerun handles both 2D and 3D in one tool. |
| **protobuf** | Requires code generation. `prost` is std-only. `nanopb` is C. Adds build complexity for no benefit over packed structs at this message volume. |
| **ROS2 / micro-ROS** | Massive dependency. DDS transport adds latency and complexity. The entire system is two nodes — ROS2 solves a problem that doesn't exist here. |
| **TCP** (WiFi leg) | Head-of-line blocking adds unpredictable latency. A dropped telemetry packet is preferable to a delayed one. Position estimates are especially latency-sensitive. |

## Implementation Order

1. **`cybflight-msgs` crate** — extract message types from `msgs.rs` into
   shared crate with `#[repr(C, packed)]` wire types and msg_id constants.
2. **UART telemetry TX task** on STM32 — subscribes to all PubSubChannels,
   packs wire structs, COBS-encodes, DMA writes to UART.
3. **UART position RX task** on STM32 — reads UART, COBS-decodes,
   deserializes `VehicleOdometry`, publishes to channel.
4. **ESP32 bridge firmware** — UART↔UDP forwarding, AP mode WiFi.
5. **Python ground station** — UDP receive, rerun visualization, Vicon
   forwarding.
