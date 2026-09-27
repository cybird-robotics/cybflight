# Hardware support

The reference configurations select SAKURAH7, CRSF, the primary ICM42688P IMU, the INDI inner loop, and the MPC outer loop. `sakura_vicon` uses motion capture; `sakura_um982` selects the Unicore receiver and a single antenna. A dual-antenna build exists, but requires its own installation and validation before using heading fusion.

| Implementation | Release status |
|---|---|
| SAKURAH7 / `sakurah743` | Reference flight controller |
| Waveshare ESP32-C6-Zero, Seeed XIAO ESP32-C6 | Reference bridges; separate build features and pinouts |
| Vicon through `cybgcs` | Indoor reference positioning |
| UM982 BESTNAV with RTCM corrections | Outdoor reference positioning |
| u-blox F9 RTK / `sakura_ublox_f9` | Integrated outdoor route with CI build coverage; verify receiver model, firmware, and saved configuration on hardware |
| FOXEERH743, MICOAIR743V2 | Experimental BSPs; not recently flight-validated |
| Secondary IMU, barometers, magnetometers | Retained drivers; disabled in the SAKURA reference setup |
| GHST, alternate controllers/IMU rates | Experimental paths requiring integration validation |

Experimental source is retained where it is isolated and useful. Its existence does not establish working integration. To maintain another board, start from `experimental/vehicles`, create a configuration under `vehicles`, and document wiring, clocks, sensor orientation, actuator mapping, and repeatable bench/flight results.

## Wiring

Use 3.3 V UART signaling and a shared ground. Cross TX and RX.

| SAKURA function | Pins | Rate |
|---|---|---|
| ESP32 UART link, USART1 | PA9 TX, PA10 RX | 921600, 8N1 |
| GNSS receiver, USART3 | PD8 TX, PD9 RX | 115200 for UM982; 230400 for u-blox F9; 8N1 |
| CRSF, UART4 | PB9 TX, PB8 RX | Defined by CRSF driver |

ESP32 GPIO16 TX connects to SAKURA PA10 RX; GPIO17 RX connects to PA9 TX. RTCM output uses GPIO14 on Waveshare and GPIO0 on XIAO. Connect it to the receiver port configured to accept RTCM at the bridge's `gnss_baud` (115200 by default). XIAO GPIO14 controls its antenna switch; it is not the correction output. The receiver's output and correction-input port configuration must match the physical installation.

Configure and save UM982 BESTNAVB output before use. Enable UNIHEADINGB only for an appropriate dual-antenna installation. The driver reads receiver output; it does not issue the receiver setup commands. Corrections may come from NTRIP or a serial base source supported by `cybgcs`.

The F9 path uses the same GPS task and estimator as UM982, with a separate u-blox parser and startup handshake. It consumes NAV-PVT and does not provide dual-antenna heading. Configure the receiver before connecting it; the driver does not set baud rate, navigation rate, dynamics, or RTCM input. See [u-blox F9 configuration](gps-configuration.md) for the supported interface and remaining driver limitations.

## Position frames

The outdoor estimator waits for an RTK-fixed position before initializing. It uses the vehicle YAML's optional fixed origin or the first RTK-fixed sample. The reference firmware does not accept a ground-station origin command. Coordinates displayed by `cybgcs` for the RTK base are diagnostics, not confirmation of the aircraft's navigation origin.

Vicon routing must match the rigid-body name, roster entry, bridge address, and firmware `airframe.name`. Check metres, quaternion convention, timestamps, and axis mapping before enabling position control. Fleet-state telemetry is diagnostic scaffolding; it does not implement formation or collision-avoidance control.

## Bring-up and release evidence

1. With propellers removed, verify board identity, clocks, IMU axes, RC channels, arming/disarming, motor order and direction, and parameter persistence.
2. Configure bridge WiFi over USB; reserve its DHCP address and update the GCS roster. Confirm telemetry and time synchronization in both directions.
3. Check the selected positioning route, loss of fixes/poses, stale data, and reconnection behavior. Outdoor operation must reach RTK-fixed with corrections flowing.
4. Check loop timing, memory headroom, logging, failsafe behavior, and resets on the exact build.
5. Record board revisions, antenna configuration, airframe, component revisions, and results of a controlled flight before declaring a release hardware-validated.

Parameter blob version 56 resets saved version-55 layouts to the baked defaults. Export and review settings before updating an existing airframe; reapply only settings appropriate to the new reference configuration.
