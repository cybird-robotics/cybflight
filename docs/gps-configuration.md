# u-blox F9 RTK integration

The u-blox route is implemented in [the GPS driver](../crates/drivers/src/gps/ublox.rs), selected through the same GPS task and estimator used by UM982. The SAKURA example is `sakura_ublox_f9`:

```sh
just print-features sakura_ublox_f9
just build sakura_ublox_f9
```

Its vehicle settings select `pos_source: gps`, `gps_model: ublox`, and `gps_dual_antenna: no`. The u-blox driver is selected when `gps_unicore` is absent; there is no separate `gps_ublox` feature. The example shares the SAKURA airframe parameters and outdoor mission schedule with `sakura_um982`; review physical parameters, tuning, origin, and mission coordinates for the installation.

## Receiver and UART setup

| Connection | Configuration |
|---|---|
| SAKURA USART3, PD8 TX / PD9 RX | 230400 baud, 8N1 for u-blox |
| Receiver navigation port | Saved baud matching SAKURA; UBX input accepted and NAV-PVT output enabled |
| ESP32 RTCM output | Waveshare GPIO14 or XIAO GPIO0; receiver correction-input baud must match `gnss_baud` (115200 by default) |
| Correction source | NTRIP or a serial base receiver routed through `cybgcs` and the ESP32 bridge |

Cross TX/RX and share ground. Configure the F9 receiver before connecting it. The driver sends legacy `UBX-CFG-MSG` to enable NAV-PVT and waits for ACK; it does **not** configure baud, measurement rate, navigation dynamics, constellations, or RTCM input, and it does not save receiver settings. NMEA output is not disabled by the driver.

Legacy command availability depends on the exact F9 model and firmware. Check the applicable vendor interface description and record the receiver firmware with bench results. The [ZED-F9P integration manual](https://content.u-blox.com/sites/default/files/ZED-F9P_IntegrationManual_UBX-18010802.pdf) describes the modern configuration interface. The current driver has no CFG-VALSET or MON-VER negotiation path; this build example does not establish compatibility with every F9 variant or firmware.

## Navigation behavior

The parser consumes 92-byte NAV-PVT messages, including position, velocity, accuracy estimates, differential status, and carrier-solution status. It feeds the common GPS health, telemetry, and ESKF path. The estimator requires an **RTK-fixed** position (`carr_soln >= 2`) to initialize, including when a fixed navigation origin is baked into the vehicle YAML. A standalone or float fix cannot initialize this outdoor configuration.

The u-blox path produces position fixes only. It does not decode RELPOSNED or emit dual-antenna heading events. Dual-antenna heading fusion currently requires the UM982 integration. Standalone M8/M10 receivers are not substitutes for the RTK configuration described here; the former M10 CFG-VALSET setup notes did not describe the current driver.

## Validation and remaining work

CI builds the SAKURA/F9 example and enforces agreement between vehicle YAML and Cargo features. Compilation does not validate the receiver firmware, saved settings, correction stream, or flight behavior.

The current driver has two known hardening gaps:

- Incoming NAV-PVT and ACK checksum bytes are consumed without validation. Checksum verification and corruption/recovery tests are needed before treating transport integrity as verified.
- Startup retries count complete UBX frames rather than elapsed time. A silent receiver can hold the first scan until the board's outer 30-second initialization timeout; the nominal five attempts do not guarantee timed retransmission. Silence, delayed startup, ACK/NAK, and recovery behavior need tests.

Record the F9 model and firmware, both UART settings, NAV-PVT rate, RTK correction source, time to RTK-fixed, and behavior after correction loss or receiver restart. Preserve the RTK initialization and readiness requirements while validating this route.
