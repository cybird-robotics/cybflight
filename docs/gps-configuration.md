# u-blox M10 GPS Configuration Reference

Datasheet: [SAM-M10Q Integration Manual (UBX-22020019)](https://content.u-blox.com/sites/default/files/documents/SAM-M10Q_IntegrationManual_UBX-22020019.pdf)

Protocol reference: [u-blox M10 SPG 5.10 Interface Description (UBX-21035062)](https://content.u-blox.com/sites/default/files/u-blox-M10-SPG-5.10_InterfaceDescription_UBX-21035062.pdf)

Driver: `crates/drivers/src/gps/ublox_m10.rs`

## Current Configuration

Our driver sends these settings via CFG-VALSET on init:

| Parameter | Value | Effect |
|---|---|---|
| `CFG_UART1OUTPROT_NMEA` | 0 (off) | Disable NMEA output |
| `CFG_UART1OUTPROT_UBX` | 1 (on) | Enable UBX binary output |
| `CFG_MSGOUT_UBX_NAV_PVT_UART1` | 1 | NAV-PVT every measurement cycle |
| `CFG_RATE_MEAS` | 200 ms | 5 Hz navigation rate |
| `CFG_NAVSPG_DYNMODEL` | 8 | Airborne <4g |
| `CFG_SIGNAL_*_ENA` | 1 each | GPS, Galileo, BeiDou, GLONASS, SBAS, QZSS |

## CFG-VALSET Protocol

All configuration uses UBX-CFG-VALSET (class=0x06, id=0x8A). Payload format:

```
[0]    version = 0x00
[1]    layer bitmask: 0x01=RAM, 0x02=BBR, 0x04=Flash
[2-3]  reserved = 0x0000
[4..]  key-value pairs (4-byte LE key + typed value)
```

Key ID encoding: `0xSSKKKKKK` where top nibble `S` encodes size
(1=1-bit/bool stored as U1, 2=U1, 3=U2, 4=U4, 5=U8, etc.).

---

## All Tunable Parameters

### Navigation — CFG-NAVSPG

Controls the navigation engine's solver behavior, input quality gates,
and output accuracy masks.

#### Dynamic Platform Model (CFG_NAVSPG_DYNMODEL)

Key: `0x20110021` (U1)

| Value | Model | Max Alt | Max Vel | Max Accel | Notes |
|---|---|---|---|---|---|
| 0 | Portable | 12 km | 310 m/s | — | General purpose |
| 2 | Stationary | 9 km | 10 m/s | — | Static applications |
| 3 | Pedestrian | 9 km | 30 m/s | — | Walking |
| 4 | Automotive | 9 km | 84 m/s | — | Road vehicles |
| 5 | Sea | 9 km | 25 m/s | — | Marine |
| 6 | Airborne <1g | 50 km | 100 m/s | 1g | Low-dynamics air |
| 7 | Airborne <2g | 50 km | 250 m/s | 2g | Medium-dynamics air |
| **8** | **Airborne <4g** | **50 km** | **500 m/s** | **4g** | **Our setting — drones** |
| 9 | Wrist | 9 km | 30 m/s | — | Wearable |
| 10 | Bike | 9 km | 84 m/s | — | Cycling |

Airborne <4g is optimal for drones: raises altitude limit to 50 km and
allows up to 4g acceleration dynamics in the Kalman filter.

#### Fix Mode (CFG_NAVSPG_FIXMODE)

Key: `0x20110011` (U1)

| Value | Mode |
|---|---|
| 1 | 2D only |
| 2 | 3D only |
| 3 | Auto 2D/3D (default) |

For EKF fusion, 3D-only (2) is preferred — prevents the solver from
outputting 2D fixes with assumed altitude that would corrupt the filter.

#### Input Filters — Minimum Acquisition Quality

These gate which satellites the solver considers:

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x201100A1` | MINSVS | U1 | 0 | Min SVs for navigation |
| `0x201100A2` | MAXSVS | U1 | 0 (=no limit) | Max SVs for navigation |
| `0x201100A3` | MINCNO | U1 | 0 | Min C/N0 (dBHz) to use a SV |
| `0x201100A4` | MINELEV | I1 | 5 | Min elevation angle (deg) |
| `0x201100AB` | NCNOTHRS | U1 | 0 | Number of SVs required above CNOTHRS |
| `0x201100AC` | CNOTHRS | U1 | 0 | C/N0 threshold for NCNOTHRS check |

**Tuning notes:**
- `MINCNO=6` rejects very weak signals that add noise to the solution
- `MINELEV=10` rejects low-elevation SVs with high multipath
- Keep `MINSVS=0` to let the solver use as many as available

#### Output Filters — Accuracy Masks

These suppress output when accuracy estimates exceed thresholds:

| Key ID | Name | Type | Default | Unit | Description |
|---|---|---|---|---|---|
| `0x201100B1` | PDOP_MASK | U2 | 250 | 0.1 scale | Max PDOP (25.0) |
| `0x201100B2` | TDOP_MASK | U2 | 250 | 0.1 scale | Max TDOP (25.0) |
| `0x201100B3` | PACC_MASK | U2 | 300 | m | Max position accuracy |
| `0x201100B4` | TACC_MASK | U2 | 300 | m | Max time accuracy |
| `0x201100B5` | FACC_MASK | U2 | 0 (=off) | 0.01 m/s | Max speed accuracy |

**Tuning notes:**
- For EKF: tighten `PDOP_MASK` to ~100 (10.0) and `PACC_MASK` to ~50 m
  to avoid feeding degraded fixes into the filter
- The EKF should also gate on `h_acc_mm` from NAV-PVT, but hardware-level
  masking provides an additional safety layer

#### Signal Attenuation Compensation (CFG_NAVSPG_SIGATTCOMP)

Key: `0x201100D6` (U1)

| Value | Mode |
|---|---|
| 0 | Disabled (default) |
| 1 | Automatic |

Compensates for antenna signal attenuation. Enable if the antenna is
mounted under a frame/canopy that attenuates GPS signals.

#### Constrained Altitude (2D fix fallback)

| Key ID | Name | Type | Description |
|---|---|---|---|
| `0x201100C1` | CONSTR_ALT | I4 | Fixed altitude for 2D fix (mm) |
| `0x201100C2` | CONSTR_ALTVAR | U4 | Fixed altitude variance (mm^2) |

Only applies in 2D mode or when fix degrades to 2D.

#### UTC Standard

Key: `0x2011000C` `CFG_NAVSPG_UTCSTANDARD` (U1)

| Value | Standard |
|---|---|
| 0 | Automatic |
| 3 | USNO (GPS) |
| 5 | European (Galileo) |
| 6 | SU (GLONASS) |
| 7 | NTSC (BeiDou) |

#### Other NAVSPG Keys

| Key ID | Name | Type | Description |
|---|---|---|---|
| `0x30110017` | WKNROLLOVER | U2 | GPS week rollover number |
| `0x40110009` | USRDATUM_DATMAJ_X | R4 | User datum semi-major axis |
| `0x4011000A` | USRDATUM_DATMAJ_Y | R4 | User datum flattening |
| `0x4011000B` | USRDATUM_DATROT_X | R4 | User datum rotation X |
| `0x4011000C` | USRDATUM_DATROT_Y | R4 | User datum rotation Y |
| `0x4011000D` | USRDATUM_DATROT_Z | R4 | User datum rotation Z |
| `0x40110010` | USRDATUM_DATSCALE | R4 | User datum scale factor |

---

### Measurement & Navigation Rate — CFG-RATE

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x30210001` | MEAS | U2 | 1000 | Measurement period (ms), min 25 ms |
| `0x30210002` | NAV | U2 | 1 | Nav solutions per measurement (1-127) |
| `0x20210003` | TIMEREF | U1 | 0 | Time system reference |

TIMEREF values: 0=UTC, 1=GPS, 2=GLONASS, 3=BeiDou, 4=Galileo

**Tuning notes:**
- Our setting: MEAS=200 ms → 5 Hz. The M10 supports down to 25 ms (40 Hz)
  but multi-constellation fixes are limited to ~10 Hz
- For EKF fusion, 5-10 Hz is the sweet spot — faster rates reduce fix
  quality because fewer measurements accumulate per solution
- NAV=1 means one solution per measurement cycle (normal)

---

### Signal / Constellation Enable — CFG-SIGNAL

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x1031001F` | GPS_ENA | L | true | Enable GPS L1C/A |
| `0x10310001` | GPS_L1CA_ENA | L | true | GPS L1C/A signal |
| `0x10310021` | GAL_ENA | L | true | Enable Galileo |
| `0x10310007` | GAL_E1_ENA | L | true | Galileo E1 signal |
| `0x10310025` | GLO_ENA | L | true | Enable GLONASS |
| `0x10310004` | GLO_L1_ENA | L | true | GLONASS L1 signal |
| `0x10310022` | BDS_ENA | L | true | Enable BeiDou |
| `0x1031000D` | BDS_B1_ENA | L | true | BeiDou B1 signal |
| `0x10310020` | SBAS_ENA | L | true | Enable SBAS |
| `0x10310005` | SBAS_L1CA_ENA | L | true | SBAS L1C/A signal |
| `0x10310024` | QZSS_ENA | L | true | Enable QZSS |
| `0x10310009` | QZSS_L1CA_ENA | L | true | QZSS L1C/A signal |

**Tuning notes:**
- All constellations enabled maximizes SV count for best accuracy
- The M10 is single-band (L1) — no L5/E5 signals available
- SBAS provides ionospheric corrections in supported regions (WAAS in
  North America, EGNOS in Europe) — keep enabled
- QZSS improves coverage in Asia-Pacific; negligible overhead elsewhere
- Our driver sends each constellation as a separate CFG-VALSET and
  tolerates NAK (some modules may not support all constellations)

---

### SBAS Configuration — CFG-SBAS

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x10360100` | USE_TESTMODE | L | false | Use SBAS in test mode |
| `0x10360101` | USE_RANGING | L | true | Use SBAS SVs for ranging |
| `0x10360102` | USE_DIFFCORR | L | true | Use SBAS differential corrections |
| `0x10360103` | USE_INTEGRITY | L | false | Use SBAS integrity information |
| `0x40360104` | PRNSCANMASK | U4 | 0 (=all) | Bitmask of SBAS PRNs to search |

**Tuning notes:**
- Default SBAS config is fine for most use cases
- `USE_INTEGRITY=true` can cause the receiver to reject fixes when SBAS
  integrity data is unavailable — keep disabled for drone flight
- `USE_RANGING=true` lets SBAS geostationary satellites augment the
  position solution (free extra SV)

---

### Static Hold / Motion Detection — CFG-MOT

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x20250023` | GNSSSPEED_THRS | U1 | 0 | Speed threshold (cm/s), 0=disabled |
| `0x20250024` | GNSSDIST_THRS | U2 | 0 | Distance threshold (m), 0=disabled |

When speed drops below `GNSSSPEED_THRS`, the receiver enters static hold
mode and locks position output until movement exceeds `GNSSDIST_THRS`.

**Tuning notes:**
- Useful for ground stations to reduce position wander at rest
- **Disable for drones** (our default) — static hold would mask low-speed
  movements during hover/landing and confuse the EKF

---

### Odometer — CFG-ODO

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x10220001` | USE_ODO | L | false | Enable odometer |
| `0x10220002` | USE_COG | L | false | Use COG filter |
| `0x20220005` | PROFILE | U1 | 0 | Filter profile |
| `0x20220031` | COGMAXSPEED | U1 | 0 | Max speed for COG filter (m/s) |
| `0x20220032` | COGMAXPOSACC | U1 | 0 | Max position accuracy for COG (m) |
| `0x10220004` | OUTLPVEL | L | false | Output low-pass filtered velocity |
| `0x10220003` | OUTLPCOG | L | false | Output low-pass filtered COG |

PROFILE values: 0=Running, 1=Cycling, 2=Swimming, 3=Car, 4=Custom

**Tuning notes:**
- Odometer is irrelevant for drone flight — leave disabled
- COG (course over ground) filter smooths heading at low speeds but
  introduces latency — not suitable for EKF fusion

---

### Jamming / Interference Detection — CFG-ITFM

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x20410001` | BBTHRESHOLD | U1 | 3 | Broadband jamming threshold (dB) |
| `0x20410002` | CWTHRESHOLD | U1 | 15 | CW (narrowband) jamming threshold (dB) |
| `0x10410005` | ENABLE | L | true | Enable jamming monitor |
| `0x10410010` | ANTSETTING | U1 | 0 | Antenna setting (0=unknown, 1=passive, 2=active) |

**Tuning notes:**
- Keep jamming monitor enabled — NAV-PVT `flags` byte reports jamming state
- Set `ANTSETTING` to match your antenna (passive patch = 1, active = 2)
  for better threshold calibration
- Default thresholds are reasonable; only adjust if getting false positives

---

### Power Management — CFG-PM

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x20D00001` | OPERATEMODE | U1 | 0 | Operating mode |
| `0x30D00002` | POSUPDATEPERIOD | U4 | 0 | Position update period (ms) |
| `0x30D00003` | ACQPERIOD | U4 | 0 | Acquisition retry period (ms) |
| `0x30D00004` | GRIDOFFSET | U4 | 0 | Grid offset (ms) |
| `0x30D00005` | ONTIME | U2 | 0 | On time after fix (s) |
| `0x30D00006` | MINACQTIME | U1 | 0 | Min acquisition time (s) |
| `0x30D00007` | MAXACQTIME | U1 | 0 | Max acquisition time (s) |
| `0x10D00008` | DONOTENTEROFF | L | false | Prevent entering OFF state |
| `0x10D00009` | WAITTIMEFIX | L | true | Wait for time fix before sleep |
| `0x10D0000A` | UPDATEEPH | L | false | Update ephemeris before sleep |
| `0x10D0000B` | EXTINT0WAKE | L | true | EXTINT0 wakes receiver |
| `0x10D0000C` | EXTINT0BACKUP | L | false | EXTINT0 forces backup |
| `0x10D0000D` | EXTINT1BACKUP | L | false | EXTINT1 forces backup |
| `0x10D0000E` | LIMITPEAKCURR | L | true | Limit peak current |

OPERATEMODE values:
- 0 = Full power (continuous tracking, default)
- 1 = PSMOO (Power Save Mode — On/Off operation)
- 2 = PSMCT (Power Save Mode — Cyclic Tracking)

**Tuning notes:**
- **Always use full power (0) for drone flight** — power save modes
  introduce latency and reduce fix rate
- Power save modes are for battery-powered trackers, not real-time EKF

---

### UART1 Configuration — CFG-UART1

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x40520001` | BAUDRATE | U4 | 38400 | UART1 baud rate |
| `0x10520005` | ENABLED | L | true | Enable UART1 |
| `0x20520008` | DATABITS | U1 | 3 | 0=8bit, 1=7bit (3=auto?) |
| `0x20520009` | STOPBITS | U1 | 1 | 0=0.5, 1=1, 2=1.5, 3=2 |
| `0x2052000A` | PARITY | U1 | 0 | 0=none, 1=odd, 2=even |

**Tuning notes:**
- Default 38400 baud is fine for 5 Hz NAV-PVT (92-byte payload ≈ 4 kB/s)
- At 10 Hz or with additional messages, consider 115200 baud
- Must match the UART baud rate configured in the BSP

#### UART1 Protocol Filters

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x10730001` | UART1INPROT_UBX | L | true | Accept UBX input |
| `0x10730002` | UART1INPROT_NMEA | L | true | Accept NMEA input |
| `0x10740001` | UART1OUTPROT_UBX | L | true | Output UBX messages |
| `0x10740002` | UART1OUTPROT_NMEA | L | true | Output NMEA messages |

Our driver disables NMEA output and enables UBX-only for efficient binary parsing.

---

### I2C (DDC) Configuration — CFG-I2C

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x20510001` | ADDRESS | U1 | 0x42 | I2C slave address |
| `0x10510003` | ENABLED | L | true | Enable I2C interface |
| `0x20510006` | MAXRETRIES | U1 | 0 | Max I2C stretch retries |

Not used in cybflight (we use UART), but relevant if switching to I2C.

#### I2C Protocol Filters

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x10710001` | I2CINPROT_UBX | L | true | Accept UBX on I2C |
| `0x10710002` | I2CINPROT_NMEA | L | true | Accept NMEA on I2C |
| `0x10720001` | I2COUTPROT_UBX | L | true | Output UBX on I2C |
| `0x10720002` | I2COUTPROT_NMEA | L | true | Output NMEA on I2C |

---

### Message Output Rates — CFG-MSGOUT

Per-message, per-port output rate. Value = number of navigation cycles
between outputs (0=disabled, 1=every cycle, 2=every other, etc.).

Key format: `0x2091XXYY` where XX-YY encodes message + port.

#### NAV Messages (UART1)

| Key ID | Name | Description |
|---|---|---|
| `0x20910007` | UBX_NAV_PVT_UART1 | Position/Velocity/Time **(we use this)** |
| `0x2091001A` | UBX_NAV_SAT_UART1 | Satellite information |
| `0x20910038` | UBX_NAV_STATUS_UART1 | Receiver navigation status |
| `0x20910044` | UBX_NAV_DOP_UART1 | Dilution of precision |
| `0x20910061` | UBX_NAV_POSLLH_UART1 | Position (LLH) |
| `0x20910069` | UBX_NAV_VELNED_UART1 | Velocity (NED frame) |
| `0x2091004D` | UBX_NAV_TIMEGPS_UART1 | GPS time solution |
| `0x20910056` | UBX_NAV_TIMEUTC_UART1 | UTC time solution |
| `0x2091003E` | UBX_NAV_COV_UART1 | Covariance matrices |
| `0x2091002F` | UBX_NAV_SIG_UART1 | Signal information |
| `0x20910075` | UBX_NAV_CLOCK_UART1 | Clock solution |
| `0x209100A1` | UBX_NAV_SVIN_UART1 | Survey-in data |

**Tuning notes for EKF:**
- NAV-PVT alone is sufficient — it contains position, velocity, accuracy
  estimates, fix type, and satellite count
- Consider enabling NAV-DOP at a lower rate (e.g., every 5th cycle) for
  EKF measurement noise scaling based on DOP decomposition
- NAV-COV provides full covariance matrices — useful for advanced EKF
  measurement noise adaptation, but large payload (64 bytes)
- NAV-SAT is useful for diagnostics but large; enable only for debugging

#### MON Messages (UART1)

| Key ID | Name | Description |
|---|---|---|
| `0x20910196` | UBX_MON_RF_UART1 | RF status (jamming, noise) |
| `0x209101A2` | UBX_MON_IO_UART1 | I/O subsystem status |
| `0x209101B4` | UBX_MON_COMMS_UART1 | Communication port info |
| `0x2091034F` | UBX_MON_TEMP_UART1 | Temperature |

---

### Timepulse — CFG-TP

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x40050001` | TP1_ENA | L | true | Enable timepulse 1 |
| `0x40050002` | FREQ_TP1 | U4 | 1 | Frequency before fix (Hz) |
| `0x40050003` | FREQ_LOCK_TP1 | U4 | 1 | Frequency after fix (Hz) |
| `0x40050004` | DUTY_TP1 | R8 | 50.0 | Duty cycle (%) |
| `0x40050005` | DUTY_LOCK_TP1 | R8 | 50.0 | Duty cycle after fix (%) |
| `0x20050009` | POL_TP1 | U1 | 1 | Polarity (0=falling, 1=rising) |
| `0x2005000A` | TIMEGRID_TP1 | U1 | 0 | Time grid (0=UTC, 1=GPS) |

**Tuning notes:**
- Timepulse can provide a PPS signal for precise time synchronization
- Not currently used in cybflight but useful for camera trigger sync
  or inter-board time alignment

---

### TX Ready — CFG-TXREADY

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x10A20001` | ENABLED | L | false | Enable TX-ready indication |
| `0x10A20002` | POLARITY | U1 | 0 | 0=high-active, 1=low-active |
| `0x30A20003` | THRESHOLD | U2 | 0 | Byte count threshold |
| `0x10A20004` | INTERFACE | U1 | 0 | Interface (0=I2C, 1=SPI) |

Not used — this is for host-side flow control on I2C/SPI interfaces.

---

### Hardware Configuration — CFG-HW

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x10A3000A` | ANT_CFG_VOLTCTRL | L | false | Active antenna voltage control |
| `0x10A3000B` | ANT_CFG_SHORTDET | L | false | Short circuit detection |
| `0x10A3000C` | ANT_CFG_SHORTOPEN | L | false | Open circuit detection |
| `0x10A3000D` | ANT_CFG_OPENDET | L | false | Open detection |
| `0x10A30014` | ANT_CFG_RECOVER | L | true | Auto-recover from short |
| `0x30A30018` | ANT_SUP_SWITCH_PIN | U1 | 255 | Antenna supervisor switch PIO |
| `0x30A30019` | ANT_SUP_SHORT_PIN | U1 | 255 | Antenna supervisor short PIO |
| `0x3011001A` | ANT_SUP_OPEN_PIN | U1 | 255 | Antenna supervisor open PIO |

**Tuning notes:**
- Only relevant with active antennas that need bias voltage control
- SAM-M10Q has an integrated patch antenna — leave defaults

---

### Security — CFG-SEC

| Key ID | Name | Type | Description |
|---|---|---|---|
| `0x10F60009` | CFG_SEC_UNIQID | U5 (read-only) | Unique chip ID |

The M10 security features are primarily read-only identifiers.

---

### Informational Messages — CFG-INFMSG

| Key ID | Name | Type | Default | Description |
|---|---|---|---|---|
| `0x20920001` | UBX_UART1 | U1 | 0 | UBX informational msg filter (UART1) |
| `0x20920002` | NMEA_UART1 | U1 | 0 | NMEA informational msg filter (UART1) |

Bitmask: bit0=ERROR, bit1=WARNING, bit2=NOTICE, bit3=TEST, bit4=DEBUG.

---

### Receiver Inventory — CFG-RINV

| Key ID | Name | Type | Description |
|---|---|---|---|
| `0x10C40001` | DUMP | L | Dump receiver inventory on startup |
| `0x10C40002` | BINARY | L | Inventory data is binary |
| `0x50C40003` | DATA | X (30 bytes) | Inventory string data |

User-defined inventory string stored in BBR. Not relevant for flight.

---

## Recommended EKF-Optimized Configuration

Beyond our current settings, these additional parameters would improve
GPS data quality for EKF fusion:

```rust
// Fix mode: 3D only (prevent 2D fixes with assumed altitude)
// Key: 0x20110011, Value: 2
off = append_kv_u8(&mut payload, off, 0x20110011, 2);

// Min C/N0: reject weak signals below 6 dBHz
// Key: 0x201100A3, Value: 6
off = append_kv_u8(&mut payload, off, 0x201100A3, 6);

// Min elevation: reject low-elevation SVs (multipath)
// Key: 0x201100A4, Value: 10
off = append_kv_i8(&mut payload, off, 0x201100A4, 10);

// PDOP mask: reject solutions with PDOP > 10.0
// Key: 0x201100B1, Value: 100 (0.1 scale)
off = append_kv_u16(&mut payload, off, 0x201100B1, 100);

// Position accuracy mask: reject solutions with hAcc > 50m
// Key: 0x201100B3, Value: 50
off = append_kv_u16(&mut payload, off, 0x201100B3, 50);
```

These are suggestions — not yet applied in the driver. The EKF should
also perform its own gating on NAV-PVT accuracy fields (`h_acc_mm`,
`v_acc_mm`, `pdop`) as a second layer of quality control.
