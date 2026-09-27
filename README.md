<p align="center">
  <img src="docs/assets/cybird-logo.png" alt="Cybird logo" width="160" height="160">
</p>

<h1 align="center">Cybflight</h1>

<p align="center">Open-source flight control, written in Rust.</p>

<p align="center">
  <a href="#getting-started">Get started</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="#ecosystem">Ecosystem</a> ·
  <a href="CONTRIBUTING.md">Contribute</a>
</p>

Cybflight is a modular autopilot for multirotors, part of the [Cybird project](https://github.com/cybird-robotics). It runs on STM32H743 flight controllers, with an ESP32 WiFi bridge for telemetry, positioning data, and RTK corrections.

- **Control and estimation** — INDI, model predictive control, and an error-state Kalman filter.
- **Indoor and outdoor positioning** — Vicon motion capture and RTK GNSS.
- **Configuration and diagnostics** — YAML airframes, a USB parameter shell, live telemetry, and onboard blackbox logging.

## Getting started

Install [Rust](https://rustup.rs/) with the platform prerequisites listed by its installer. The repository pins the Rust toolchain and embedded target; build commands use Bash and `just`.

```sh
git clone https://github.com/cybird-robotics/cybflight.git
cd cybflight
cargo install just --locked
just build sakura_vicon
```

Use `just vehicles` to list configurations. Choose `sakura_um982` or `sakura_ublox_f9` for RTK GNSS. Builds produce an ELF and binary in `target/thumbv7em-none-eabihf/release/`.

See [hardware setup](docs/hardware-support.md) for wiring, receiver configuration, and bring-up. Adapt the airframe settings, tuning, and mission coordinates to your vehicle before flying.

## Documentation

- [Architecture](docs/architecture.md) — boards, drivers, tasks, and control.
- [Hardware setup](docs/hardware-support.md) — SAKURA, ESP32 boards, and positioning.
- [Parameters](docs/parameters.md) — configuration and tuning.
- [Blackbox logging](docs/blackbox.md) — recording and analyzing flight data.
- [Coordinated releases](docs/releasing.md) — compatible firmware, bridge, and ground-station versions.

## Ecosystem

| Project | Purpose |
|---|---|
| [cybesp-bridge](https://github.com/cybird-robotics/cybesp-bridge) | WiFi bridge for ESP32-C6 boards, including Waveshare and Seeed XIAO |
| [cybgcs](https://github.com/cybird-robotics/cybgcs) | Ground station for Vicon, RTK corrections, telemetry, and visualization |
| [cybflight-msgs](https://github.com/cybird-robotics/cybflight-msgs) | Shared wire protocol |

The [um982](https://github.com/cybird-robotics/um982) driver and [vicon-sdk](https://github.com/cybird-robotics/vicon-sdk) bindings are available on crates.io.

## Contributing

Bug fixes, hardware support, controllers, documentation, and tools are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

```sh
just test
just test-drivers
```

## License

[Apache-2.0](LICENSE). Copyright 2026 The Cybird project contributors. See [NOTICE](NOTICE).

## Citation

If you use Cybflight in your research, please cite:

```bibtex
@misc{lin2026cybflight,
  title  = {Cybflight: An Embedded Rust Autopilot for Aerial Robotics Research},
  author = {Yifan Lin and Chao Qin and H. S. Helson Go and Hugh H.-T. Liu},
  year   = {2026},
  howpublished = {IEEE IROS 2026 Workshop Why Rust for Robotics: A Perspective From Industry}
}
```
