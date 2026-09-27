# Coordinated releases

Cybflight firmware, the ESP32 bridge, and the ground station share a wire protocol through `cybflight-msgs`. Keep that dependency at the same exact version across all three repositories. Endpoints do not negotiate protocol versions at runtime.

## Library versions

| Library | Version | Consumers |
|---|---|---|
| `cybflight-msgs` | 0.2.0 | Firmware, ESP32 bridge, ground station |
| `um982` | 0.1.0 | Firmware |
| `vicon-sdk` | 0.1.0 | Ground station |

Each library publishes to crates.io when a stable GitHub release is created with a tag matching its Cargo package version. Configure the repository Actions secret `CRATES_IO_TOKEN` before publishing. Application releases distribute firmware or desktop binaries through GitHub Releases.

## Release process

1. Publish any changed libraries, then update the consumers' version requirements. Review and commit `Cargo.lock`; verify resolution with `cargo metadata --locked --format-version 1`.
2. Build the SAKURA Vicon, UM982, and u-blox F9 configurations and both ESP32-C6 board variants with `--locked`. Run host tests and simulation regressions. Build and test `cybgcs` with the matching Vicon native SDK installed.
3. Validate the firmware, bridge, and ground station together on hardware. Record source commits, compiler, board revisions, wiring, airframe settings, correction source, and results with the release.
4. Tag the reviewed commits and create GitHub releases. Attach binaries and checksums, and document compatible firmware, bridge, and ground-station versions. Use a new tag for each release.

## Native and Git dependencies

The embedded FAT filesystem is pinned to a Git revision whose I/O traits match the firmware.

Install the Vicon native SDK separately. `VICON_SDK_DIR` controls link-time discovery; configure the operating system's library loader for runtime use.
