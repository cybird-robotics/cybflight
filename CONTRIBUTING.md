# Contributing

Use a focused change with a clear problem, behavior, and validation record. Start with `docs/architecture.md`; use the pinned toolchain and keep reference builds consistent with vehicle YAMLs.

Run `just test-all` for algorithm or integration changes; it covers core, build-tool, driver, and regression tests. Build `sakura_vicon`, `sakura_um982`, and `sakura_ublox_f9` with the configuration guard enabled. Mutually exclusive firmware features must be checked as separate configurations, not with `--all-features`.

Controller and estimator changes need meaningful host tests, simulation evidence, and target timing checks. Hardware changes need documented wiring and bench evidence; flight validation remains separate from compilation. Keep unvalidated hardware outside the reference matrix until someone can maintain and test it.

Do not put credentials, machine paths, or deployment-specific settings in examples. Preserve wire message IDs, review layout changes across all endpoints, and publish a compatible `cybflight-msgs` version before updating consumers.

Contributions are licensed under Apache-2.0. Preserve third-party notices when adding dependencies or incorporating third-party material.
