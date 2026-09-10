# Parameter reference

Auto-generated from the `#[derive(Params)]` registry by
`just params-doc` — do not edit by hand. 316 parameters.

Values shown by `param list`/`get`; set with `param set <name> <value>`
(disarmed), persist with `param save`, inspect overrides with
`param diff [--yaml]`, revert with `param reset <name>|all`.
Baked defaults come from `vehicles/<VEHICLE>.yaml`; runtime overrides
live in the flash KV store and win per-key — so a saved override
shadows a re-flashed YAML edit. Follow `param reset` with
`param save --prune` to drop the key from the store; a plain save
only records the current baked value (the log has no tombstone).

| Name | Unit | Min | Max | Apply | Description |
|---|---|---|---|---|---|
| `mass` | kg | 0.05 | 20 | reboot | Vehicle mass (kg). |
| `ixx` | kg·m² | -1 | 1 | reboot | Inertia tensor stored row-major: [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]. |
| `ixy` | kg·m² | -1 | 1 | reboot | Inertia tensor stored row-major: [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]. |
| `ixz` | kg·m² | -1 | 1 | reboot | Inertia tensor stored row-major: [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]. |
| `iyx` | kg·m² | -1 | 1 | reboot | Inertia tensor stored row-major: [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]. |
| `iyy` | kg·m² | -1 | 1 | reboot | Inertia tensor stored row-major: [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]. |
| `iyz` | kg·m² | -1 | 1 | reboot | Inertia tensor stored row-major: [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]. |
| `izx` | kg·m² | -1 | 1 | reboot | Inertia tensor stored row-major: [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]. |
| `izy` | kg·m² | -1 | 1 | reboot | Inertia tensor stored row-major: [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]. |
| `izz` | kg·m² | -1 | 1 | reboot | Inertia tensor stored row-major: [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]. |
| `max_rate_r` | rad/s | 0.1 | 50 | reboot |  |
| `max_rate_p` | rad/s | 0.1 | 50 | reboot |  |
| `max_rate_y` | rad/s | 0.1 | 50 | reboot |  |
| `m0_px` | m | -1 | 1 | reboot | Motor position in body XY plane [x_m, y_m] in FLU frame. |
| `m0_py` | m | -1 | 1 | reboot | Motor position in body XY plane [x_m, y_m] in FLU frame. |
| `m0_spin` | — | — | — | reboot | Propeller spin direction (viewed from above). |
| `m0_thrust` | N | 0.1 | 200 | reboot | Maximum thrust this motor+propeller produces at full throttle (Newtons). |
| `m0_torque` | m | 0 | 0.2 | reboot | Reaction torque per unit thrust (metres). Ratio of yaw reaction torque to |
| `m0_tau` | s | 0.001 | 1 | live | First-order spool-up time constant of motor+ESC+prop [s], from a |
| `m0_omega_max` | rad/s | 50 | 20000 | live | Motor speed at full throttle [rad/s] — Kv × pack voltage, or read |
| `m0_g2_rr` | — | -10000 | 10000 | live | G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular |
| `m0_g2_rp` | — | -10000 | 10000 | live | G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular |
| `m0_g2_ry` | — | -10000 | 10000 | live | G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular |
| `m0_nonlin` | — | 0.025 | 1 | live | Thrust-curve nonlinearity `k` for this motor+prop. **0 selects the |
| `m1_px` | m | -1 | 1 | reboot | Motor position in body XY plane [x_m, y_m] in FLU frame. |
| `m1_py` | m | -1 | 1 | reboot | Motor position in body XY plane [x_m, y_m] in FLU frame. |
| `m1_spin` | — | — | — | reboot | Propeller spin direction (viewed from above). |
| `m1_thrust` | N | 0.1 | 200 | reboot | Maximum thrust this motor+propeller produces at full throttle (Newtons). |
| `m1_torque` | m | 0 | 0.2 | reboot | Reaction torque per unit thrust (metres). Ratio of yaw reaction torque to |
| `m1_tau` | s | 0.001 | 1 | live | First-order spool-up time constant of motor+ESC+prop [s], from a |
| `m1_omega_max` | rad/s | 50 | 20000 | live | Motor speed at full throttle [rad/s] — Kv × pack voltage, or read |
| `m1_g2_rr` | — | -10000 | 10000 | live | G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular |
| `m1_g2_rp` | — | -10000 | 10000 | live | G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular |
| `m1_g2_ry` | — | -10000 | 10000 | live | G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular |
| `m1_nonlin` | — | 0.025 | 1 | live | Thrust-curve nonlinearity `k` for this motor+prop. **0 selects the |
| `m2_px` | m | -1 | 1 | reboot | Motor position in body XY plane [x_m, y_m] in FLU frame. |
| `m2_py` | m | -1 | 1 | reboot | Motor position in body XY plane [x_m, y_m] in FLU frame. |
| `m2_spin` | — | — | — | reboot | Propeller spin direction (viewed from above). |
| `m2_thrust` | N | 0.1 | 200 | reboot | Maximum thrust this motor+propeller produces at full throttle (Newtons). |
| `m2_torque` | m | 0 | 0.2 | reboot | Reaction torque per unit thrust (metres). Ratio of yaw reaction torque to |
| `m2_tau` | s | 0.001 | 1 | live | First-order spool-up time constant of motor+ESC+prop [s], from a |
| `m2_omega_max` | rad/s | 50 | 20000 | live | Motor speed at full throttle [rad/s] — Kv × pack voltage, or read |
| `m2_g2_rr` | — | -10000 | 10000 | live | G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular |
| `m2_g2_rp` | — | -10000 | 10000 | live | G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular |
| `m2_g2_ry` | — | -10000 | 10000 | live | G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular |
| `m2_nonlin` | — | 0.025 | 1 | live | Thrust-curve nonlinearity `k` for this motor+prop. **0 selects the |
| `m3_px` | m | -1 | 1 | reboot | Motor position in body XY plane [x_m, y_m] in FLU frame. |
| `m3_py` | m | -1 | 1 | reboot | Motor position in body XY plane [x_m, y_m] in FLU frame. |
| `m3_spin` | — | — | — | reboot | Propeller spin direction (viewed from above). |
| `m3_thrust` | N | 0.1 | 200 | reboot | Maximum thrust this motor+propeller produces at full throttle (Newtons). |
| `m3_torque` | m | 0 | 0.2 | reboot | Reaction torque per unit thrust (metres). Ratio of yaw reaction torque to |
| `m3_tau` | s | 0.001 | 1 | live | First-order spool-up time constant of motor+ESC+prop [s], from a |
| `m3_omega_max` | rad/s | 50 | 20000 | live | Motor speed at full throttle [rad/s] — Kv × pack voltage, or read |
| `m3_g2_rr` | — | -10000 | 10000 | live | G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular |
| `m3_g2_rp` | — | -10000 | 10000 | live | G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular |
| `m3_g2_ry` | — | -10000 | 10000 | live | G2 rate-dependent effectiveness [roll, pitch, yaw]: the angular |
| `m3_nonlin` | — | 0.025 | 1 | live | Thrust-curve nonlinearity `k` for this motor+prop. **0 selects the |
| `mag_hi_x` | — | — | — | reboot | Magnetometer hard-iron offset, sensor frame (raw units). The |
| `mag_hi_y` | — | — | — | reboot | Magnetometer hard-iron offset, sensor frame (raw units). The |
| `mag_hi_z` | — | — | — | reboot | Magnetometer hard-iron offset, sensor frame (raw units). The |
| `gps_ant_x` | m | -2 | 2 | reboot | GPS ANT1 (position antenna) phase-centre lever arm from the IMU, |
| `gps_ant_y` | m | -2 | 2 | reboot | GPS ANT1 (position antenna) phase-centre lever arm from the IMU, |
| `gps_ant_z` | m | -2 | 2 | reboot | GPS ANT1 (position antenna) phase-centre lever arm from the IMU, |
| `gps_base_x` | — | -1 | 1 | reboot | ANT1→ANT2 baseline unit direction, body/FLU frame (dual-antenna |
| `gps_base_y` | — | -1 | 1 | reboot | ANT1→ANT2 baseline unit direction, body/FLU frame (dual-antenna |
| `gps_base_z` | — | -1 | 1 | reboot | ANT1→ANT2 baseline unit direction, body/FLU frame (dual-antenna |
| `motor_poles` | — | 2 | 60 | reboot | Motor pole count, for the eRPM → RPM conversion on DShot |
| `imu_accel_lpf_hz` | Hz | 1 | 2000 | reboot | IMU software LPF cutoff for accel (Hz), applied by `ImuReader`. |
| `imu_gyro_lpf_hz` | Hz | 1 | 2000 | reboot | IMU software LPF cutoff for gyro (Hz), applied by `ImuReader`. |
| `gps_fuse_vel` | — | — | — | reboot | Fuse GNSS velocity into the ESKF (`update_vel`). Works with any |
| `eskf_acc_noise` | — | 0.000001 | 10 | reboot | Accelerometer noise density (m/s² per √Hz). |
| `eskf_gyro_noise` | — | 0.00000001 | 1 | reboot | Gyroscope noise density (rad/s per √Hz). |
| `eskf_acc_bias_rw` | — | 0.000000001 | 1 | reboot | Accelerometer bias random walk (m/s² per √s). |
| `eskf_gyro_bias_rw` | — | 0.000000001 | 1 | reboot | Gyroscope bias random walk (rad/s per √s). |
| `eskf_baro_std` | m | 0.01 | 100 | reboot | Barometer altitude measurement 1-σ (m). |
| `eskf_mag_std` | — | 0.0001 | 10 | reboot | Magnetometer field measurement 1-σ (body-frame µT equivalent). |
| `eskf_gate_sigma` | — | 1 | 100 | reboot | Mahalanobis outlier gate, in sigma units per dimension: a |
| `eskf_max_pos_jump_m` | m | 0.05 | 100 | reboot | Absolute position-innovation reject threshold (m), independent of |
| `eskf_max_att_jump_rad` | rad | 0.05 | 3.15 | reboot | Absolute attitude-innovation reject threshold (rad), independent |
| `eskf_init_pos_var` | — | 0.000001 | 10000 | reboot | Initial position variance [m²]. Set it to the bootstrap source's |
| `eskf_init_vel_var` | — | 0.000001 | 10000 | reboot | Initial velocity variance [(m/s)²]. |
| `eskf_init_acc_bias_var` | — | 0.000000001 | 100 | reboot | Initial accelerometer-bias variance [(m/s²)²]. |
| `eskf_init_gyro_bias_var` | — | 0.000000001 | 100 | reboot | Initial gyro-bias variance [(rad/s)²]. **Coupled to the guards' |
| `eskf_init_att_var_rp` | rad^2 | 0.000001 | 100 | reboot | Initial roll/pitch orientation variance [rad²]. Source-independent |
| `eskf_inflation_cap` | — | 1 | 1000000 | reboot | Hard cap on the measurement-noise inflation factor. Past it a |
| `eskf_mag_norm_gate` | — | 0.01 | 2 | reboot | Magnetometer norm gate: reject a sample whose field magnitude |
| `eskf_max_predict_dt_s` | s | 0.002 | 1 | reboot | Longest IMU gap [s] the estimator will integrate across. |
| `eskf_gps_max_jumps` | — | 1 | 100 | reboot | Consecutive jump-gated PVTs before the guard disarms (and, when |
| `eskf_gps_max_rejects` | — | 1 | 200 | reboot | Consecutive filter-rejected PVTs before the guard disarms. |
| `eskf_gps_stale_s` | s | 0.1 | 60 | reboot | No accepted PVT for this long ⇒ stale: odometry publishing stops |
| `eskf_gps_rtk_fix_debounce_s` | s | 0 | 60 | reboot | Sustained `carr_soln ≥ 2` for this long raises `rtk_quality_ok`. |
| `eskf_gps_rtk_loss_debounce_s` | s | 0 | 60 | reboot | Sustained `carr_soln < 2` for this long clears `rtk_quality_ok`. |
| `eskf_gps_bias_cov_thresh` | — | 0.000001 | 1 | reboot | Gyro-bias covariance xy-trace below which the filter counts as |
| `eskf_gps_init_yaw_cov` | — | 0.001 | 100 | reboot | Yaw covariance seeded at init / re-init. GPS-only cannot observe |
| `eskf_gps_min_sv` | — | 0 | 60 | reboot | Minimum satellite count for a PVT to be usable. |
| `eskf_gps_h_acc_max_m` | m | 0.01 | 1000 | reboot | Maximum receiver-reported horizontal accuracy for a usable PVT. |
| `eskf_gps_pos_sigma_fix_m` | m | 0.001 | 100 | reboot | Position measurement σ floor at `carr_soln = 2` (RTK-fixed). |
| `eskf_gps_pos_sigma_float_m` | m | 0.001 | 100 | reboot | Position measurement σ floor at `carr_soln = 1` (RTK-float). |
| `eskf_gps_pos_sigma_none_m` | m | 0.001 | 100 | reboot | Position measurement σ floor at `carr_soln = 0` (stand-alone). |
| `eskf_gps_vel_sigma_m_s` | — | 0.001 | 100 | reboot | Velocity measurement σ floor (applies when `gps_fuse_vel` is set). |
| `eskf_gps_reinit_min_carr_soln` | — | 0 | 2 | reboot | Minimum `carr_soln` for a jump-cascade re-init to fire. 2 = |
| `eskf_gps_heading_sigma_floor_rad` | rad | 0.0001 | 1 | reboot | σ floor for the dual-antenna heading/pitch measurement (rad). |
| `eskf_mocap_max_jumps` | — | 1 | 100 | reboot | Consecutive jump-gated poses before the guard disarms. Mocap does |
| `eskf_mocap_max_rejects` | — | 1 | 200 | reboot | Consecutive filter-rejected poses before the guard disarms. |
| `eskf_mocap_stale_s` | s | 0.01 | 10 | reboot | No accepted pose for this long ⇒ stale. Couple this to the mocap |
| `eskf_mocap_bias_cov_thresh` | — | 0.000001 | 1 | reboot | Gyro-bias covariance trace below which the filter counts as |
| `eskf_mocap_pos_std` | m | 0.0001 | 10 | reboot | Mocap position measurement 1-σ (m). |
| `eskf_mocap_att_std` | rad | 0.0001 | 3.15 | reboot | Mocap attitude measurement 1-σ (rad). |
| `eskf_mocap_reanchor_m` | m | 0.001 | 1 | reboot | Mutual-agreement radius for the **disarmed** re-anchor escape hatch |
| `eskf_mocap_reanchor_frames` | — | 1 | 100 | reboot | Consecutive mutually-agreeing poses required before the disarmed |
| `eskf_fault_pos_timeout_s` | s | 0.05 | 60 | reboot | No position/velocity/attitude update for this long asserts the |
| `eskf_fault_cov_blowup_m2` | m^2 | 0.1 | 10000 | reboot | Position covariance trace above this asserts `COV_TRACE_BLOWUP` [m²]. |
| `eskf_fault_nan_hold_s` | s | 0.1 | 60 | reboot | How long a NaN re-init keeps its fault bit asserted [s]. |
| `eskf_fault_cascade_hold_s` | s | 0.1 | 60 | reboot | How long a guard cascade keeps its fault bit asserted [s]. |
| `mahony_kp` | — | 0 | 20 | reboot | Proportional gain on the accel/mag correction, all axes. |
| `mahony_ki` | — | 0 | 10 | reboot | Integral gain estimating gyro bias, all axes. Raise it for a |
| `mahony_min_accel_g` | g | 0.01 | 1 | reboot | Accelerometer norm floor as a fraction of 1 g. Below it the |
| `mahony_min_mag_ut` | uT | 0.1 | 100 | reboot | Magnetometer norm floor [µT]. Below it the filter degrades from |
| `batt_nominal_v` | V | 6 | 36 | reboot | Bootstrap pack voltage before the first POWER_STATUS frame arrives. |
| `batt_min_v` | V | 5 | 30 | reboot | Plausibility floor — voltage frames below this are dropped as ADC |
| `batt_max_v` | V | 10 | 60 | reboot | Plausibility ceiling — voltage frames above this are dropped as ADC |
| `batt_cell_detect_v` | V | 3 | 5 | reboot | Per-cell voltage used to auto-detect the pack's cell count at |
| `batt_no_battery_v` | V | 0.5 | 10 | reboot | Pack voltage below this reads as "no battery connected", which |
| `batt_lpf_hz` | Hz | 0.1 | 50 | reboot | Cutoff of the single-pole low-pass on the pack-voltage ADC, in Hz. |
| `batt_settle_ticks` | — | 1 | 1000 | reboot | Power-task ticks discarded before the cell count is detected. |
| `batt_max_cells` | — | 1 | 24 | reboot | Largest cell count the detector will report. |
| `rc_min_us` | us | 500 | 1500 | reboot | Channel pulse width at full-low travel [µs]. |
| `rc_mid_us` | us | 800 | 2200 | reboot | Channel pulse width at centre detent [µs]. |
| `rc_max_us` | us | 1500 | 2500 | reboot | Channel pulse width at full-high travel [µs]. |
| `rc_arm_channel` | — | 0 | 15 | reboot | Zero-based channel index of the arm switch (AETR order, so 4 = |
| `rc_arm_threshold_us` | us | 900 | 2100 | reboot | Arm switch reads "armed" above this pulse width [µs]. |
| `rc_throttle_mincheck_us` | us | 900 | 1500 | reboot | Throttle must be below this to permit arming [µs] |
| `rc_mission_channel` | — | 0 | 15 | reboot | Zero-based channel index of the mission trigger. |
| `rc_mission_high_us` | us | 900 | 2100 | reboot | Mission trigger asserts above this [µs] (Schmitt upper edge). |
| `rc_mission_low_us` | us | 900 | 2100 | reboot | Mission trigger releases below this [µs] (Schmitt lower edge). |
| `rc_throttle_land_us` | us | 900 | 1500 | reboot | Throttle below this [µs] commands a descent-to-land. |
| `rc_launch_us` | us | 1000 | 2100 | reboot | Throttle above this [µs] starts the launch latch counting. |
| `rc_launch_confirm_frames` | — | 1 | 100 | reboot | Consecutive RC frames above `rc_launch_us` before launch latches. |
| `rc_xy_deadband` | — | 0 | 0.5 | reboot | Deadband on the horizontal position sticks (normalized). |
| `rc_throttle_deadband` | — | 0 | 0.5 | reboot | Deadband around throttle centre (normalized). Sized for the TX's |
| `rc_rate_deadband` | — | 0 | 0.5 | reboot | Deadband on the rate sticks in `outer_rate` mode (normalized). |
| `rc_xy_rate_m_s` | m/s | 0.05 | 20 | reboot | Full-stick horizontal setpoint slew rate [m/s]. |
| `rc_z_rate_m_s` | m/s | 0.05 | 20 | reboot | Full-stick vertical setpoint slew rate [m/s]. |
| `rc_land_rate_m_s` | m/s | 0.05 | 5 | reboot | Descent rate while landing [m/s]. |
| `rc_land_lead_m` | m | 0.2 | 5 | reboot | How far below the vehicle's own *measured* altitude the landing |
| `rc_max_rate_rp` | rad/s | 0.1 | 50 | reboot | Full-stick roll/pitch body-rate command in `outer_rate` mode |
| `rc_max_rate_yaw` | rad/s | 0.1 | 50 | reboot | Full-stick yaw-rate command [rad/s]. Same stick-scaling |
| `gravity_m_s2` | m/s^2 | 9.6 | 10 | reboot | Local gravitational acceleration [m/s²]. Varies ~0.5% between the |
| `fence_enable` | — | — | — | reboot | Enable the stick-integrator position envelope. Defaults **off**, |
| `fence_x_m` | m | 0.1 | 1000 | reboot | Envelope half-extent along world X [m] (±). |
| `fence_y_m` | m | 0.1 | 1000 | reboot | Envelope half-extent along world Y [m] (±). |
| `fence_z_max_m` | m | 0.1 | 1000 | reboot | Envelope ceiling [m]. |
| `fence_z_min_m` | m | -100 | 1000 | reboot | Lowest altitude the vehicle will ever command [m]. |
| `arm_max_tilt_deg` | deg | 1 | 90 | reboot | Maximum roll/pitch tilt permitted at arming [deg]. |
| `arm_min_link_quality` | % | 0 | 100 | reboot | Minimum RC link quality permitted at arming [%]. |
| `arm_eskf_mahony_tol_deg` | deg | 0.5 | 90 | reboot | Maximum ESKF↔Mahony attitude disagreement permitted at arming [deg]. |
| `arm_accel_tol_m_s2` | m/s^2 | 0.05 | 20 | reboot | Maximum \|‖accel‖ − g\| for the attitude-health accel gate [m/s²]. |
| `arm_gyro_limit_rad_s` | rad/s | 0.01 | 20 | reboot | Maximum per-axis gyro magnitude for the attitude-health gate [rad/s]. |
| `arm_switch_hold_s` | s | 0 | 5 | reboot | Arm switch must be held this long before arming [s]. |
| `arm_link_stats_max_age_s` | s | 0.05 | 10 | reboot | Link statistics older than this block arming [s]. |
| `fs_rxloss_trigger_s` | s | 0.02 | 5 | reboot | No valid RC frame for this long enters the failsafe guard period [s]. |
| `fs_guard_period_s` | s | 0.05 | 30 | reboot | Total RC-loss duration before disarm [s]. Must exceed |
| `fs_recovery_period_s` | s | 0 | 10 | reboot | Continuous valid RC required to clear a failsafe [s]. |
| `fs_ctrl_timeout_s` | s | 0.05 | 10 | reboot | Controller-heartbeat watchdog timeout before disarm [s]. |
| `g1_fx_m0` | — | -10000 | 10000 | live | G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values. |
| `g1_fy_m0` | — | -10000 | 10000 | live | G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values. |
| `g1_fz_m0` | — | -10000 | 10000 | live | G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values. |
| `g1_fx_m1` | — | -10000 | 10000 | live | G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values. |
| `g1_fy_m1` | — | -10000 | 10000 | live | G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values. |
| `g1_fz_m1` | — | -10000 | 10000 | live | G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values. |
| `g1_fx_m2` | — | -10000 | 10000 | live | G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values. |
| `g1_fy_m2` | — | -10000 | 10000 | live | G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values. |
| `g1_fz_m2` | — | -10000 | 10000 | live | G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values. |
| `g1_fx_m3` | — | -10000 | 10000 | live | G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values. |
| `g1_fy_m3` | — | -10000 | 10000 | live | G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values. |
| `g1_fz_m3` | — | -10000 | 10000 | live | G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values. |
| `g1_rr_m0` | — | -10000 | 10000 | live | G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values |
| `g1_rp_m0` | — | -10000 | 10000 | live | G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values |
| `g1_ry_m0` | — | -10000 | 10000 | live | G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values |
| `g1_rr_m1` | — | -10000 | 10000 | live | G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values |
| `g1_rp_m1` | — | -10000 | 10000 | live | G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values |
| `g1_ry_m1` | — | -10000 | 10000 | live | G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values |
| `g1_rr_m2` | — | -10000 | 10000 | live | G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values |
| `g1_rp_m2` | — | -10000 | 10000 | live | G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values |
| `g1_ry_m2` | — | -10000 | 10000 | live | G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values |
| `g1_rr_m3` | — | -10000 | 10000 | live | G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values |
| `g1_rp_m3` | — | -10000 | 10000 | live | G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values |
| `g1_ry_m3` | — | -10000 | 10000 | live | G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values |
| `indi_rate_r` | 1/s | 0 | 1000 | reboot | Rate error -> angular acceleration gains [roll, pitch, yaw] (rad/s^2 per rad/s). |
| `indi_rate_p` | 1/s | 0 | 1000 | reboot | Rate error -> angular acceleration gains [roll, pitch, yaw] (rad/s^2 per rad/s). |
| `indi_rate_y` | 1/s | 0 | 1000 | reboot | Rate error -> angular acceleration gains [roll, pitch, yaw] (rad/s^2 per rad/s). |
| `indi_sync_hz` | Hz | 1 | 500 | reboot | Biquad low-pass cutoff for synchronized filters (Hz). |
| `indi_ctrl_div` | — | 1 | 16 | reboot | INDI steps once per this many IMU samples (control rate = IMU ODR / this). |
| `wls_wv_fx` | — | 0.001 | 1000000 | reboot | WLS pseudo-control weights [fx, fy, fz, roll, pitch, yaw]. |
| `wls_wv_fy` | — | 0.001 | 1000000 | reboot | WLS pseudo-control weights [fx, fy, fz, roll, pitch, yaw]. |
| `wls_wv_fz` | — | 0.001 | 1000000 | reboot | WLS pseudo-control weights [fx, fy, fz, roll, pitch, yaw]. |
| `wls_wv_rr` | — | 0.001 | 1000000 | reboot | WLS pseudo-control weights [fx, fy, fz, roll, pitch, yaw]. |
| `wls_wv_rp` | — | 0.001 | 1000000 | reboot | WLS pseudo-control weights [fx, fy, fz, roll, pitch, yaw]. |
| `wls_wv_ry` | — | 0.001 | 1000000 | reboot | WLS pseudo-control weights [fx, fy, fz, roll, pitch, yaw]. |
| `wls_wu_m0` | — | 0.001 | 1000000 | reboot | WLS actuator penalty weights [m0, m1, m2, m3]. |
| `wls_wu_m1` | — | 0.001 | 1000000 | reboot | WLS actuator penalty weights [m0, m1, m2, m3]. |
| `wls_wu_m2` | — | 0.001 | 1000000 | reboot | WLS actuator penalty weights [m0, m1, m2, m3]. |
| `wls_wu_m3` | — | 0.001 | 1000000 | reboot | WLS actuator penalty weights [m0, m1, m2, m3]. |
| `indi_idle_norm` | — | 0 | 0.3 | reboot | Per-motor idle throttle (normalized 0..1) held while armed and |
| `indi_ground_gyro_dps` | deg/s | 1 | 2000 | reboot | Ground-contact detection: gyro magnitude below this counts as |
| `indi_ground_accel_g` | g | 0.1 | 2 | reboot | Ground-contact detection: specific-force magnitude within this |
| `indi_ground_thrust_sp` | — | 0 | 50 | reboot | Ground-contact detection: vertical thrust setpoint below this |
| `indi_rpm_stale_gaps` | — | 2 | 200 | reboot | RPM staleness gate: how many consecutive *expected* telemetry |
| `indi_omega_kf` | — | — | — | reboot | Source of the ω / ω̇ that INDI's incremental law consumes. |
| `indi_nan_rampdown` | — | 0.5 | 1 | reboot | Per-tick multiplier on the held actuator state while the WLS |
| `rpm_est_omega_var` | (rad/s)^2 | 0.01 | 1000000 | live | Measurement-noise variance on ω at/below 2000 rad/s [(rad/s)²]. |
| `rpm_est_thr_psd` | s | 0.000000001 | 1 | live | Intensity of the throttle-error process driving ω [s]. |
| `rpm_est_cm_psd` | — | 0.000001 | 10000 | live | Random-walk intensity on the estimated full-throttle speed `c_m` |
| `rpm_est_nis_gate` | — | 1 | 100 | live | Normalized-innovation-squared gate (χ², 1 DOF). Samples with |
| `rpm_est_tau_d` | s | 0 | 0.02 | live | Transport delay from commanding a throttle to it affecting ω [s]. |
| `rpm_est_omega_ref` | rad/s | 100 | 20000 | live | Reference speed [rad/s] at which `rpm_est_omega_var` is stated. |
| `rpm_est_init_omega_var` | (rad/s)^2 | 1 | 1000000 | live | Initial and re-seed variance on the ω state [(rad/s)²]. |
| `rpm_est_escape_rejects` | — | 1 | 1000 | live | Consecutive NIS rejections before ω is re-seeded from the |
| `rpm_est_plausible_frac` | — | 1 | 10 | live | Hard plausibility bound on a measurement, as a multiple of the |
| `rpm_notch_en` | — | — | — | reboot | Enable the RPM-notch path. The banks are always allocated |
| `rpm_notch_q` | — | 1 | 20 | reboot | Notch Q: higher = narrower = less off-band phase loss but worse |
| `rpm_notch_min_hz` | Hz | 20 | 500 | reboot | Below this motor frequency the notch fades to passthrough. |
| `rpm_notch_fade_hz` | Hz | 1 | 200 | reboot | Fade-in window above `min_hz`. |
| `rpm_notch_lpf_hz` | Hz | 20 | 500 | reboot | PT1 cutoff for the notch-frequency tracker (must track motor 1P |
| `pos_kp_x` | — | — | — | reboot | Position proportional gains [x, y, z]. |
| `pos_kp_y` | — | — | — | reboot | Position proportional gains [x, y, z]. |
| `pos_kp_z` | — | — | — | reboot | Position proportional gains [x, y, z]. |
| `pos_kd_x` | — | — | — | reboot | Position derivative (velocity) gains [x, y, z]. |
| `pos_kd_y` | — | — | — | reboot | Position derivative (velocity) gains [x, y, z]. |
| `pos_kd_z` | — | — | — | reboot | Position derivative (velocity) gains [x, y, z]. |
| `att_k_r` | — | — | — | reboot | Attitude error to body-rate gains [roll, pitch, yaw]. |
| `att_k_p` | — | — | — | reboot | Attitude error to body-rate gains [roll, pitch, yaw]. |
| `att_k_y` | — | — | — | reboot | Attitude error to body-rate gains [roll, pitch, yaw]. |
| `att_kt_r` | — | — | — | reboot | Body-rate error to torque gains [roll, pitch, yaw]. The second |
| `att_kt_p` | — | — | — | reboot | Body-rate error to torque gains [roll, pitch, yaw]. The second |
| `att_kt_y` | — | — | — | reboot | Body-rate error to torque gains [roll, pitch, yaw]. The second |
| `cascade_rate_hz` | Hz | 25 | 500 | reboot | Geometric-cascade outer-loop tick rate. |
| `pos_err_max_x` | m | 0.01 | 100 | reboot | Per-axis clamp on the position error the PD stage acts on [m]. |
| `pos_err_max_y` | m | 0.01 | 100 | reboot | Per-axis clamp on the position error the PD stage acts on [m]. |
| `pos_err_max_z` | m | 0.01 | 100 | reboot | Per-axis clamp on the position error the PD stage acts on [m]. |
| `vel_err_max_x` | m/s | 0.01 | 100 | reboot | Per-axis clamp on the velocity error the PD stage acts on [m/s]. |
| `vel_err_max_y` | m/s | 0.01 | 100 | reboot | Per-axis clamp on the velocity error the PD stage acts on [m/s]. |
| `vel_err_max_z` | m/s | 0.01 | 100 | reboot | Per-axis clamp on the velocity error the PD stage acts on [m/s]. |
| `cascade_odom_stale_s` | s | 0.005 | 1 | reboot | Oldest odometry [s] the cascade will accept as its state. |
| `mpc_w_pos_x` | — | 0 | 1000000 | live | Position tracking weights [x, y, z]. |
| `mpc_w_pos_y` | — | 0 | 1000000 | live | Position tracking weights [x, y, z]. |
| `mpc_w_pos_z` | — | 0 | 1000000 | live | Position tracking weights [x, y, z]. |
| `mpc_w_vel_x` | — | 0 | 1000000 | live | Velocity tracking weights [x, y, z]. |
| `mpc_w_vel_y` | — | 0 | 1000000 | live | Velocity tracking weights [x, y, z]. |
| `mpc_w_vel_z` | — | 0 | 1000000 | live | Velocity tracking weights [x, y, z]. |
| `mpc_w_att_r` | — | 0 | 1000000 | live | Attitude tracking weights [roll, pitch, yaw]. |
| `mpc_w_att_p` | — | 0 | 1000000 | live | Attitude tracking weights [roll, pitch, yaw]. |
| `mpc_w_att_y` | — | 0 | 1000000 | live | Attitude tracking weights [roll, pitch, yaw]. |
| `mpc_w_rate_r` | — | 0 | 1000000 | live | Body-rate tracking weights [roll, pitch, yaw]. |
| `mpc_w_rate_p` | — | 0 | 1000000 | live | Body-rate tracking weights [roll, pitch, yaw]. |
| `mpc_w_rate_y` | — | 0 | 1000000 | live | Body-rate tracking weights [roll, pitch, yaw]. |
| `mpc_w_thrust` | — | 0 | 1000000 | live | Control effort weight (uniform across motors). |
| `mpc_dt` | s | 0.005 | 0.5 | live | Integration timestep [s] for the prediction horizon. |
| `mpc_rho` | — | 0.001 | 1000000000 | live | Cubic constraint penalty weight (input bound enforcement). |
| `mpc_thrust_frac` | — | 0.1 | 1 | live | Fraction of the summed per-motor max thrust available as the |
| `mpc_drag_x` | — | 0 | 0.01 | live | Rotor-drag coefficients `c = −m·k` per body axis [N·s²/(m·rad)] |
| `mpc_drag_y` | — | 0 | 0.01 | live | Rotor-drag coefficients `c = −m·k` per body axis [N·s²/(m·rad)] |
| `mpc_drag_z` | — | 0 | 0.01 | live | Rotor-drag coefficients `c = −m·k` per body axis [N·s²/(m·rad)] |
| `mpc_bodydrag_x` | — | 0 | 1 | live | Quadratic body-drag coefficients `½ρC_dA` per body axis [N·s²/m²] |
| `mpc_bodydrag_y` | — | 0 | 1 | live | Quadratic body-drag coefficients `½ρC_dA` per body axis [N·s²/m²] |
| `mpc_bodydrag_z` | — | 0 | 1 | live | Quadratic body-drag coefficients `½ρC_dA` per body axis [N·s²/m²] |
| `mpc_learned_cost` | — | — | — | live | Run the baked situation-conditioned cost policy before every solve |
| `mpc_learned_gain` | — | 0 | 1 | live | Scale on the cost policy's output, `z_eff = gain·z` — the ramp-in |
| `mpc_pos_cost_mode` | — | — | — | live | Position-cost formulation (`mpc_pos_cost_mode`: 0 = Quadratic, |
| `mpc_rate_hz` | Hz | 25 | 200 | reboot | MPC outer-loop solve/tick rate, independent of the `mpc_dt` horizon spacing. |
| `mpc_max_iters` | — | 1 | 30 | live | SQP iterations per solve. 1 = RTI (real-time iteration): one |
| `mpc_horizon_n` | — | 1 | 20 | live | Active prediction-horizon length in stages (`mpc_dt` apart). The |
| `mpc_kkt_tol` | — | 0.000001 | 0.1 | live | SQP convergence tolerance on the KKT residual (max \|feedforward\| |
| `mpc_rate_barrier_tau` | — | 0 | 100 | live | Body-rate state-constraint barrier weight τ (`FullQuadModel` only — |
| `mpc_rate_barrier_delta` | rad/s | 0.01 | 2 | live | Relaxed-barrier margin δ [rad/s]: below this distance-to-bound the |
| `mpc_tilt_max_deg` | deg | 10 | 178 | live | Maximum-tilt state-constraint limit θ_max (both MPC models — |
| `mpc_tilt_barrier_tau` | — | 0 | 100 | live | Tilt-constraint barrier weight τ. **0 = tilt constraint off.** |
| `mpc_tilt_barrier_delta` | — | 0.005 | 0.5 | live | Relaxed-barrier margin δ for the tilt constraint, in **cos units** |
| `mpc_odom_stale_s` | s | 0.005 | 1 | live | Oldest odometry [s] the outer loop will accept as the MPC's |
| `mpc_u_ref_ff` | — | — | — | live | Bias each horizon step's input cost toward the trajectory's own |
| `plan_max_vel` | m/s | 0.1 | 200 | live |  |
| `plan_max_tilt` | rad | 0.05 | 1.55 | live |  |
| `plan_w_time` | — | 0 | 1000000 | live | Weight on total trajectory time Σ T_i (higher → faster trajectories). |
| `plan_w_energy` | — | 0 | 1000000 | live | Weight on energy (∫‖jerk‖² or ∫‖snap‖²) — controls smoothness. |
| `plan_w_vel` | — | 0 | 1000000 | live | Weight on velocity constraint penalty (soft ‖v‖ ≤ max_vel). |
| `plan_w_tilt` | — | 0 | 1000000 | live | Weight on tilt angle penalty. Set to 0 to disable. |
| `plan_w_body_rate` | — | 0 | 1000000 | live | Weight on body rate penalty. Set to 0 to disable. |
| `plan_w_thrust` | — | 0 | 1000000 | live | Weight on thrust constraint penalty (soft thrust bounds). |
| `plan_smooth_eps` | — | 0.000001 | 10 | live | Smoothing width ε of the smoothed-L1 penalty; must be > 0. |
| `plan_num_check` | — | 1 | 64 | live | Trapezoidal sub-intervals per piece for constraint evaluation (≥ 1). |
| `plan_dur_min_s` | s | 0.05 | 60 | live | Shortest trajectory [s] the planner will publish. |
| `plan_dur_max_s` | s | 1 | 3600 | live | Longest trajectory [s] the planner will publish. Belt-and-braces |
| `plan_waypoint_radius` | m | 0.0001 | 10 | live | Radius [m] of the ball each intermediate waypoint is pinned |
| `plan_bfgs_delta_init` | — | 0.0001 | 1000 | live | Initial trust-region radius (decision-vector units). |
| `plan_bfgs_delta_max` | — | 0.0001 | 10000 | live | Cap on the trust-region radius. |
| `plan_bfgs_eta` | — | 0 | 0.5 | live | Acceptance threshold: accept step if actual/predicted > eta. |
| `plan_bfgs_g_eps` | — | 0.0000000001 | 0.01 | live | Gradient convergence test: ‖g‖∞ / max(1, ‖x‖∞) < g_epsilon. |
| `plan_bfgs_max_iter` | — | 0 | 10000 | live | Outer-iteration cap. 0 disables the cap. |
| `plan_bfgs_past` | — | 0 | 15 | live | Cost-stagnation lookback in accepted iterations (0 disables; clamped to 15). |
| `plan_bfgs_delta_conv` | — | 0 | 0.01 | live | Relative cost change over `past` iterations that triggers `Stop`. |
| `plan_bfgs_delta_collapse` | — | 0.000000000001 | 0.01 | live | Trust radius below which the solve is declared collapsed and |
| `sampler_kind` | — | — | — | live | Reference-sampler selection (`sampler_kind`: 0 = Time, 1 = Position). |
| `sampler_max_lag_s` | s | 0 | 5 | live |  |
| `sampler_axis_weights_sqrt_x` | — | — | — | live |  |
| `sampler_axis_weights_sqrt_y` | — | — | — | live |  |
| `sampler_axis_weights_sqrt_z` | — | — | — | live |  |
| `sampler_search_dt` | s | 0.001 | 1 | live |  |
| `sampler_max_search_steps` | — | — | — | live |  |
| `sampler_radius_of_acceptance` | — | — | — | live |  |
| `sampler_max_lead_s` | s | 0 | 5 | live |  |
| `mission_profile` | — | — | — | live | Index into `cybflight::control::offline_mission::PROFILES` selecting |
| `arm_led` | — | — | — | live | External arm-LED enable. When true, the LED task drives the WS2812 |
| `blackbox_tier` | — | 0 | 4 | live | Blackbox record-set tier: 0=None, 1=Small, 2=Mid, 3=Large, |
| `blackbox_rate_div` | — | 1 | 64 | live | Blackbox rate divider for `/imu1_raw`: one sample in N is |
| `blackbox_mute_mask` | — | 0 | 131071 | live | Bitmask of muted blackbox topics, keyed by MCAP channel id |
| `peer_pose_en` | — | — | — | live | Leader→chaser pose downlink over the ESP32 bridge. Off by default: |
