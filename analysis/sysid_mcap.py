#!/usr/bin/env python3
"""System identification from a cybflight blackbox MCAP recording.

Port of the thrust/drag, actuator, and moments model fits from
optimal_quad_control_RL/analyze.py onto the cybflight blackbox wire
format (docs/blackbox.md). Three fits, each produced only when its
topics have data:

  1. **Thrust + drag model** (`--fit thrust`)
       az = k_w * sum(omega_i^2)                    (thrust)
       ax = k_x * vbx * sum(omega_i) + kq_x*vbx*|vbx|   (X drag: rotor + body)
       ay = k_y * vby * sum(omega_i) + kq_y*vby*|vby|   (Y drag)
     The quadratic `v|v|` column is the parasitic body drag that dominates
     above ~30 m/s; it is fitted jointly and reported as `mpc_bodydrag_*`
     (= -m*kq, N·s²/m²) next to the rotor term.
     Linear least squares. Needs /imu1 + /motor_state (+ /odometry
     for the drag fits — body velocity comes from rotating the ESKF
     world velocity into the body frame).

  2. **Actuator model** (`--fit actuator`)
       w_c   = (w_max - w_min) * sqrt(k*u^2 + (1-k)*u) + w_min
       dw/dt = (w_c - w) / tau
     Nonlinear fit (scipy minimize) per motor + one total fit.
     Needs /motors (commanded u, already normalized 0..1) +
     /motor_state (achieved omega).

  3. **Moments model** (`--fit moments`)
       dp = k_p1*w1^2 + k_p2*w2^2 + k_p3*w3^2 + k_p4*w4^2
       dq = k_q1*w1^2 + k_q2*w2^2 + k_q3*w3^2 + k_q4*w4^2
       dr = k_r*(s.w) + k_rd*(s.dw)     s = yaw spin-sign pattern
     Linear least squares; body rates from /imu1 gyro, rotor speeds
     and accelerations from /motor_state (KF-fused omega_dot — no
     numeric differentiation of eRPM needed).

Frame conventions (differ from the Betaflight/NED original):
cybflight logs ENU world / FLU body, z-up. Hover therefore shows
accel_z ~= +9.81 m/s^2 and the fitted k_w comes out POSITIVE
(negative in the FRD original); the drag k_x/k_y still come out
negative. Rotor speeds are rad/s as logged (no eRPM scaling), and
motor commands are already normalized (no motorOutput min/max
header dance).

All topics are resampled onto one uniform time grid (median
/motor_state rate) before fitting; by default the data is cropped
to the [first ARM .. last DISARM] window from /events. Rotor speeds
are taken from the RAW eRPM field of /motor_state (dropouts bridged)
rather than the KF-fused estimate, which diverges above its own
measurements in aggressive flight (--fused-omega restores it).

Record-set tier: mid carries everything these fits need at 100 Hz;
`blackbox set sysid` raises /motors + /motor_state to >=500 Hz,
which the actuator fit (tau ~ 10-30 ms) really wants.

Usage:
    python3 analysis/sysid_mcap.py logs/flight_0007.mcap
    python3 analysis/sysid_mcap.py f.mcap --fit actuator
    python3 analysis/sysid_mcap.py f.mcap --save out --no-show
    python3 analysis/sysid_mcap.py f.mcap --no-crop --t0 2 --t1 14
    python3 analysis/sysid_mcap.py f.mcap --from-mission --until-idle
    python3 analysis/sysid_mcap.py f.mcap --yaw-signs=1,-1,-1,1

Full operating guide + model derivations: docs/system_id.md.

Requires: pip install mcap cbor2 numpy scipy matplotlib
"""

import argparse
import collections
import json
import sys

try:
    import cbor2
    import numpy as np
    import scipy.signal
    from scipy.optimize import minimize
    from mcap.stream_reader import StreamReader
    from mcap import records as R
except ImportError:
    sys.exit("missing deps: pip install mcap cbor2 numpy scipy matplotlib")

ARM, DISARM = 0x01, 0x02              # topics::events::KIND_*
MISSION_EXECUTING, MISSION_IDLE = 0x0A, 0x0B


# ── MCAP loading (schema-driven, same shape as plot_mcap.py) ────────

def array_labels(schema_json):
    try:
        s = json.loads(schema_json)
        if s.get("type") == "array" and "prefixItems" in s:
            return [it.get("title", f"[{i}]") for i, it in enumerate(s["prefixItems"])]
    except (ValueError, AttributeError):
        pass
    return None


def load(path):
    """Return {topic: list-of-dicts} with positional arrays label-mapped."""
    schemas, chans = {}, {}
    out = collections.defaultdict(list)
    for rec in StreamReader(open(path, "rb")).records:
        if isinstance(rec, R.Schema):
            schemas[rec.id] = array_labels(rec.data)
        elif isinstance(rec, R.Channel):
            chans[rec.id] = (rec.topic, schemas.get(rec.schema_id))
        elif isinstance(rec, R.Message):
            topic, labels = chans.get(rec.channel_id, (None, None))
            if topic is None:
                continue
            obj = cbor2.loads(rec.data)
            if isinstance(obj, list) and labels:
                obj = dict(zip(labels, obj))
            if isinstance(obj, dict):
                obj.setdefault("timestamp_ns", rec.log_time)
                out[topic].append(obj)
    return out


def col(msgs, key):
    return np.array([m[key] for m in msgs], dtype=np.float64)


def vcol(msgs, key, width):
    """Nx`width` array from a per-message list field (None -> nan)."""
    return np.array(
        [[np.nan if v is None else v for v in m[key]] for m in msgs],
        dtype=np.float64,
    )


# ── quaternion: world (ENU) velocity -> body (FLU) velocity ─────────

def world_to_body(qw, qi, qj, qk, vx, vy, vz):
    """v_b = R(q)^T v_w, vectorized. q maps body->world ([w,i,j,k])."""
    n = np.sqrt(qw**2 + qi**2 + qj**2 + qk**2)
    n[n == 0] = 1.0
    w, x, y, z = qw / n, qi / n, qj / n, qk / n
    # rows of R^T = columns of R
    vbx = (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y + w * z) * vy + 2 * (x * z - w * y) * vz
    vby = 2 * (x * y - w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z + w * x) * vz
    vbz = 2 * (x * z + w * y) * vx + 2 * (y * z - w * x) * vy + (1 - 2 * (x * x + y * y)) * vz
    return vbx, vby, vbz


# ── resampling ──────────────────────────────────────────────────────

def dedup(t, *cols):
    """Sort by t, drop duplicate timestamps (np.interp wants monotonic)."""
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return (t[keep],) + tuple(c[order][keep] for c in cols)


def interp_cols(tq, t, arr2d):
    """Column-wise np.interp of an (N,k) array onto query times tq."""
    return np.stack([np.interp(tq, t, arr2d[:, i]) for i in range(arr2d.shape[1])], axis=1)


def build_dataset(data, imu_topic, no_crop, t0_arg, t1_arg, from_mission=False,
                  until_idle=False, omega_src="raw"):
    """Resample all sysid topics onto one uniform grid -> dict of arrays."""
    ms = data.get("/motor_state", [])
    if not ms:
        sys.exit("no /motor_state messages — sysid needs tier >= mid "
                 "(`blackbox set sysid` recommended)")
    t_ms = col(ms, "timestamp_ns") / 1e9
    omega = vcol(ms, "omega", 4)
    omega_dot = vcol(ms, "omega_dot", 4)
    raw = vcol(ms, "raw", 4) if "raw" in ms[0] else None
    if raw is not None:
        t_ms, omega, omega_dot, raw = dedup(t_ms, omega, omega_dot, raw)
    else:
        t_ms, omega, omega_dot = dedup(t_ms, omega, omega_dot)

    # crop window: [first ARM .. last DISARM] unless told otherwise
    lo, hi = t_ms[0], t_ms[-1]
    events = data.get("/events", [])
    arms = [e["timestamp_ns"] / 1e9 for e in events if e.get("kind") == ARM]
    disarms = [e["timestamp_ns"] / 1e9 for e in events if e.get("kind") == DISARM]
    if not no_crop and arms and disarms and max(disarms) > min(arms):
        lo, hi = min(arms), max(disarms)
        print(f"cropping to ARM..DISARM window: "
              f"+{lo - t_ms[0]:.2f}s .. +{hi - t_ms[0]:.2f}s (--no-crop to keep all)")
    if from_mission:
        # Tighten further to [first mission Executing .. last mission Idle]
        # from the outer loop's own state transitions, so the fit sees only
        # trajectory flight — no takeoff, station-keep, or landing.
        execs = [e["timestamp_ns"] / 1e9 for e in events
                 if e.get("kind") == MISSION_EXECUTING]
        idles = [e["timestamp_ns"] / 1e9 for e in events
                 if e.get("kind") == MISSION_IDLE]
        if not execs:
            sys.exit("--from-mission: no KIND_MISSION_EXECUTING event in /events "
                     "(did this flight run a mission? tier >= mid records /events)")
        lo = max(lo, min(execs))
        idles_after = [t for t in idles if t > lo]
        if idles_after:
            # --until-idle: first Idle after the start (one mission run,
            # matching plot_mcap.py); default: last Idle (every run in
            # the log, hover between runs included).
            hi = min(hi, min(idles_after) if until_idle else max(idles_after))
        elif until_idle:
            print("note: --until-idle: no MISSION_IDLE event after the start "
                  "— keeping window end")
        print(f"cropping to mission window (Executing..{'first ' if until_idle else ''}Idle): "
              f"+{lo - t_ms[0]:.2f}s .. +{hi - t_ms[0]:.2f}s"
              + ("" if idles_after else "  (no Idle event — keeping window end)"))
    t_start = t_ms[0]
    if t0_arg is not None:
        lo = max(lo, t_start + t0_arg)
    if t1_arg is not None:
        hi = min(hi, t_start + t1_arg)
    if hi <= lo:
        sys.exit("empty time window after cropping")

    # uniform grid at the median /motor_state rate
    dt = float(np.median(np.diff(t_ms)))
    t = np.arange(lo, hi, dt)
    d = {"t": t - t[0], "dt": dt, "fs": 1.0 / dt}
    print(f"time base: /motor_state @ {1/dt:.1f} Hz, "
          f"{len(t)} samples over {t[-1] - t[0]:.2f} s")

    if omega_src == "raw" and raw is not None and np.isfinite(raw).any():
        # Default: fit on the RAW eRPM measurement, not the KF-fused
        # `omega`. The fused value is the RPM Kalman filter's estimate,
        # which has been observed to diverge 2-4x ABOVE its own raw
        # measurements during aggressive flight (fused > 5000 rad/s for
        # ~90% of a time-optimal mission while raw never exceeded 3740) —
        # fitting on it rails the actuator fit at its bounds and inflates
        # every sum(omega)/sum(omega^2) regressor. Telemetry dropouts
        # (raw = None) are bridged by linear interpolation, and omega_dot
        # is the low-passed gradient of the bridged measurement (the KF's
        # omega_dot belongs to the diverged estimate).
        drop = 0.0
        for i in range(4):
            v = raw[:, i]
            ok = np.isfinite(v)
            drop += 1.0 - ok.mean()
            raw[:, i] = np.interp(t_ms, t_ms[ok], v[ok])
        d["omega"] = interp_cols(t, t_ms, raw)
        sos = scipy.signal.butter(2, min(40.0, 0.45 / dt), "low",
                                  fs=1.0 / dt, output="sos")
        d["omega_dot"] = scipy.signal.sosfiltfilt(
            np.asarray(sos), np.gradient(d["omega"], t, axis=0), axis=0)
        print(f"omega source: raw eRPM, {drop / 4 * 100:.1f}% dropouts "
              f"bridged (--fused-omega for the KF estimate)")
    else:
        if omega_src == "raw":
            print("note: no `raw` field in /motor_state — falling back to "
                  "the KF-fused omega (beware: it can diverge in "
                  "aggressive flight)")
        d["omega"] = interp_cols(t, t_ms, omega)
        d["omega_dot"] = interp_cols(t, t_ms, omega_dot)
    if not np.any(d["omega"]):
        print("WARNING: /motor_state omega is all-zero in this window — "
              "motors never spun (bench log?); every fit will return 0")

    mo = data.get("/motors", [])
    if mo:
        t_mo = col(mo, "timestamp_ns") / 1e9
        u = vcol(mo, "motor_commands", 4)
        t_mo, u = dedup(t_mo, u)
        d["u"] = interp_cols(t, t_mo, u)

    imu = data.get(imu_topic, [])
    if not imu:
        # The sysid tier mutes /imu1 (only /imu1_raw is written); Mid has
        # no /imu1_raw. Take whichever IMU stream the file has.
        other = "/imu1" if imu_topic == "/imu1_raw" else "/imu1_raw"
        if data.get(other):
            print(f"note: no {imu_topic} in this file — using {other}")
            imu_topic, imu = other, data[other]
    if imu:
        t_i = col(imu, "timestamp_ns") / 1e9
        acc = np.stack([col(imu, f"accel_{a}_m_s2") for a in "xyz"], axis=1)
        gyr = np.stack([col(imu, f"gyro_{a}_rad_s") for a in "xyz"], axis=1)
        t_i, acc, gyr = dedup(t_i, acc, gyr)
        d["acc"] = interp_cols(t, t_i, acc)          # ax, ay, az (FLU, m/s^2)
        d["gyro"] = interp_cols(t, t_i, gyr)         # p, q, r  (FLU, rad/s)
        d["imu_topic"] = imu_topic

    od = data.get("/odometry", [])
    if od:
        t_o = col(od, "timestamp_ns") / 1e9
        quat = np.stack([col(od, k) for k in ("q_w", "q_i", "q_j", "q_k")], axis=1)
        vel = np.stack([col(od, f"vel_{a}_m_s") for a in "xyz"], axis=1)
        t_o, quat, vel = dedup(t_o, quat, vel)
        q = interp_cols(t, t_o, quat)
        v = interp_cols(t, t_o, vel)
        vbx, vby, vbz = world_to_body(q[:, 0], q[:, 1], q[:, 2], q[:, 3],
                                      v[:, 0], v[:, 1], v[:, 2])
        d["vb"] = np.stack([vbx, vby, vbz], axis=1)
    return d


# ── fit 1: thrust + drag ────────────────────────────────────────────

def fit_thrust_drag_model(d, plt):
    """az = k_w*sum(w^2); ax = k_x*vbx*sum(w); ay = k_y*vby*sum(w)."""
    print("fitting thrust and drag model")
    if "acc" not in d:
        print("  skipped: no IMU data")
        return None, None
    w = d["omega"]
    sum_w2 = (w**2).sum(axis=1)
    sum_w = w.sum(axis=1)
    t = d["t"]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    # THRUST: az = k_w * sum(omega_i^2)
    X = np.stack([sum_w2])
    Y = d["acc"][:, 2]
    A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    k_w, = A
    axs[0].plot(t, Y, label="az")
    axs[0].plot(t, A @ X, label="T model")
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel("acc [m/s^2]")
    axs[0].legend()
    axs[0].set_title("Thrust model:\naz = k_w*sum(w_i^2)\nk_w = {:.4e}".format(k_w))

    # DRAG X / Y: a = k * vb * sum(omega_i) + kq * vb*|vb|   (needs body velocity)
    k_x = k_y = None
    kq_x = kq_y = None
    if "vb" in d:
        for i, (axis_name, vb_col, acc_col) in enumerate(
                (("X", 0, 0), ("Y", 1, 1)), start=1):
            vb = d["vb"][:, vb_col]
            X = np.stack([vb * sum_w, vb * np.abs(vb)])
            Y = d["acc"][:, acc_col]
            # Sign-constrained least squares: drag opposes velocity, so
            # both coefficients must be <= 0. The two regressors correlate
            # ~0.9 (both grow with |vb|), and below ~15 m/s the quadratic
            # term is so weakly excited that unconstrained lstsq can push
            # its coefficient across zero (an unphysical "anti-drag").
            # The bound makes non-identifiability land at exactly 0.
            from scipy.optimize import lsq_linear
            A = lsq_linear(X.T, Y, bounds=([-np.inf, -np.inf], [0.0, 0.0])).x
            k, kq = A
            if kq > -1e-9:
                kq = 0.0
            if kq == 0.0:
                print(f"  note: kq_{axis_name.lower()} pinned at 0 — the "
                      f"quadratic body-drag term is not identifiable from "
                      f"this data (needs sustained speed above ~15 m/s)")
            if axis_name == "X":
                k_x, kq_x = k, kq
            else:
                k_y, kq_y = k, kq
            axs[i].plot(t, Y, label=f"a{axis_name.lower()}")
            axs[i].plot(t, A @ X, label=f"D{axis_name.lower()} model")
            axs[i].set_xlabel("t [s]")
            axs[i].set_ylabel("acc [m/s^2]")
            axs[i].legend()
            axs[i].set_title(
                "Drag model {0}:\na{1} = k_{1}*vb{1}*sum(w_i) + kq_{1}*vb{1}|vb{1}|\n"
                "k_{1} = {2:.4e}, kq_{1} = {3:.4e}"
                .format(axis_name, axis_name.lower(), k, kq))
    else:
        for i in (1, 2):
            axs[i].set_title("drag fit skipped:\nno /odometry (body velocity)")
        print("  drag fits skipped: no /odometry messages")

    fig.suptitle("Thrust and Drag Model (FLU body frame: hover az ~ +9.81)")
    fig.tight_layout()
    return fig, (k_w, k_x, k_y, kq_x, kq_y)


# ── fit 2: actuator ─────────────────────────────────────────────────

def get_w_est(params, u, w0, dt):
    """Propagate dw/dt = (w_c - w)/tau on a uniform grid (lfilter)."""
    w_min, w_max, k, tau_inv = params
    w_c = (w_max - w_min) * np.sqrt(np.clip(k * u**2 + (1 - k) * u, 0, None)) + w_min
    a = dt * tau_inv
    # w_est[i] = w_est[i-1] + (w_c[i] - w_est[i-1])*a  ==  IIR: b=[a], a=[1, a-1]
    zi = np.array([w0 - a * w_c[0]])
    w_est, _ = scipy.signal.lfilter([a], [1.0, a - 1.0], w_c, zi=zi)
    return w_est


def fit_actuator_model(d, plt):
    """w_c = (w_max-w_min)*sqrt(k*u^2+(1-k)*u) + w_min; dw/dt = (w_c-w)/tau."""
    if "u" not in d:
        print("actuator fit skipped: no /motors data")
        return None, None, None
    print("fitting actuator model...")
    dt = d["dt"]
    u99 = float(np.percentile(d["u"], 99))
    print(f"  throttle coverage: p50 {float(np.percentile(d['u'], 50)):.2f}, "
          f"p99 {u99:.2f}, max {float(d['u'].max()):.2f}")
    if u99 < 0.85:
        print("  WARNING: throttle rarely exceeds "
              f"{u99:.2f} — w_max and the curve shape k are extrapolated "
              "beyond the data; fit an aggressive (full-throttle) log, or "
              "drop --until-idle/--from-mission so takeoff covers more range")

    def err(params, i):
        u, w = d["u"][:, i], d["omega"][:, i]
        return np.linalg.norm(get_w_est(params, u, w[0], dt) - w)

    initial_guess = [285, 2700, 0.75, 100]          # w_min, w_max, k, tau_inv
    bounds = [(0, 1000), (0, 6000), (0, 1), (1, 1000.0)]

    res = [minimize(lambda x, i=i: err(x, i), initial_guess, bounds=bounds)
           for i in range(4)]
    res_tot = minimize(lambda x: sum(err(x, i) for i in range(4)),
                       initial_guess, bounds=bounds)
    names = ("w_min", "w_max", "k", "1/tau")
    for label, x in [("total", res_tot.x)] + [
            (f"motor {i}", res[i].x) for i in range(4)]:
        hit = [names[j] for j in range(4)
               if x[j] >= bounds[j][1] - 1e-9 * max(1.0, bounds[j][1])
               or x[j] <= bounds[j][0] + 1e-12]
        if hit:
            print(f"  WARNING: {label} fit railed at the optimizer bound for "
                  f"{', '.join(hit)} — those values are NOT identified "
                  f"(k = 1 alone just means a pure-quadratic throttle curve)")

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for i, ax in enumerate(axs.flat):
        u, w = d["u"][:, i], d["omega"][:, i]
        ax.plot(d["t"], w, label="w")
        ax.plot(d["t"], get_w_est(res[i].x, u, w[0], dt), label="w est")
        ax.plot(d["t"], get_w_est(res_tot.x, u, w[0], dt), label="w est tot")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("w [rad/s]")
        ax.legend()
        p = res[i].x.copy()
        p[3] = 1 / p[3]
        ax.set_title("Motor {}: w_min = {:.2f}, w_max = {:.2f}, "
                     "k = {:.2f}, tau = {:.4f}".format(i + 1, *p))
    p = res_tot.x.copy()
    p[3] = 1 / p[3]
    fig.suptitle("Actuator model:\n"
                 "dw/dt = ((w_max-w_min)*sqrt(k*u^2 + (1-k)*u) + w_min - w)/tau\n"
                 "Total fit: w_min = {:.2f}, w_max = {:.2f}, k = {:.2f}, "
                 "tau = {:.4f}".format(*p))
    fig.tight_layout()
    return fig, res_tot.x, [r.x for r in res]


# ── fit 3: moments ──────────────────────────────────────────────────

def fit_moments_model(d, plt, yaw_signs, cutoff=64.0):
    """dp/dq from per-motor w^2; dr from spin-signed sum(w) + sum(dw)."""
    print("fitting moments model")
    if "gyro" not in d:
        print("  skipped: no IMU data")
        return None, None
    t, fs = d["t"], d["fs"]
    w, dw = d["omega"], d["omega_dot"]

    # body angular acceleration: filtered gradient of the gyro rates
    cutoff = min(cutoff, 0.45 * fs)  # keep the filter valid at 100 Hz logs
    sos = scipy.signal.butter(2, cutoff, "low", fs=fs, output="sos")
    dpqr = [scipy.signal.sosfiltfilt(sos, np.gradient(d["gyro"][:, i], t))
            for i in range(3)]

    s = np.asarray(yaw_signs, dtype=np.float64)
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    results = {}

    # dp, dq: per-motor omega^2
    for row, name in ((0, "p"), (1, "q")):
        X = np.stack([w[:, i]**2 for i in range(4)])
        Y = dpqr[row]
        A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
        results[f"k_{name}"] = A
        fit = A @ X
        axs[row, 0].plot(t, Y, label=f"d{name}")
        axs[row, 0].plot(t, fit, label=f"d{name} fit")
        axs[row, 0].set_ylabel(f"d{name} [rad/s^2]")
        axs[row, 0].legend()
        axs[row, 0].set_title(
            "d{0} = k_{0}1*w1^2 + k_{0}2*w2^2 + k_{0}3*w3^2 + k_{0}4*w4^2\n"
            "k_{0}1..4 = {1:.2e}, {2:.2e}, {3:.2e}, {4:.2e}".format(name, *A))
        axs[row, 1].plot(t, fit - Y, label=f"d{name} fit error")
        axs[row, 1].set_ylabel(f"d{name} err [rad/s^2]")
        axs[row, 1].legend()

    # dr: yaw from spin-signed rotor speed + rotor acceleration
    X = np.stack([(s * w).sum(axis=1), (s * dw).sum(axis=1)])
    Y = dpqr[2]
    A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    k_r, k_rd = A
    results["k_r"], results["k_rd"] = k_r, k_rd
    fit = A @ X
    sign_str = ",".join(f"{int(x):+d}" for x in s)
    axs[2, 0].plot(t, Y, label="dr")
    axs[2, 0].plot(t, fit, label="dr fit")
    axs[2, 0].set_ylabel("dr [rad/s^2]")
    axs[2, 0].set_xlabel("t [s]")
    axs[2, 0].legend()
    axs[2, 0].set_title("dr = k_r*sum(s_i*w_i) + k_rd*sum(s_i*dw_i)   s = [{}]\n"
                        "k_r, k_rd = {:.2e}, {:.2e}".format(sign_str, k_r, k_rd))
    axs[2, 1].plot(t, fit - Y, label="dr fit error")
    axs[2, 1].set_ylabel("dr err [rad/s^2]")
    axs[2, 1].set_xlabel("t [s]")
    axs[2, 1].legend()

    fig.suptitle("Moments Model (angular accel: {:.0f} Hz zero-phase LP "
                 "of gyro gradient)".format(cutoff))
    fig.tight_layout()
    return fig, results


# ── fit 4: diagonal inertia (Euler-equation fit) ────────────────────

def seam_mask(d, pad_s=0.5):
    """True away from pooled-file boundaries (gradients/filters ring there)."""
    m = np.ones(len(d["t"]), bool)
    k = int(pad_s * d["fs"])
    for s0 in d.get("seams", ()):
        m[max(0, int(s0) - k):int(s0) + k] = False
    return m


def fit_inertia_model(d, veh, plt, cutoff=64.0, rotor_inertia=None):
    """Diagonal inertia from Euler's rigid-body equations with the motor
    geometry IMPOSED, including the rotor angular momentum h:

        I*dw/dt + w x (I*w + h) + dh/dt = tau,   h_z = -J_r*sum(s_i*w_i)

    Per-motor thrusts are split by measured rotor speeds but SCALED BY THE
    ACCELEROMETER:  T_i = m * az * omega_i^2 / sum_j(omega_j^2).  m*az is
    the total thrust the IMU actually measured, so the absolute torque
    scale comes from the calibrated accelerometer; the omega^2 ratio only
    distributes it between motors, cancelling any common thrust-coefficient
    error or inflow-dependent c_T droop (the effect that makes a global
    az ~ k_w*sum(w^2) least squares regime-dependent).

    STAGE 1 — roll/pitch rows, unknowns [Ixx, Iyy, Izz, J_r, y_cg, x_cg]
    (sw = sum(s_i*w_i), sumT = sum(T_i)):

      x: Ixx*dp    - Iyy*(q*r) + Izz*(q*r) - J_r*(q*sw) + y_cg*sumT = sum(y_i*T_i)
      y: Ixx*(p*r) + Iyy*dq    - Izz*(p*r) + J_r*(p*sw) - x_cg*sumT = -sum(x_i*T_i)

    Izz enters via the gyroscopic coupling terms, J_r via rotor-precession
    torque, and the CoG columns absorb the roll/pitch torque a
    center-of-gravity offset produces from collective thrust. The yaw axis
    is deliberately NOT in stage 1: its torque model would need the
    uncertain torque_coeff_m, and letting it in shrinks Izz toward zero.

    STAGE 2 — yaw row alone, which from flight data yields only RATIOS:

      dr = (c_m/Izz)*sum(s*T) + (J_r/Izz)*sum(s*dw) + ((Ixx-Iyy)/Izz)*p*q

    On this vehicle the J_r*sum(s*dw) rotor-reaction term dominates yaw, so
    J_r/Izz is the reliable output (c_m/Izz is collinear with it — do not
    trust it). Absolute Izz is then closed, in order of preference, from:
    stage-1 gyroscopic coupling (if it passes its significance gate),
    J_r(stage 1)/(J_r/Izz), --rotor-inertia/(J_r/Izz), or the
    perpendicular-axis estimate Ixx+Iyy — the source is reported.

    Both sides of every equation pass through the same zero-phase low-pass
    (<= 30 Hz: the tau <-> gyro-derivative coherence peaks at 10-20 Hz,
    above that the raw gyro is rotor vibration); a +-30 ms shift scan
    aligns /motor_state against the IMU. Errors-in-variables: the noisy
    gyro-derivative regressor biases the forward estimate LOW, the reverse
    regression biases HIGH — both are printed as a bracket. Needs
    aggressive flight: hover carries no information about inertia.
    """
    print("fitting diagonal inertia (Euler-equation fit)")
    if veh is None:
        print("  skipped: needs --vehicle (mass + motor geometry)")
        return None, None
    if "gyro" not in d or "acc" not in d:
        print("  skipped: no IMU data")
        return None, None
    m = veh["mass"]
    w, dw = d["omega"], d["omega_dot"]
    t, fs = d["t"], d["fs"]

    sum_w2 = (w**2).sum(axis=1)
    frac = w**2 / np.where(sum_w2 > 1e4, sum_w2, np.inf)[:, None]
    T = m * np.clip(d["acc"][:, 2], 0.0, None)[:, None] * frac   # [N]
    xs = np.array([mo["pos_m"][0] for mo in veh["motors"]])
    ys = np.array([mo["pos_m"][1] for mo in veh["motors"]])
    sgn = np.array(veh["spin"])
    c_m = float(np.mean([mo["torque_coeff_m"] for mo in veh["motors"]]))
    tau = np.stack([T @ ys, -(T @ xs), T @ sgn], axis=1)  # [:,2] = sum(s*T)
    sum_t = T.sum(axis=1)
    sw = w @ sgn
    sdw = dw @ sgn

    co = min(cutoff, 30.0, 0.45 * fs)
    sos = scipy.signal.butter(2, co, "low", fs=fs, output="sos")
    lp = lambda x: scipy.signal.sosfiltfilt(sos, x, axis=0)
    dpqr = lp(np.stack([np.gradient(d["gyro"][:, i], t) for i in range(3)],
                       axis=1))
    p, q, r_ = d["gyro"].T
    qr_f, pr_f, pq_f = lp(q * r_), lp(p * r_), lp(p * q)
    base = seam_mask(d)
    max_shift = int(round(0.030 * fs))

    def stage1(shift):
        msk = base.copy()
        if shift > 0:
            msk[:shift] = False
        elif shift < 0:
            msk[shift:] = False
        tau_s = lp(np.roll(tau, shift, axis=0))[msk]
        st = lp(np.roll(sum_t, shift))[msk]
        sw_s = np.roll(sw, shift)
        qsw = lp(q * sw_s)[msk]
        psw = lp(p * sw_s)[msk]
        n = int(msk.sum())
        Z = np.zeros(n)
        dp, dq = dpqr[msk, 0], dpqr[msk, 1]
        qr, pr = qr_f[msk], pr_f[msk]
        A = np.concatenate([
            np.stack([dp, -qr, qr, -qsw, st, Z], axis=1),
            np.stack([pr, dq, -pr, psw, Z, -st], axis=1)])
        b = np.concatenate([tau_s[:, 0], tau_s[:, 1]])
        th = np.linalg.lstsq(A, b, rcond=None)[0]
        r2 = []
        for k in range(2):
            e = A[k * n:(k + 1) * n] @ th - b[k * n:(k + 1) * n]
            bb = b[k * n:(k + 1) * n]
            r2.append(1.0 - float((e**2).sum() / ((bb - bb.mean())**2).sum()))
        return th, r2, A, b, msk

    best = max(range(-max_shift, max_shift + 1),
               key=lambda sh: sum(stage1(sh)[1]))
    th, r2, A, b, msk = stage1(best)
    ixx, iyy, izz_c, jr1, ycg, xcg = (float(v) for v in th)
    dof = len(b) - A.shape[1]
    s2 = float(((A @ th - b)**2).sum()) / dof
    se = np.sqrt(s2 * np.diag(np.linalg.inv(A.T @ A)))
    cond = float(np.linalg.cond(A / np.linalg.norm(A, axis=0)))

    n = int(msk.sum())

    def eiv_bracket(y_col, others, rhs):
        Q = np.linalg.qr(others)[0]
        u = y_col - Q @ (Q.T @ y_col)
        v = rhs - Q @ (Q.T @ rhs)
        return float(u @ v / (u @ u)), float(v @ v / (u @ v))

    st1 = lp(np.roll(sum_t, best))[msk]
    ixx_lo, ixx_hi = eiv_bracket(
        dpqr[msk, 0], np.stack([qr_f[msk], st1], axis=1), b[:n])
    iyy_lo, iyy_hi = eiv_bracket(
        dpqr[msk, 1], np.stack([pr_f[msk], st1], axis=1), b[n:])

    # stage 2: yaw row -> ratios
    dr = dpqr[msk, 2]
    A2 = np.stack([lp(np.roll(tau[:, 2], best))[msk],
                   lp(np.roll(sdw, best))[msk], pq_f[msk]], axis=1)
    th2 = np.linalg.lstsq(A2, dr, rcond=None)[0]
    cm_over_izz, jr_over_izz, dii_over_izz = (float(v) for v in th2)
    e2 = A2 @ th2 - dr
    r2_yaw = 1.0 - float((e2**2).sum() / ((dr - dr.mean())**2).sum())

    # close absolute Izz, best available route first
    izz, izz_src = None, None
    if izz_c > 0 and se[2] < 0.5 * izz_c:
        izz, izz_src = izz_c, "roll/pitch gyroscopic coupling"
    elif jr1 > 0 and se[3] < 0.5 * jr1 and jr_over_izz > 0:
        izz, izz_src = jr1 / jr_over_izz, "J_r(precession) / (J_r/Izz)(yaw)"
    elif rotor_inertia is not None and jr_over_izz > 0:
        izz, izz_src = rotor_inertia / jr_over_izz, "--rotor-inertia / (J_r/Izz)(yaw)"
    else:
        izz, izz_src = ixx + iyy, "perpendicular-axis Ixx+Iyy (planar approx)"
    jr = jr_over_izz * izz

    print(f"  low-pass (both sides): {co:.0f} Hz; "
          f"/motor_state-vs-IMU shift: {best / fs * 1e3:+.1f} ms "
          f"(scanned ±{max_shift / fs * 1e3:.0f} ms)")
    print(f"  Ixx = {ixx:.4e} ± {se[0]:.1e}  (EIV bracket "
          f"{ixx_lo:.3e} .. {ixx_hi:.3e})")
    print(f"  Iyy = {iyy:.4e} ± {se[1]:.1e}  (EIV bracket "
          f"{iyy_lo:.3e} .. {iyy_hi:.3e})")
    print(f"  Izz = {izz:.4e}  [source: {izz_src}]")
    print(f"    coupling route: {izz_c:.3e} ± {se[2]:.1e}"
          f"{'' if izz_src.startswith('roll') else '  (failed gate)'}; "
          f"precession J_r = {jr1:.3e} ± {se[3]:.1e}")
    print(f"  CoG offset: x = {xcg * 1e3:+.1f} mm, y = {ycg * 1e3:+.1f} mm; "
          f"R² roll/pitch = {r2[0]:.3f}/{r2[1]:.3f}, cond = {cond:.1f}")
    print(f"  yaw ratios: J_r/Izz = {jr_over_izz:.3e} (dominant, R² = "
          f"{r2_yaw:.3f}); c_m/Izz = {cm_over_izz:.3f} and "
          f"(Ixx-Iyy)/Izz = {dii_over_izz:.3f} are collinear — indicative only")
    print(f"  implied J_r = {jr:.3e} kg·m² "
          f"(cf. YAML c_m {c_m:.3g} -> Izz {c_m / cm_over_izz:.3e} — "
          f"untrustworthy, see above)")

    tm = t[msk]
    pred = A @ th
    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for k, name in enumerate(("roll", "pitch")):
        axs[k].plot(tm, b[k * n:(k + 1) * n], lw=0.6,
                    label=f"tau_{name} (motors, accel-scaled)")
        axs[k].plot(tm, pred[k * n:(k + 1) * n], lw=0.6,
                    label=f"I*dw + w x (Iw+h) (fit, R²={r2[k]:.2f})")
        axs[k].set_ylabel("torque [N·m]")
        axs[k].legend(loc="upper right", fontsize=8)
    axs[2].plot(tm, dr, lw=0.6, label="dr (gyro)")
    axs[2].plot(tm, A2 @ th2, lw=0.6, label=f"yaw ratio fit (R²={r2_yaw:.2f})")
    axs[2].set_ylabel("dr [rad/s²]")
    axs[2].legend(loc="upper right", fontsize=8)
    axs[2].set_xlabel("t [s]")
    fig.suptitle("Euler-equation inertia fit: "
                 f"I = diag({ixx:.3e}, {iyy:.3e}, {izz:.3e}) kg·m² "
                 f"(Izz: {izz_src}), CoG ({xcg * 1e3:+.1f}, {ycg * 1e3:+.1f}) mm, "
                 f"shift {best / fs * 1e3:+.1f} ms")
    fig.tight_layout()
    return fig, {"ixx": ixx, "iyy": iyy, "izz": izz, "izz_src": izz_src,
                 "izz_coupling": izz_c, "jr": jr, "jr_precession": jr1,
                 "jr_over_izz": jr_over_izz, "xcg": xcg, "ycg": ycg,
                 "r2": r2 + [r2_yaw], "se": list(se),
                 "shift_ms": best / fs * 1e3, "cond": cond,
                 "ixx_bracket": (ixx_lo, ixx_hi),
                 "iyy_bracket": (iyy_lo, iyy_hi)}

# ── vehicle YAML: unit conversion into firmware / sim keys ─────────

def load_vehicle(path):
    """mass, inertia diagonal, and per-motor spin signs from a vehicle YAML."""
    try:
        import yaml
    except ImportError:
        sys.exit("--vehicle needs pyyaml: pip install pyyaml")
    with open(path) as f:
        v = yaml.safe_load(f)
    af = v["airframe"]
    inertia = af["inertia_kg_m2"]
    spin = [1.0 if m["spin"].lower() == "cw" else -1.0 for m in af["motors"]]
    if len(spin) != 4:
        sys.exit(f"{path}: expected 4 motors, found {len(spin)}")
    for mo in af["motors"]:
        mo.setdefault("torque_coeff_m", 0.0)
    return {"mass": float(af["mass_kg"]),
            # row-major 3x3 tensor -> diagonal [Ixx, Iyy, Izz]
            "inertia": [float(inertia[0]), float(inertia[4]), float(inertia[8])],
            "inertia_full": [float(v) for v in inertia],
            "spin": spin,
            "motors": af["motors"]}


def identify_inertia(veh, r, omega_mean, rotor_inertia=None):
    """Ixx/Iyy/Izz that reproduce the identified effectiveness through the
    firmware's *geometric* G1 (pos_m, max_thrust_n, torque_coeff_m,
    inertia_kg_m2) — docs/system_id.md 4.2. Returns None if the thrust or
    moments fit is missing."""
    if not ("k_p" in r and "k_w" in r and r["k_w"] > 0):
        return None
    m = veh["mass"]
    ixx_e = [m * r["k_w"] * mo["pos_m"][1] / r["k_p"][i]
             for i, mo in enumerate(veh["motors"]) if r["k_p"][i] != 0]
    iyy_e = [-m * r["k_w"] * mo["pos_m"][0] / r["k_q"][i]
             for i, mo in enumerate(veh["motors"]) if r["k_q"][i] != 0]
    if not (ixx_e and iyy_e):
        return None
    out = {"ixx_per_motor": ixx_e, "iyy_per_motor": iyy_e,
           "ixx": float(np.mean(ixx_e)), "iyy": float(np.mean(iyy_e))}
    # Izz closes through the YAML's torque_coeff_m (the extra constraint
    # the current parameter structure already carries): geometric yaw
    # column s*c_m*T/Izz must equal the identified s*k_r*w_max^2/omega_mean.
    c_m = float(np.mean([mo["torque_coeff_m"] for mo in veh["motors"]]))
    if "k_r" in r and r["k_r"] != 0 and omega_mean > 0:
        out["izz"] = c_m * m * r["k_w"] * omega_mean / r["k_r"]
        out["izz_source"] = f"torque_coeff_m={c_m:.4g} (YAML) * m*k_w*omega_mean/k_r"
    out["izz_perp_axis"] = out["ixx"] + out["iyy"]
    if rotor_inertia is not None and r.get("k_rd"):
        out["izz_from_jr"] = rotor_inertia / r["k_rd"]
    return out


def print_yaml_block(veh, r, omega_mean, g1_override=False, rotor_inertia=None):
    """Paste-ready YAML: identified values converted per docs/system_id.md 4.1."""
    m, (ixx, iyy, izz) = veh["mass"], veh["inertia"]
    inert = identify_inertia(veh, r, omega_mean, rotor_inertia)
    izz_used = inert["izz"] if inert and "izz" in inert else izz
    print("\n── vehicle YAML (see docs/system_id.md section 4.1) ──")
    print(f"# identified from blackbox; mean rotor speed {omega_mean:.0f} rad/s")
    diag = r.get("inertia_diag")
    if diag:
        print(f"# identified inertia diag (Euler-equation fit) [Ixx, Iyy, Izz] = "
              f"[{diag[0]:.4g}, {diag[1]:.4g}, {diag[2]:.4g}] kg·m²"
              f"   (YAML: [{ixx:.4g}, {iyy:.4g}, {izz:.4g}]; "
              f"Izz via {r.get('izz_src', '?')})")
    elif inert:
        print(f"# identified inertia diag (effectiveness route) [Ixx, Iyy, Izz] = "
              f"[{inert['ixx']:.4g}, {inert['iyy']:.4g}, {izz_used:.4g}] kg·m²"
              f"   (YAML: [{ixx:.4g}, {iyy:.4g}, {izz:.4g}])")
    if "k_w" in r and "w_max" in r:
        print("airframe:")
        if inert:
            fmt = lambda v: ", ".join(f"{x:.4g}" for x in v)
            print(f"  # Ixx per motor: [{fmt(inert['ixx_per_motor'])}]   YAML: {ixx:.4g}")
            print(f"  # Iyy per motor: [{fmt(inert['iyy_per_motor'])}]   YAML: {iyy:.4g}")
            if "izz" in inert:
                print(f"  # Izz = {inert['izz_source']}   YAML: {izz:.4g}")
                print(f"  #     (re-derive if torque_coeff_m changes; cross-checks: "
                      f"Ixx+Iyy = {inert['izz_perp_axis']:.4g}"
                      + (f", J_r/k_rd = {inert['izz_from_jr']:.4g}"
                         if "izz_from_jr" in inert else "") + ")")
            else:
                print(f"  # Izz: no yaw fit — keeping YAML {izz:.4g} "
                      f"(cross-check Ixx+Iyy = {inert['izz_perp_axis']:.4g})")
            print(f"  inertia_kg_m2: [{inert['ixx']:.4g}, 0.0, 0.0, 0.0, "
                  f"{inert['iyy']:.4g}, 0.0, 0.0, 0.0, {izz_used:.4g}]"
                  + ("   # effectiveness route — prefer the Euler fit below"
                     if diag else ""))
        if diag:
            print(f"  inertia_kg_m2: [{diag[0]:.4g}, 0.0, 0.0, 0.0, {diag[1]:.4g}, "
                  f"0.0, 0.0, 0.0, {diag[2]:.4g}]   # Euler-equation fit (--fit inertia)")
        print("  motors:   # max_thrust_n = m*k_w*w_max^2 ; torque_coeff_m kept "
              "from YAML (Izz above absorbs the yaw fit)")
        for i, mo in enumerate(veh["motors"]):
            w_max_i = r["per_motor"][i][1] if r.get("per_motor") else r["w_max"]
            t_max = m * r["k_w"] * w_max_i ** 2
            print(f"    - {{ pos_m: {mo['pos_m']}, spin: {mo['spin']}, "
                  f"max_thrust_n: {t_max:.3f}, torque_coeff_m: {mo['torque_coeff_m']} }}")
    print("tuning:")
    if "tau" in r:
        pm = r.get("per_motor") or [[r["w_min"], r["w_max"], r["k"], 1 / r["tau"]]] * 4
        for i in range(4):
            print(f"  m{i}_tau: {1 / pm[i][3]:.4f}")
        for i in range(4):
            print(f"  m{i}_omega_max: {pm[i][1]:.0f}")
        for i in range(4):
            print(f"  m{i}_nonlin: {r['k']:.3f}")
    if "k_rd" in r:
        for i, s_i in enumerate(veh["spin"]):
            print(f"  m{i}_g2_ry: {s_i * r['k_rd']:.4e}")
    if g1_override and "k_p" in r and "w_max" in r:
        # G1 override — inertia-free alternative (docs/system_id.md 4.2b).
        # Units: rad/s^2 (torque rows) and m/s^2 (force rows) per unit throttle.
        print("  # G1 override (--g1-override), inertia-free: rr=k_p*w_max^2 rp=k_q*w_max^2 "
              "ry=s*k_r*w_max^2/omega_mean fz=k_w*w_max^2")
        for i, s_i in enumerate(veh["spin"]):
            w_max_i = r["per_motor"][i][1] if r.get("per_motor") else r["w_max"]
            fz = r["k_w"] * w_max_i ** 2 if "k_w" in r else 0.0
            ry = (s_i * r["k_r"] * w_max_i ** 2 / omega_mean
                  if "k_r" in r and omega_mean > 0 else 0.0)
            print(f"  g1_fx_m{i}: 0\n  g1_fy_m{i}: 0\n  g1_fz_m{i}: {fz:.3f}")
            print(f"  g1_rr_m{i}: {r['k_p'][i] * w_max_i**2:.2f}\n"
                  f"  g1_rp_m{i}: {r['k_q'][i] * w_max_i**2:.2f}\n"
                  f"  g1_ry_m{i}: {ry:.3f}")
    if "k_x" in r and r["k_x"] is not None:
        print("  # ── mpc — drag in the prediction model (same fit as sim: below) ──")
        print(f"  mpc_drag_x: {-m * r['k_x']:.4e}")
        print(f"  mpc_drag_y: {-m * r['k_y']:.4e}")
        print("  mpc_drag_z: 0.0        # vertical drag is not identified")
        print(f"  mpc_bodydrag_x: {-m * r['kq_x']:.4e}")
        print(f"  mpc_bodydrag_y: {-m * r['kq_y']:.4e}")
        print("  mpc_bodydrag_z: 0.0")
    print("sim:")
    if "k_x" in r and r["k_x"] is not None:
        print(f"  aero_drag: [{-m * r['k_x']:.4e}, {-m * r['k_y']:.4e}, 0.0]")
        print(f"  body_drag: [{-m * r['kq_x']:.4e}, {-m * r['kq_y']:.4e}, 0.0]")
    if "w_min" in r:
        print(f"  rotor_omega_min_rad_s: {r['w_min']:.1f}")
        print(f"  rotor_throttle_curve_k: {r['k']:.3f}")
    if "k_rd" in r:
        print(f"  rotor_inertia_kg_m2: {izz_used * r['k_rd']:.4e}")


# ── main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[2:]),
    )
    ap.add_argument("path", nargs="+",
                    help="path(s) to cybflight .mcap recordings; several files are "
                         "cropped independently (ARM/mission windows per file) and "
                         "concatenated into one pooled fit — the recommended way to "
                         "identify drag from a set of missions with different tracks "
                         "and speeds")
    ap.add_argument("--fit",
                    choices=["thrust", "actuator", "moments", "inertia", "all"],
                    default="all", help="which model(s) to fit (default: all; "
                    "inertia = Euler-equation diagonal-inertia fit, needs --vehicle)")
    ap.add_argument("--fused-omega", action="store_true",
                    help="fit on the KF-fused /motor_state omega instead of the "
                         "raw eRPM measurement (the pre-2026-09 behavior; the "
                         "fused estimate can diverge far above raw during "
                         "aggressive flight — see docs/system_id.md)")
    ap.add_argument("--raw", action="store_true",
                    help="use /imu1_raw (pre-filter) instead of /imu1; either falls back "
                         "to the other if absent (sysid tier writes only /imu1_raw)")
    ap.add_argument("--no-crop", action="store_true",
                    help="do not crop to the ARM..DISARM window")
    ap.add_argument("--from-mission", action="store_true",
                    help="crop to the mission window [first Executing .. last Idle] "
                         "from /events (KIND_MISSION_EXECUTING/IDLE) — fit only "
                         "trajectory flight, excluding takeoff/hover/landing; "
                         "combines with --t0/--t1 (still relative to the first sample)")
    ap.add_argument("--until-idle", action="store_true",
                    help="with --from-mission: end the window at the FIRST "
                         "MISSION_IDLE event after the mission start (one "
                         "mission run, as in plot_mcap.py) instead of the "
                         "last one (all runs in the log)")
    ap.add_argument("--t0", type=float, default=None, metavar="S",
                    help="start of fit window, seconds from first sample")
    ap.add_argument("--t1", type=float, default=None, metavar="S",
                    help="end of fit window, seconds from first sample")
    ap.add_argument("--yaw-signs", default="1,-1,-1,1", metavar="S1,S2,S3,S4",
                    help="rotor spin-sign pattern for the yaw moment fit, "
                         "+1 = CW from above (mixer order M0..M3; default "
                         "%(default)s = QuadX per docs/motor_mixing.md)")
    ap.add_argument("--cutoff", type=float, default=64.0, metavar="HZ",
                    help="low-pass cutoff for angular acceleration (default: 64)")
    ap.add_argument("--save", metavar="PREFIX",
                    help="write PREFIX_thrust_drag.png / PREFIX_actuator.png / "
                         "PREFIX_moments.png")
    ap.add_argument("--no-show", action="store_true", help="skip interactive show")
    ap.add_argument("--vehicle", metavar="YAML",
                    help="vehicle YAML (vehicles/<v>.yaml): takes mass, inertia and "
                         "motor spin signs from it and prints a paste-ready tuning/sim "
                         "block with the identified values converted (docs/system_id.md "
                         "section 4.1; needs pyyaml)")
    ap.add_argument("--g1-override", action="store_true",
                    help="with --vehicle: also print the inertia-free g1_* override "
                         "block (alternative to the geometric G1; docs 4.2b)")
    ap.add_argument("--rotor-inertia", type=float, default=None, metavar="KG_M2",
                    help="with --vehicle: prop+bell polar inertia J_r, adds the "
                         "Izz = J_r/k_rd cross-check")
    args = ap.parse_args()
    if args.until_idle and not args.from_mission:
        print("note: --until-idle only modifies --from-mission — ignored")

    vehicle = load_vehicle(args.vehicle) if args.vehicle else None
    if vehicle:
        yaw_signs = vehicle["spin"]
        print(f"vehicle {args.vehicle}: mass {vehicle['mass']:.3f} kg, "
              f"I = diag({', '.join(f'{v:.4g}' for v in vehicle['inertia'])}), "
              f"spin signs {yaw_signs}")
    else:
        yaw_signs = [float(x) for x in args.yaw_signs.split(",")]
    if len(yaw_signs) != 4:
        sys.exit("--yaw-signs needs exactly 4 comma-separated values")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts = []
    for path in args.path:
        print("Loading", path)
        data = load(path)
        parts.append(build_dataset(data, "/imu1_raw" if args.raw else "/imu1",
                                   args.no_crop, args.t0, args.t1, args.from_mission,
                                   args.until_idle,
                                   "fused" if args.fused_omega else "raw"))
    d = parts[0]
    if len(parts) > 1:
        # Concatenate onto one virtual timeline (gaps between files collapse
        # to one grid step — fine for the least-squares fits; the actuator
        # fit re-seeds w0 per lfilter run only at the very start, so a
        # seam introduces one transient sample per file boundary).
        keys = [k for k in ("omega", "omega_dot", "u", "acc", "gyro", "vb")
                if all(k in p for p in parts)]
        dropped = [k for k in ("u", "acc", "gyro", "vb")
                   if any(k in p for p in parts) and k not in keys]
        if dropped:
            print(f"note: dropping {dropped} — not present in every file")
        dt = d["dt"]
        t_cat, off = [], 0.0
        for p in parts:
            t_cat.append(p["t"] + off)
            off = t_cat[-1][-1] + dt
        d = {"t": np.concatenate(t_cat), "dt": dt, "fs": 1.0 / dt}
        for k in keys:
            d[k] = np.concatenate([p[k] for p in parts], axis=0)
        d["seams"] = np.cumsum([len(p["t"]) for p in parts])[:-1]
        print(f"pooled {len(parts)} files: {len(d['t'])} samples, "
              f"{d['t'][-1]:.1f} s total")

    figs, summary, ident = {}, [], {}

    if args.fit in ("thrust", "all"):
        fig, res = fit_thrust_drag_model(d, plt)
        if fig:
            figs["thrust_drag"] = fig
        if res:
            k_w, k_x, k_y, kq_x, kq_y = res
            ident.update(k_w=k_w, k_x=k_x, k_y=k_y, kq_x=kq_x, kq_y=kq_y)
            summary.append(f"k_w = {k_w:.4e}"
                           + (f", k_x = {k_x:.4e}, k_y = {k_y:.4e}, "
                              f"kq_x = {kq_x:.4e}, kq_y = {kq_y:.4e}"
                              if k_x is not None else "  (drag: n/a)"))

    if args.fit in ("actuator", "all"):
        fig, res, per_motor = fit_actuator_model(d, plt)
        if fig:
            figs["actuator"] = fig
        if res is not None:
            w_min, w_max, k, tau_inv = res
            ident.update(w_min=w_min, w_max=w_max, k=k, tau=1 / tau_inv,
                         per_motor=per_motor)
            summary.append(f"w_min = {w_min:.2f}, w_max = {w_max:.2f}, "
                           f"k = {k:.3f}, tau = {1/tau_inv:.4f} s")

    if args.fit in ("moments", "all"):
        fig, res = fit_moments_model(d, plt, yaw_signs, args.cutoff)
        if fig:
            figs["moments"] = fig
        if res:
            ident.update(res)
            summary.append("k_p1..4 = " + ", ".join(f"{v:.3e}" for v in res["k_p"]))
            summary.append("k_q1..4 = " + ", ".join(f"{v:.3e}" for v in res["k_q"]))
            summary.append(f"k_r = {res['k_r']:.3e}, k_rd = {res['k_rd']:.3e}")

    if args.fit in ("inertia", "all"):
        fig, res = fit_inertia_model(d, vehicle, plt, args.cutoff,
                                     rotor_inertia=args.rotor_inertia)
        if fig:
            figs["inertia"] = fig
        if res:
            ident["inertia_diag"] = [res["ixx"], res["iyy"], res["izz"]]
            ident["izz_src"] = res["izz_src"]
            ident["jr_euler"] = res["jr"]
            ident["jr_over_izz"] = res["jr_over_izz"]
            ident["cog"] = (res["xcg"], res["ycg"])
            summary.append(f"Ixx = {res['ixx']:.4e}, Iyy = {res['iyy']:.4e}, "
                           f"Izz = {res['izz']:.4e} kg·m² "
                           f"(Euler fit, R² {res['r2'][0]:.2f}/{res['r2'][1]:.2f}/"
                           f"{res['r2'][2]:.2f}; Izz via {res['izz_src']})")
            summary.append(f"J_r/Izz = {res['jr_over_izz']:.4e} "
                           f"(-> J_r = {res['jr']:.3e} kg·m²), "
                           f"CoG offset x = {res['xcg'] * 1e3:+.1f} mm, "
                           f"y = {res['ycg'] * 1e3:+.1f} mm")

    if summary:
        print("\n── fitted parameters ──")
        for line in summary:
            print("  " + line)
    if vehicle and ident:
        print_yaml_block(vehicle, ident, float(np.mean(d["omega"])),
                         g1_override=args.g1_override, rotor_inertia=args.rotor_inertia)

    if not figs:
        sys.exit("nothing fitted (missing topics? tier < mid?)")
    if args.save:
        for tag, fig in figs.items():
            out = f"{args.save}_{tag}.png"
            fig.savefig(out, dpi=130)
            print(f"saved {out}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
