#!/usr/bin/env python3
"""Plot a dataset of flown missions against their reference trajectories.

A dataset directory holds one pair of files per trajectory shape:

    <name>.mcap   blackbox recording of the flight (see plot_mcap.py)
    <name>.csv    planner reference: t, p_xyz, q_xyzw, v_xyz, a_lin_xyz, …

Every `.mcap` with a sibling `.csv` is one column; the layout scales
with the number of shapes found (3 today, N tomorrow). Each recording
is cropped to the mission segment exactly like
`plot_mcap.py --from-mission --until-idle` (mission trigger → first
`MISSION_IDLE` event; the trigger is the first mission-switch rising
edge on `/rc`, or — for recordings that do not log `/rc` — the first
`MISSION_PLANNING` / `MISSION_EXECUTING` event, and if a log has no
idle event the pilot's first throttle descent command or the DISARM
event is used to close the segment), and the reference is
time-aligned to the flight by the lag that minimises position error
(the planner needs a moment to solve after the trigger).

Figures (each `1 × N` or `k × N` columns, one column per shape):

  1. **xy**       top-down executed trajectory vs reference
  2. **vel_acc**  |v| and |a| executed vs reference (accel is the
                  low-passed (~2 Hz) derivative of the ESKF velocity)
  3. **rpy**      roll / pitch / yaw executed vs reference. Default is
                  the tilt–torsion split (roll/pitch = tilt-vector x/y,
                  yaw = torsion), which stays continuous through the ~90°
                  tilts of the time-optimal missions where every Euler
                  sequence gimbal-locks; `--euler zyx|zxy` for those.

and a per-shape summary of everything the log says about the flown
trajectory, over the mission segment only:

  mission          duration from the trigger to the MISSION_IDLE event,
                   against the planned duration, path length and the
                   fraction of the path actually completed
  speed / accel    executed peak and mean vs the plan's, plus the
                   path-length/duration average speed (executed accel
                   is the low-passed derivative, see below)
  tilt / rotation  tilt angle (thrust axis off vertical, gimbal-lock
                   free) and the total attitude rotation angle, plus
                   the yaw span
  body rate        |ω| peak, per-axis peaks, p95 and mean in rad/s,
                   straight off the ESKF twist (no differentiation)
  motor command    peak / mean / p95 of the normalized commands, the
                   fraction of the mission with any motor saturated,
                   and the per-motor trim spread (mixer asymmetry)
  geometric RMSE   RMS of the distance from every executed sample to
                   the closest point on the reference path (time-free,
                   "how far off the path was it"), split into
                     contouring  distance to the reference *polyline*
                     progress τ  how far behind schedule, in ms and in
                                 cm of arclength (see `path_metrics`)
  temporal RMSE    RMS of |p_exec(t) − p_ref(t)| after lag alignment
                   ("how far from where it should have been *now*")

`--save` writes the same fields, one row per shape, to `PREFIX_rmse.csv`.

Usage:
    python3 analysis/plot_dataset.py analysis/datasets/indoor_exp_slow
    python3 analysis/plot_dataset.py DIR --save out --no-show   # out_xy.png, out_vel_acc.png, out_rpy.png, out_rmse.csv
    python3 analysis/plot_dataset.py DIR --lag 0.8              # fixed reference lag instead of auto

Requires: pip install mcap cbor2 numpy matplotlib
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_mcap import (col, crop, euler_deg, load, mission_window,  # noqa: E402
                       quat_to_rotm)

# Firmware fallback RC mapping (rc_interpreter.rs / params.rs). Override
# from the CLI if the vehicle YAML pins different values.
MISSION_CHANNEL, MISSION_HIGH_US, MISSION_LOW_US = 4, 1700, 1300
THROTTLE_CHANNEL, THROTTLE_DEADBAND = 2, 0.12
# A normalized motor command at or above this counts as saturated — the
# point where the allocator has no authority left (matches paper_metrics).
SAT_THRESHOLD = 0.98


# ── loading ──────────────────────────────────────────────────────────────

def find_pairs(dataset_dir):
    names = sorted(f[:-5] for f in os.listdir(dataset_dir) if f.endswith(".mcap"))
    pairs = []
    for n in names:
        csv_path = os.path.join(dataset_dir, n + ".csv")
        if os.path.exists(csv_path):
            pairs.append((n, os.path.join(dataset_dir, n + ".mcap"), csv_path))
        else:
            print(f"note: {n}.mcap has no {n}.csv reference — skipped")
    return pairs


def pretty(name):
    """exp_figure8_slow -> figure8 slow"""
    parts = name.split("_")
    if parts and parts[0] == "exp":
        parts = parts[1:]
    return " ".join(parts)


def load_reference(path, seq):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    g = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    ref = {
        "t": g("t"),
        "pos": np.stack([g("p_x"), g("p_y"), g("p_z")], 1),
        "vel": np.stack([g("v_x"), g("v_y"), g("v_z")], 1),
        "acc": np.stack([g("a_lin_x"), g("a_lin_y"), g("a_lin_z")], 1),
    }
    ref["rpy"] = np.stack(euler_deg(g("q_w"), g("q_x"), g("q_y"), g("q_z"), seq), 1)
    return ref


def load_flight(path, args):
    data = load(path)
    rise, end, start_src, trim = mission_window(data, args)
    if rise is None:
        sys.exit(f"{path}: no mission trigger — no rising edge on /rc channel "
                 f"{args.mission_channel} and no MISSION_PLANNING/MISSION_EXECUTING "
                 "event")
    if trim != "MISSION_IDLE event":
        print(f"note: {path}: no MISSION_IDLE event after trigger — using {trim}")
    crop(data, rise, end)
    odom = data.get("/odometry", [])
    if not odom:
        sys.exit(f"{path}: no /odometry messages in the mission segment")

    qw, qi, qj, qk = (col(odom, k) for k in ("q_w", "q_i", "q_j", "q_k"))
    t = (col(odom, "timestamp_ns") - rise) / 1e9
    pos = np.stack([col(odom, f"pos_{a}_m") for a in "xyz"], 1)
    vel = np.stack([col(odom, f"vel_{a}_m_s") for a in "xyz"], 1)
    # body rates come straight off the ESKF twist — no differentiation
    omega = np.stack([col(odom, f"omega_{a}_rad_s") for a in "xyz"], 1)
    rpy = np.stack(euler_deg(qw, qi, qj, qk, args.euler), 1)
    # tilt = angle between the thrust axis and world +Z (gimbal-lock free);
    # rot = the geodesic rotation angle of the whole attitude, |q| → angle.
    Rm = quat_to_rotm(qw, qi, qj, qk)
    tilt = np.degrees(np.arccos(np.clip(Rm[:, 2, 2], -1.0, 1.0)))
    rot = np.degrees(2 * np.arccos(np.clip(np.abs(qw), 0.0, 1.0)))

    mot = data.get("/motors", [])
    u = np.array([m["motor_commands"] for m in mot], dtype=float) if mot else None
    t_u = (col(mot, "timestamp_ns") - rise) / 1e9 if mot else None

    return {"t": t, "pos": pos, "vel": vel, "acc": smooth_derivative(t, vel),
            "rpy": rpy, "omega": omega, "tilt": tilt, "rot": rot,
            "u": u, "t_u": t_u, "trim": trim, "trigger": start_src,
            "segment_s": (end - rise) / 1e9 if end else float(t[-1] - t[0])}


def smooth_derivative(t, y, window_s=0.2, fs=100.0):
    """d/dt of `y`, robust to the 1 kHz ESKF velocity's correction jumps.

    Resamples onto a uniform `fs` grid (the raw stamps have duplicates
    and jitter), applies two passes of a `window_s` moving average
    (≈ triangular low-pass), differentiates, and interpolates back to
    `t`. With the defaults the cut-off is ~2–3 Hz, plenty for the
    ≤1 Hz manoeuvres of these missions.
    """
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    yu = interp_rows(t, y, tu)
    n = max(1, int(round(window_s * fs)))
    if n > 1:
        k = np.ones(n) / n
        for _ in range(2):
            yu = np.stack([np.convolve(np.pad(yu[:, i], (n // 2, n - 1 - n // 2), mode="edge"),
                                       k, mode="valid") for i in range(yu.shape[1])], 1)
    dy = np.gradient(yu, tu, axis=0)
    return interp_rows(tu, dy, t)


# ── alignment & error metrics ────────────────────────────────────────────

def interp_rows(t_src, y_src, t_dst):
    return np.stack([np.interp(t_dst, t_src, y_src[:, i]) for i in range(y_src.shape[1])], 1)


def temporal_rmse(flight, ref, lag):
    """RMS of |p_exec(t) − p_ref(t − lag)| over the overlap of both."""
    t_ref = ref["t"] + lag
    mask = (flight["t"] >= t_ref[0]) & (flight["t"] <= t_ref[-1])
    if mask.sum() < 10:
        return np.nan
    p_ref = interp_rows(t_ref, ref["pos"], flight["t"][mask])
    return float(np.sqrt(np.mean(np.sum((flight["pos"][mask] - p_ref) ** 2, 1))))


def find_lag(flight, ref, lo=0.0, hi=3.0, step=0.01):
    """Reference lag (s after the trigger) minimising temporal RMSE."""
    lags = np.arange(lo, hi + step / 2, step)
    errs = np.array([temporal_rmse(flight, ref, l) for l in lags])
    if np.all(np.isnan(errs)):
        return 0.0
    return float(lags[np.nanargmin(errs)])


def geometric_rmse(p_exec, p_ref, chunk=2048):
    """RMS over executed samples of the distance to the nearest reference
    sample (the reference is dense — 100 Hz — so point-to-point is fine)."""
    d2 = np.empty(len(p_exec))
    for i in range(0, len(p_exec), chunk):
        blk = p_exec[i:i + chunk]
        diff = blk[:, None, :] - p_ref[None, :, :]
        d2[i:i + chunk] = np.min(np.sum(diff ** 2, 2), 1)
    return float(np.sqrt(np.mean(d2)))


def path_metrics(flight, ref):
    """Split the tracking error into its contouring and progress parts.

    `geometric_rmse` above answers "how far off the path", but it says
    nothing about *where along* the path the vehicle was, and an error
    that is purely a delay looks identical to one that is purely a
    cross-track offset. The split is the standard contouring
    decomposition, computed with the primitives in `paper_metrics.py`:

      contour  e_c  distance to the nearest point on the reference
                    *polyline* (segment projection, so it does not
                    depend on the planner's output rate — unlike
                    `geometric_rmse`, which snaps to the nearest
                    reference *sample*).
      progress τ    how far behind schedule the vehicle is. The
                    executed trajectory is matched to the reference by
                    a globally monotone alignment (DTW), which is what
                    makes it correct on the self-intersecting figure-8
                    and splits paths, and the matched reference point
                    carries a reference time. Also reported in metres
                    of arclength.

    The reference clock is first shifted by the lag that aligns only
    the first couple of seconds (the planner's post-trigger latency),
    so that start-up delay is not counted as tracking lag.
    """
    import paper_metrics as pm  # deferred: paper_metrics imports this module

    p_exec, t = flight["pos"], flight["t"]
    s_ref = pm.arclength(ref["pos"])
    e_c, _, _ = pm.polyline_distance(p_exec, ref["pos"])

    fidx = pm.dtw_progress(p_exec, ref["pos"], t=t)
    s_exec = np.interp(fidx, np.arange(len(s_ref)), s_ref)
    t_matched = np.interp(fidx, np.arange(len(ref["t"])), ref["t"])

    lag0 = pm.startup_lag({"t": t, "pos": p_exec}, ref)
    tau = (t - lag0) - t_matched
    s_ref_now = np.interp(np.clip(t - lag0, ref["t"][0], ref["t"][-1]), ref["t"], s_ref)

    return {
        "contour_rmse": float(np.sqrt(np.mean(e_c ** 2))),
        "contour_p95": float(np.percentile(e_c, 95)),
        "contour_max": float(e_c.max()),
        "tau_rms": float(np.sqrt(np.mean(tau ** 2))),
        "tau_median": float(np.median(tau)),
        "tau_p95": float(np.percentile(tau, 95)),
        "tau_rms_m": float(np.sqrt(np.mean((s_ref_now - s_exec) ** 2))),
        "startup_lag": lag0,
        "path_length": float(s_ref[-1]),
        "completion": float(s_exec[-1] / s_ref[-1]),
    }


def describe(r):
    """Everything the log says about one flown trajectory, as text."""
    f, ref, pm_ = r["flight"], r["ref"], r["path"]
    v, a = np.linalg.norm(f["vel"], axis=1), np.linalg.norm(f["acc"], axis=1)
    w = f["omega"]                      # rad/s, straight off the ESKF twist
    w_norm = np.linalg.norm(w, axis=1)
    v_ref = np.linalg.norm(ref["vel"], axis=1)
    a_ref = np.linalg.norm(ref["acc"], axis=1)
    yaw = f["rpy"][:, 2]

    L = [f'── {r["label"]} ' + "─" * max(0, 60 - len(r["label"]))]
    L.append(f'  mission       {f["segment_s"]:8.2f} s  '
             f'{f["trigger"]} → {f["trim"]}'
             f'   (plan {ref["t"][-1] - ref["t"][0]:.2f} s,'
             f' path {pm_["path_length"]:.1f} m, completed {pm_["completion"]*100:.1f} %)')
    L.append(f'  speed         {v.max():8.2f} m/s max   (plan {v_ref.max():.2f})'
             f'   mean {v.mean():.2f} (plan {v_ref.mean():.2f})'
             f'   path/time {pm_["path_length"] / f["segment_s"]:.2f}')
    L.append(f'  acceleration  {a.max():8.2f} m/s² max  (plan {a_ref.max():.2f})'
             f'   mean {a.mean():.2f} (plan {a_ref.mean():.2f})'
             f'   [executed = ~2 Hz low-passed dv/dt]')
    L.append(f'  tilt angle    {f["tilt"].max():8.2f} °  max     p95 {np.percentile(f["tilt"], 95):.2f}'
             f'   mean {f["tilt"].mean():.2f}')
    L.append(f'  rotation      {f["rot"].max():8.2f} °  max total attitude angle'
             f'   yaw span {yaw.max() - yaw.min():.1f} ° ({yaw.min():.1f} → {yaw.max():.1f})')
    L.append(f'  body rate     {w_norm.max():8.2f} rad/s |ω| max'
             f'   per axis {w[:, 0].max():.2f} / {w[:, 1].max():.2f} / {w[:, 2].max():.2f}'
             f'   p95 {np.percentile(w_norm, 95):.2f}   mean {w_norm.mean():.2f}')
    if f["u"] is not None:
        u = f["u"]
        sat = float((u >= SAT_THRESHOLD).any(1).mean())
        L.append(f'  motor cmd     {u.max():8.3f}    max        mean {u.mean():.3f}'
                 f'   p95 {np.percentile(u, 95):.3f}'
                 f'   ≥{SAT_THRESHOLD:.2f} for {sat*100:.1f} % of the mission')
        L.append(f'                per motor mean '
                 + " / ".join(f"{x:.3f}" for x in u.mean(0))
                 + f'   spread {u.mean(0).max() - u.mean(0).min():.3f}')
    else:
        L.append('  motor cmd     — (no /motors in this recording)')
    L.append(f'  geometric RMSE{r["geom_rmse"]*100:8.2f} cm  (nearest reference sample)')
    L.append(f'    contouring  {pm_["contour_rmse"]*100:8.2f} cm RMS'
             f'   p95 {pm_["contour_p95"]*100:.2f}   max {pm_["contour_max"]*100:.2f}')
    L.append(f'    progress τ  {pm_["tau_rms"]*1000:8.1f} ms RMS'
             f'   median {pm_["tau_median"]*1000:.1f}   p95 {pm_["tau_p95"]*1000:.1f}'
             f'   = {pm_["tau_rms_m"]*100:.1f} cm along path')
    L.append(f'  temporal RMSE {r["temp_rmse"]*100:8.2f} cm  at a fixed reference lag of'
             f' {r["lag"]:.2f} s   (start-up lag {pm_["startup_lag"]:.2f} s)')
    return "\n".join(L)


# ── plotting ─────────────────────────────────────────────────────────────

def grid(plt, rows, ncols, height_per_row, width_per_col=4.5, sharex=False):
    fig, axes = plt.subplots(rows, ncols, squeeze=False, sharex=sharex,
                             figsize=(width_per_col * ncols, height_per_row * rows))
    return fig, axes


def plot_xy(plt, runs):
    fig, axes = grid(plt, 1, len(runs), 4.8)
    for ax, r in zip(axes[0], runs):
        ax.plot(r["ref"]["pos"][:, 0], r["ref"]["pos"][:, 1], "k--", lw=1, label="reference")
        ax.plot(r["flight"]["pos"][:, 0], r["flight"]["pos"][:, 1], lw=1, label="executed")
        ax.scatter(*r["flight"]["pos"][0, :2], c="green", s=30, zorder=3, label="start")
        ax.scatter(*r["flight"]["pos"][-1, :2], c="red", s=30, zorder=3, label="end")
        ax.set_title(f'{r["label"]}\ngeom. RMSE {r["geom_rmse"]*100:.1f} cm')
        ax.set_xlabel("x east (m)")
        ax.set_ylabel("y north (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_vel_acc(plt, runs):
    fig, axes = grid(plt, 2, len(runs), 2.8, sharex="col")
    for c, r in enumerate(runs):
        f, ref = r["flight"], r["ref"]
        t_ref = ref["t"] + r["lag"]
        for row, key, unit in ((0, "vel", "m/s"), (1, "acc", "m/s²")):
            ax = axes[row][c]
            ax.plot(t_ref, np.linalg.norm(ref[key], axis=1), "k--", lw=1, label="reference")
            ax.plot(f["t"], np.linalg.norm(f[key], axis=1), lw=0.8, label="executed")
            ax.set_ylabel(f"|{key}| ({unit})")
            ax.grid(True, alpha=0.3)
        axes[0][c].set_title(r["label"])
        axes[1][c].set_xlabel("t since trigger (s)")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_rpy(plt, runs, args):
    fig, axes = grid(plt, 3, len(runs), 2.4, sharex="col")
    for c, r in enumerate(runs):
        f, ref = r["flight"], r["ref"]
        t_ref = ref["t"] + r["lag"]
        for row, name in enumerate(("roll", "pitch", "yaw")):
            ax = axes[row][c]
            ax.plot(t_ref, ref["rpy"][:, row], "k--", lw=1, label="reference")
            ax.plot(f["t"], f["rpy"][:, row], lw=0.8, label="executed")
            ax.set_ylabel(f"{name} (deg)")
            ax.grid(True, alpha=0.3)
        axes[0][c].set_title(r["label"])
        axes[2][c].set_xlabel("t since trigger (s)")
    fig.suptitle("tilt–torsion split" if args.euler == "tilt" else f"{args.euler.upper()} euler", fontsize=9)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


# ── tabular summary ──────────────────────────────────────────────────────

SUMMARY_COLUMNS = [
    "trajectory", "mission_s", "trim", "reference_s", "path_length_m",
    "completion_frac", "v_max_m_s", "v_mean_m_s", "v_avg_path_m_s",
    "v_ref_max_m_s", "v_ref_mean_m_s",
    "a_max_m_s2", "a_mean_m_s2", "a_ref_max_m_s2", "a_ref_mean_m_s2",
    "tilt_max_deg", "tilt_p95_deg",
    "rot_max_deg", "yaw_span_deg", "omega_max_rad_s", "omega_p95_rad_s", "omega_mean_rad_s",
    "u_max", "u_mean", "u_p95", "sat_frac", "u_trim_spread",
    "geometric_rmse_m", "contour_rmse_m", "contour_p95_m", "contour_max_m",
    "tau_rms_s", "tau_median_s", "tau_p95_s", "tau_rms_m",
    "temporal_rmse_m", "lag_s", "startup_lag_s",
]


def summary_row(r):
    f, ref, pm_ = r["flight"], r["ref"], r["path"]
    v, a = np.linalg.norm(f["vel"], axis=1), np.linalg.norm(f["acc"], axis=1)
    v_ref = np.linalg.norm(ref["vel"], axis=1)
    a_ref = np.linalg.norm(ref["acc"], axis=1)
    w = np.linalg.norm(f["omega"], axis=1)
    yaw = f["rpy"][:, 2]
    u = f["u"]
    um = u.mean(0) if u is not None else None
    g = lambda x, n=4: "" if x is None else f"{x:.{n}f}"  # noqa: E731
    return [
        r["name"], g(f["segment_s"], 3), f["trim"], g(ref["t"][-1] - ref["t"][0], 3),
        g(pm_["path_length"], 3), g(pm_["completion"]),
        g(v.max(), 3), g(v.mean(), 3), g(pm_["path_length"] / f["segment_s"], 3),
        g(v_ref.max(), 3), g(v_ref.mean(), 3),
        g(a.max(), 3), g(a.mean(), 3), g(a_ref.max(), 3), g(a_ref.mean(), 3),
        g(f["tilt"].max(), 2), g(np.percentile(f["tilt"], 95), 2),
        g(f["rot"].max(), 2), g(yaw.max() - yaw.min(), 2),
        g(w.max(), 3), g(np.percentile(w, 95), 3), g(w.mean(), 3),
        g(None if u is None else u.max(), 4),
        g(None if u is None else u.mean(), 4),
        g(None if u is None else np.percentile(u, 95), 4),
        g(None if u is None else float((u >= SAT_THRESHOLD).any(1).mean())),
        g(None if um is None else um.max() - um.min()),
        g(r["geom_rmse"], 5), g(pm_["contour_rmse"], 5), g(pm_["contour_p95"], 5),
        g(pm_["contour_max"], 5), g(pm_["tau_rms"], 5), g(pm_["tau_median"], 5),
        g(pm_["tau_p95"], 5), g(pm_["tau_rms_m"], 5),
        g(r["temp_rmse"], 5), g(r["lag"], 3), g(pm_["startup_lag"], 3),
    ]


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset_dir", help="directory with <name>.mcap + <name>.csv pairs")
    ap.add_argument("--save", metavar="PREFIX",
                    help="write PREFIX_xy.png / PREFIX_vel_acc.png / PREFIX_rpy.png / PREFIX_rmse.csv")
    ap.add_argument("--no-show", action="store_true", help="skip interactive show")
    ap.add_argument("--lag", type=float, default=None,
                    help="fixed reference lag after the trigger [s] (default: auto per run)")
    ap.add_argument("--euler", choices=("tilt", "zxy", "zyx"), default="tilt",
                    help="euler sequence for the rpy figure (see plot_mcap.py --euler)")
    ap.add_argument("--mission-channel", type=int, default=MISSION_CHANNEL)
    ap.add_argument("--mission-high", type=int, default=MISSION_HIGH_US)
    ap.add_argument("--mission-low", type=int, default=MISSION_LOW_US)
    ap.add_argument("--throttle-channel", type=int, default=THROTTLE_CHANNEL)
    ap.add_argument("--throttle-deadband", type=float, default=THROTTLE_DEADBAND)
    args = ap.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = find_pairs(args.dataset_dir)
    if not pairs:
        sys.exit(f"{args.dataset_dir}: no <name>.mcap + <name>.csv pairs found")

    runs = []
    for name, mcap_path, csv_path in pairs:
        flight = load_flight(mcap_path, args)
        ref = load_reference(csv_path, args.euler)
        lag = args.lag if args.lag is not None else find_lag(flight, ref)
        runs.append({
            "name": name, "label": pretty(name), "flight": flight, "ref": ref,
            "lag": lag,
            "geom_rmse": geometric_rmse(flight["pos"], ref["pos"]),
            "temp_rmse": temporal_rmse(flight, ref, lag),
            "duration": float(flight["t"][-1] - flight["t"][0]),
            "path": path_metrics(flight, ref),
        })

    for r in runs:
        print(describe(r), end="\n\n")

    print(f"{'trajectory':<22}{'geom RMSE [m]':>15}{'temporal RMSE [m]':>19}"
          f"{'lag [s]':>9}{'flown [s]':>11}{'ref [s]':>9}")
    for r in runs:
        print(f"{r['label']:<22}{r['geom_rmse']:>15.4f}{r['temp_rmse']:>19.4f}"
              f"{r['lag']:>9.2f}{r['duration']:>11.2f}{r['ref']['t'][-1]:>9.2f}")

    figs = {"xy": plot_xy(plt, runs),
            "vel_acc": plot_vel_acc(plt, runs),
            "rpy": plot_rpy(plt, runs, args)}

    if args.save:
        for tag, fig in figs.items():
            out = f"{args.save}_{tag}.png"
            fig.savefig(out, dpi=130)
            print(f"saved {out}")
        out = f"{args.save}_rmse.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(SUMMARY_COLUMNS)
            for r in runs:
                w.writerow(summary_row(r))
        print(f"saved {out}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
