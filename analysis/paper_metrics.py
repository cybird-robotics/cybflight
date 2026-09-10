#!/usr/bin/env python3
"""Paper-grade tracking / compute / effort metrics for the indoor experiments.

The metric core behind `paper_tables.py` (LaTeX) and `paper_figures.py`
(matplotlib). It reuses the mission-segment cropping and reference
loading of `plot_dataset.py` — every run is cropped exactly like
`plot_mcap.py --from-mission --until-idle` (mission trigger -> first
`MISSION_IDLE` event), which is the only part of a flight that belongs
in a tracking table. The trigger is the `/rc` mission-switch rising
edge where the recording logs `/rc`, and the firmware's own
`MISSION_PLANNING` / `MISSION_EXECUTING` event where it does not; see
`plot_mcap.mission_start` for why the two are interchangeable here.

Metrics, and why each one is here
---------------------------------

**contour error** `e_c(t)` — distance from the executed sample to the
  *nearest point on the reference polyline* (segment projection, not
  nearest reference sample, so the number does not depend on the
  planner's output rate). Reported as RMS / p95 / max. This is the
  "how far off the path" number and the paper's headline metric.

**temporal lag** `tau(t)` — how far *behind schedule* the vehicle is:
  the matched reference point carries a reference time, and `tau = t -
  t_ref`. Contour error alone is gamed by flying slower, so it is never
  reported without a progress metric beside it.

  The match here is **not** the global nearest point. `figure8` and
  `splits` cross themselves and `circle` closes, so a global search
  snaps to whichever branch happens to be nearer and reports phantom
  multi-second lags. Progress comes instead from a globally monotone
  minimum-cost alignment (`dtw_progress`); see that function for why a
  greedy causal tracker is not enough. The contour error keeps the
  global definition — distance to the path *set* — so it stays
  comparable with `plot_dataset.geometric_rmse`.

  The reference is first aligned to the flight by the lag that minimises
  position error **over the first `ALIGN_WINDOW_S` seconds only**. That
  removes the planner's start-up latency (the solver needs a moment
  after the trigger) without absorbing the steady-state lag that the
  contouring cost deliberately accepts — aligning over the whole run,
  as `plot_dataset.find_lag` does, would hide exactly the effect the
  time-cost ablation is meant to expose.

**along-path lag** — the same quantity in metres of arclength, for
  readers who prefer a distance.

**solve time / iterations / converged / tick period** — from `/mpc`,
  one message per solve, and absent from the metric dict entirely when
  the recording mutes that topic. The tick period is the *logged* solve cadence
  and is subject to blackbox backpressure (multi-100 ms gaps appear in
  every log), so the median is used and the mean is not reported.

**trim asymmetry** — spread of the four per-motor mean commands. A
  payload on the centre of gravity raises all four equally; one mounted
  off-centre makes the allocator hold a steady differential, and this
  number is that differential. It is what distinguishes a pure mass
  disturbance from a mass-plus-moment disturbance in the logs.

**saturation** — fraction of the mission with **any** motor command at
  or above `SAT_THRESHOLD` of full scale; that is the moment the
  allocator loses authority, which is the quantitative form of a
  robustness margin. Note `u_mean` is the mean over all four motors and
  the whole mission, a proxy for the collective thrust the vehicle had
  to hold up. A *high* value at the `timeopt` operating point is the
  expected and desirable result — a genuinely time-optimal trajectory
  rides the actuator limits — so this column reads as confirmation
  there, and as lost margin at `slow`/`mid`.

**bootstrap CI** — a moving-block bootstrap (block ~`BLOCK_S` seconds,
  so the error's autocorrelation survives resampling) on the contour
  RMSE. This is a **within-run precision interval**: it says how well
  the RMSE of *this flight* is determined, NOT how much the result
  would vary across repeated flights. It is not a substitute for
  repeated trials and must not be described as one in the paper.

Usage:
    python3 analysis/paper_metrics.py --dump out.json      # every run
    python3 analysis/paper_metrics.py --dump out.json --no-bootstrap
    python3 analysis/paper_metrics.py --print              # human table

Requires: pip install mcap cbor2 numpy
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_dataset as pdd  # noqa: E402
from plot_mcap import col, crop, load, mission_window, quat_to_rotm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(HERE, "datasets")

SPEEDS = ["slow", "mid", "timeopt"]
SHAPES = ["circle", "figure8", "slalom", "splits"]
VARIANTS = ["load", "iter2", "timecost"]

ALIGN_WINDOW_S = 2.0   # start-up alignment window (see module docstring)
ALIGN_MAX_S = 1.5      # search range for that alignment
DTW_RATE_HZ = 50.0     # decimation rate for the monotone alignment
DTW_MAX_REF = 600      # reference points kept for that alignment
SAT_THRESHOLD = 0.98   # motor command counted as saturated at/above this
BLOCK_S = 0.5          # moving-block bootstrap block length
N_BOOT = 1000


class _Args:
    """Stand-in for the argparse namespace `plot_dataset` helpers expect."""
    mission_channel = pdd.MISSION_CHANNEL
    mission_high = pdd.MISSION_HIGH_US
    mission_low = pdd.MISSION_LOW_US
    throttle_channel = pdd.THROTTLE_CHANNEL
    throttle_deadband = pdd.THROTTLE_DEADBAND
    euler = "tilt"


# ── geometry ─────────────────────────────────────────────────────────────

def polyline_distance(p, ref, chunk=512):
    """Distance from each row of `p` to the nearest point on the polyline
    through `ref` (projection onto the closest segment), plus the index of
    that segment and the in-segment parameter."""
    a, b = ref[:-1], ref[1:]
    d = b - a
    dd = np.sum(d * d, 1)
    dd = np.where(dd == 0, 1e-12, dd)
    dist = np.empty(len(p))
    seg = np.empty(len(p), dtype=int)
    tpar = np.empty(len(p))
    for i in range(0, len(p), chunk):
        blk = p[i:i + chunk]
        w = blk[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(w * d[None], 2) / dd[None], 0.0, 1.0)
        proj = a[None] + t[..., None] * d[None]
        d2 = np.sum((blk[:, None, :] - proj) ** 2, 2)
        j = np.argmin(d2, 1)
        rows = np.arange(len(blk))
        dist[i:i + chunk] = np.sqrt(d2[rows, j])
        seg[i:i + chunk] = j
        tpar[i:i + chunk] = t[rows, j]
    return dist, seg, tpar


def arclength(ref):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))])


def dtw_progress(p, ref, max_ref=600, rate_hz=50.0, t=None):
    """Monotone alignment of the executed trajectory to the reference path.

    Returns, for every row of `p`, a fractional reference index. The
    assignment is a **globally monotone** minimum-cost warp (dynamic time
    warping with the standard (1,0)/(0,1)/(1,1) step set), which is what
    makes it correct on the paths here: `figure8` and `splits` cross
    themselves and `circle` closes, so a nearest-point match snaps to
    whichever branch is momentarily closer and reports phantom
    multi-second lags. A greedy causal tracker fixes the crossings but
    then stalls on the tightly-woven `slalom`, where consecutive passes
    are only metres apart along the path. The monotone constraint is a
    global property, so DTW gets both right without a window to tune.

    Both endpoints are anchored: every mission in this dataset runs to
    its `MISSION_IDLE` event with a duration within ~2% of the plan, so
    the flight really does start and finish at the ends of the path. The
    lag at the very first and last samples is therefore pinned to ~0 by
    construction and should not be quoted on its own.

    The warp is computed on trajectories decimated to `rate_hz` and
    `max_ref` reference points (the cost matrix is then a few 10^4
    cells) and interpolated back to full rate.
    """
    if t is None:
        t = np.arange(len(p), dtype=float)
    # decimate the executed path to a uniform grid
    tu = np.arange(t[0], t[-1], 1.0 / rate_hz)
    if len(tu) < 4:
        tu = t
    pu = np.stack([np.interp(tu, t, p[:, i]) for i in range(p.shape[1])], 1)
    # decimate the reference
    ri = np.linspace(0, len(ref) - 1, min(max_ref, len(ref)))
    rr = np.stack([np.interp(ri, np.arange(len(ref)), ref[:, i])
                   for i in range(ref.shape[1])], 1)

    n, m = len(pu), len(rr)
    cost = np.sqrt(((pu[:, None, :] - rr[None, :, :]) ** 2).sum(2))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        prev, cur = D[i - 1], D[i]
        c = cost[i - 1]
        # cur[j] = c[j-1] + min(D[i-1,j-1], D[i-1,j], cur[j-1]); the last
        # term is a running minimum, so the row is filled left to right.
        best_diag_up = np.minimum(prev[:-1], prev[1:])   # over j-1, j
        run = np.inf
        for j in range(1, m + 1):
            run = min(best_diag_up[j - 1], run) + c[j - 1]
            cur[j] = run
            run = cur[j]
    # backtrack
    i, j = n, m
    match = np.empty(n)
    while i > 0 and j > 0:
        match[i - 1] = j - 1
        opts = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
        k = int(np.argmin(opts))
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    while i > 0:
        match[i - 1] = 0
        i -= 1
    # decimated reference index -> full reference index, then back to full rate
    full = np.interp(match, np.arange(m), ri)
    return np.interp(t, tu, full)


def startup_lag(flight, ref, window_s=ALIGN_WINDOW_S, hi=ALIGN_MAX_S, step=0.005):
    """Reference lag [s] that best aligns only the first `window_s` of the
    flight — the planner's post-trigger latency, not the tracking lag."""
    mask = flight["t"] <= flight["t"][0] + window_s
    if mask.sum() < 10:
        return 0.0
    sub = {"t": flight["t"][mask], "pos": flight["pos"][mask]}
    lags = np.arange(0.0, hi + step / 2, step)
    errs = np.array([pdd.temporal_rmse(sub, ref, l) for l in lags])
    if np.all(np.isnan(errs)):
        return 0.0
    return float(lags[np.nanargmin(errs)])


def block_bootstrap_rmse(e, dt_s, block_s=BLOCK_S, n=N_BOOT, seed=0):
    """95% CI of RMS(e) by moving-block bootstrap (within-run precision)."""
    k = max(1, int(round(block_s / max(dt_s, 1e-6))))
    nblk = max(1, len(e) // k)
    if len(e) < 2 * k:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(e) - k, size=(n, nblk))
    idx = starts[:, :, None] + np.arange(k)[None, None, :]
    samp = e[idx.reshape(n, -1)]
    r = np.sqrt(np.mean(samp ** 2, 1))
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


# ── one run ──────────────────────────────────────────────────────────────

def analyse(mcap_path, csv_path, bootstrap=True, series=False):
    """Full metric dict for one (recording, reference) pair.

    With `series=True` the per-sample time series used by the figures are
    attached under the `_series` key (not JSON-serialisable).
    """
    data = load(mcap_path)
    rise, end, start_src, trim = mission_window(data, _Args)
    if rise is None:
        raise RuntimeError(f"{mcap_path}: no mission trigger (no /rc mission-switch "
                           "edge and no MISSION_PLANNING/MISSION_EXECUTING event)")
    crop(data, rise, end)

    odom = data.get("/odometry", [])
    if not odom:
        raise RuntimeError(f"{mcap_path}: no /odometry in the mission segment")
    t = (col(odom, "timestamp_ns") - rise) / 1e9
    pos = np.stack([col(odom, f"pos_{a}_m") for a in "xyz"], 1)
    vel = np.stack([col(odom, f"vel_{a}_m_s") for a in "xyz"], 1)
    Rm = quat_to_rotm(col(odom, "q_w"), col(odom, "q_i"),
                      col(odom, "q_j"), col(odom, "q_k"))
    tilt = np.degrees(np.arccos(np.clip(Rm[:, 2, 2], -1.0, 1.0)))

    ref = pdd.load_reference(csv_path, _Args.euler)
    s_ref = arclength(ref["pos"])

    # contour error: distance to the path *set* (global nearest point)
    e_c, _, _ = polyline_distance(pos, ref["pos"])

    # progress: globally monotone alignment (self-intersecting paths, see docstring)
    fidx = dtw_progress(pos, ref["pos"], t=t)
    s_exec = np.interp(fidx, np.arange(len(s_ref)), s_ref)
    t_matched = np.interp(fidx, np.arange(len(ref["t"])), ref["t"])

    # temporal lag against the start-up-aligned reference clock
    lag0 = startup_lag({"t": t, "pos": pos}, ref)
    tau = (t - lag0) - t_matched
    s_ref_now = np.interp(np.clip(t - lag0, ref["t"][0], ref["t"][-1]),
                          ref["t"], s_ref)
    lag_m = s_ref_now - s_exec

    speed = np.linalg.norm(vel, axis=1)
    dt_s = float(np.median(np.diff(t))) if len(t) > 1 else 1e-3

    m = {
        "mcap": os.path.relpath(mcap_path, HERE),
        "csv": os.path.relpath(csv_path, HERE),
        "trim": trim,
        "trigger": start_src,
        "duration_s": float(t[-1] - t[0]),
        "ref_duration_s": float(ref["t"][-1] - ref["t"][0]),
        "path_length_m": float(s_ref[-1]),
        "startup_lag_s": lag0,
        # contour error
        "contour_rmse_m": float(np.sqrt(np.mean(e_c ** 2))),
        "contour_mean_m": float(np.mean(e_c)),
        "contour_p95_m": float(np.percentile(e_c, 95)),
        "contour_max_m": float(e_c.max()),
        # progress / lag
        "lag_rms_s": float(np.sqrt(np.mean(tau ** 2))),
        "lag_median_s": float(np.median(tau)),
        "lag_p95_s": float(np.percentile(tau, 95)),
        "lag_end_s": float(tau[-1]),
        "lag_rms_m": float(np.sqrt(np.mean(lag_m ** 2))),
        "completion_frac": float(s_exec[-1] / s_ref[-1]),
        # aggressiveness
        "v_mean_m_s": float(speed.mean()),
        "v_max_m_s": float(speed.max()),
        "v_ref_max_m_s": float(np.linalg.norm(ref["vel"], axis=1).max()),
        "a_ref_max_m_s2": float(np.linalg.norm(ref["acc"], axis=1).max()),
        "tilt_p95_deg": float(np.percentile(tilt, 95)),
        "tilt_max_deg": float(tilt.max()),
    }
    if bootstrap:
        lo, hi = block_bootstrap_rmse(e_c, dt_s)
        m["contour_rmse_ci_m"] = [lo, hi]

    # solver telemetry (absent when /mpc is muted, e.g. the `large`-tier
    # nominal recordings) — every consumer reads these keys with `.get`.
    mpc = data.get("/mpc", [])
    if mpc:
        st = col(mpc, "solve_time_us")
        it = col(mpc, "iterations")
        cv = np.array([bool(x.get("converged")) for x in mpc])
        tick = np.diff(col(mpc, "timestamp_ns")) / 1e6  # ms
        m.update({
            "solve_mean_ms": float(st.mean() / 1000),
            "solve_p99_ms": float(np.percentile(st, 99) / 1000),
            "solve_max_ms": float(st.max() / 1000),
            "iters_mean": float(it.mean()),
            "iters_max": int(it.max()),
            "converged_frac": float(cv.mean()),
            "tick_median_ms": float(np.median(tick)) if len(tick) else float("nan"),
            "tick_rate_hz": float(1000 / np.median(tick)) if len(tick) else float("nan"),
            "mpc_n": len(mpc),
        })

    mot = data.get("/motors", [])
    t_u = sat = u = None
    if mot:
        u = np.array([x["motor_commands"] for x in mot], dtype=float)
        t_u = (col(mot, "timestamp_ns") - rise) / 1e9
        sat = (u >= SAT_THRESHOLD).any(1)
        per_motor = u.mean(0)
        m.update({
            "u_mean": float(u.mean()),
            "u_p95": float(np.percentile(u, 95)),
            "u_max": float(u.max()),
            "sat_frac": float(sat.mean()),
            "u_per_motor": [float(v) for v in per_motor],
            # Spread of the per-motor means: the steady trim the allocator
            # is holding. It rises when the added mass sits off the centre
            # of gravity, so it separates "heavier" from "heavier and
            # off-centre" without any extra instrumentation.
            "u_trim_asym": float(per_motor.max() - per_motor.min()),
        })

    if series:
        m["_series"] = {
            "t": t, "pos": pos, "e_c": e_c, "tau": tau,
            "s_exec": s_exec, "s_ref_total": s_ref[-1],
            "ref_pos": ref["pos"], "speed": speed, "tilt": tilt,
            "solve_ms": col(mpc, "solve_time_us") / 1000 if mpc else None,
            "t_u": t_u, "sat": sat, "u": u,
        }
    return m


# ── run catalogue ────────────────────────────────────────────────────────

def reference_for(shape, speed):
    return os.path.join(DATASETS, f"indoor_exp_{speed}", f"exp_{shape}_{speed}.csv")


def catalogue(datasets=DATASETS):
    """[(key, shape, speed, variant, mcap, csv)] for every run on disk.

    `variant` is "nominal" for the indoor_exp_<speed> datasets and the
    suffix (load / iter2 / timecost) for indoor_exp_ablation. Ablation
    runs borrow the reference of their (shape, speed) nominal run — the
    ablations change the *controller*, never the planned trajectory.
    """
    runs = []
    for speed in SPEEDS:
        d = os.path.join(datasets, f"indoor_exp_{speed}")
        if not os.path.isdir(d):
            continue
        for shape in SHAPES:
            mcap = os.path.join(d, f"exp_{shape}_{speed}.mcap")
            csv = reference_for(shape, speed)
            if os.path.exists(mcap) and os.path.exists(csv):
                runs.append((f"{shape}/{speed}/nominal", shape, speed, "nominal",
                             mcap, csv))
    abl = os.path.join(datasets, "indoor_exp_ablation")
    if os.path.isdir(abl):
        for f in sorted(os.listdir(abl)):
            if not f.endswith(".mcap"):
                continue
            parts = f[:-5].split("_")          # exp_<shape>_<speed>_<variant>
            shape, speed, variant = parts[1], parts[2], "_".join(parts[3:])
            csv = reference_for(shape, speed)
            if not os.path.exists(csv):
                print(f"note: {f}: no reference {os.path.basename(csv)} — skipped")
                continue
            runs.append((f"{shape}/{speed}/{variant}", shape, speed, variant,
                         os.path.join(abl, f), csv))
    return runs


def compute_all(datasets=DATASETS, bootstrap=True, verbose=True):
    out = {}
    for key, shape, speed, variant, mcap, csv in catalogue(datasets):
        try:
            m = analyse(mcap, csv, bootstrap=bootstrap)
        except Exception as e:                      # noqa: BLE001 — report and go on
            print(f"ERR {key}: {e}")
            continue
        m.update(shape=shape, speed=speed, variant=variant)
        out[key] = m
        if verbose:
            print(f"ok  {key:<28} contour RMSE {m['contour_rmse_m']*100:5.1f} cm")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", default=DATASETS)
    ap.add_argument("--dump", metavar="JSON", help="write every metric to JSON")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print a human-readable table")
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="skip the (slow) bootstrap CI on the contour RMSE")
    args = ap.parse_args()

    res = compute_all(args.datasets, bootstrap=not args.no_bootstrap)
    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {args.dump} ({len(res)} runs)")
    if args.show:
        hdr = (f"{'run':<28}{'RMSE cm':>9}{'p95':>7}{'max':>7}{'lag s':>8}"
               f"{'dur s':>8}{'vmax':>7}{'solve ms':>10}{'sat %':>7}")
        print(hdr, "-" * len(hdr), sep="\n")
        for k in sorted(res):
            m = res[k]
            print(f"{k:<28}{m['contour_rmse_m']*100:>9.1f}{m['contour_p95_m']*100:>7.1f}"
                  f"{m['contour_max_m']*100:>7.1f}{m['lag_rms_s']:>8.3f}"
                  f"{m['duration_s']:>8.2f}{m['v_max_m_s']:>7.2f}"
                  f"{m.get('solve_mean_ms', float('nan')):>10.2f}"
                  f"{m.get('sat_frac', float('nan'))*100:>7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
