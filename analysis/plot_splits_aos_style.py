#!/usr/bin/env python3
"""Splits-track experiment plotted in the style of the AOS/TOGT paper figure.

Reproduces the top-down panel of `aos_plots/fig_togt/plot_exp_togt_aos_cpc.py`
for a cybflight indoor recording: the seven gates of the UTIAS splits track,
the planned trajectory as a dashed line, and the flown trajectory drawn as a
speed-coloured ribbon with an inset colour bar.

Two figures:

  1. **xy**    top-down view (the only view the AOS figure needs here).
  2. **rmse**  evolution of the geometric tracking error over the mission:
               the running (cumulative) RMSE of the contour error `e_c(t)`
               (distance to the nearest point on the reference *polyline*),
               against the whole-run RMSE it converges to.

Both use the metric core in `paper_metrics.py`, so the mission segment
(trigger -> MISSION_IDLE) and the contour-error definition are identical to
the paper tables.

    python3 analysis/plot_splits_aos_style.py                       # show
    python3 analysis/plot_splits_aos_style.py --save docs/img/splits_timeopt
    python3 analysis/plot_splits_aos_style.py --mcap A.mcap --csv A.csv

The gate poses are those of the AOS figure's `race_0323_slits.yaml`; they are
inlined here so the script does not depend on that repository. Gates are
vertical squares, so in the top-down view each one projects to a segment.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_metrics as pm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# UTIAS splits track — position [m] and orientation [deg] of each gate,
# from aos_plots/fig_togt/data/utias_exp/race_0323_slits.yaml and the
# `orientations` table of plot_exp_togt_aos_cpc.py.
GATES = [
    ((-0.3267, -2.231, 1.6), (0.0, -90.0, -105.0)),
    ((-1.845, 1.942, 1.0), (0.0, -90.0, 85.0)),
    ((2.292, 1.637, 1.0), (0.0, -90.0, -60.0)),
    ((2.547, -2.108, 1.8), (0.0, -90.0, 90.0)),
    ((2.547, -2.108, 0.8), (0.0, -90.0, 90.0)),
    ((0.3099, 0.3554, 1.0), (0.0, -90.0, 0.0)),
    ((-2.396, -2.214, 1.0), (0.0, -90.0, -75.0)),
]
GATE_HALF = 0.6  # half side of the square gate [m]
CBAR_PAD_M = 0.9  # right margin kept clear for the inset colour bar [m]

# Speed ramp of the AOS figure: yellow (slow) -> red (fast).
C_SLOW, C_FAST = (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)
C_PLANNED = "#A20"


def rpy_to_rotation_matrix(rpy):
    """ZYX (yaw-pitch-roll) rotation matrix from degrees — same convention
    as the AOS figure, kept verbatim so the gates land identically."""
    r, p, y = np.array(rpy) * np.pi / 180
    cos, sin = np.cos, np.sin
    R = np.eye(3)
    R[0, 0] = cos(y) * cos(p)
    R[1, 0] = sin(y) * cos(p)
    R[2, 0] = -sin(p)
    R[0, 1] = cos(y) * sin(p) * sin(r) - sin(y) * cos(r)
    R[1, 1] = sin(y) * sin(p) * sin(r) + cos(y) * cos(r)
    R[2, 1] = cos(p) * sin(r)
    R[0, 2] = cos(y) * sin(p) * cos(r) + sin(y) * sin(r)
    R[1, 2] = sin(y) * sin(p) * cos(r) - cos(y) * sin(r)
    R[2, 2] = cos(p) * cos(r)
    return R


def gate_color(position):
    """The AOS figure's position-keyed gate shading, so neighbouring gates
    stay visually distinguishable."""
    rgb = [0.3, 0.3, -0.1] + np.array([0.1, 0.1, 0.4]) * np.array(position)
    return np.clip(rgb, 0.0, 1.0)


def plot_gates(ax):
    for position, rpy in GATES:
        r = GATE_HALF
        verts = np.array([[-r, r, 0], [-r, -r, 0], [r, -r, 0],
                          [r, r, 0], [-r, r, 0]])
        verts = verts @ rpy_to_rotation_matrix(rpy).T + np.array(position)
        ax.plot(verts[:, 0], verts[:, 1], "-", lw=3.0,
                color=gate_color(position), zorder=1, solid_capstyle="round")


def resample_by_arclength(pos, extra, n):
    """`n` samples uniform in path length, with `extra` (a 1-D signal, e.g.
    speed) carried along. Evens out the colour ribbon where the vehicle
    dwells, exactly like the AOS figure's 500-point resampling."""
    s = pm.arclength(pos)
    if s[-1] <= 0 or n >= len(pos):
        return pos, extra
    su = np.linspace(0.0, s[-1], n)
    return (np.stack([np.interp(su, s, pos[:, i]) for i in range(3)], 1),
            np.interp(su, s, extra))


def plot_trajectory(ax, ref_pos, pos, speed, v0, v1, cmap):
    ax.plot(ref_pos[:, 0], ref_pos[:, 1], "--", color=C_PLANNED, lw=1.0,
            zorder=0)
    seg = np.stack([pos[:-1, :2], pos[1:, :2]], 1)
    lc = LineCollection(seg, cmap=cmap, norm=Normalize(v0, v1),
                        linewidth=2.0, zorder=2)
    lc.set_array((speed[:-1] + speed[1:]) / 2)
    ax.add_collection(lc)
    return lc


def running_rmse(e):
    return np.sqrt(np.cumsum(e ** 2) / np.arange(1, len(e) + 1))


def main():
    ds = os.path.join(HERE, "datasets", "indoor_exp_timeopt")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mcap", default=os.path.join(ds, "exp_splits_timeopt.mcap"))
    p.add_argument("--csv", default=os.path.join(ds, "exp_splits_timeopt.csv"))
    p.add_argument("--save", metavar="PREFIX",
                   help="write PREFIX_xy.{pdf,png} and PREFIX_rmse.{pdf,png}")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--vmax", type=float, help="top of the speed colour ramp "
                   "[m/s] (default: the run's peak, rounded up)")
    p.add_argument("--resample", type=int, default=1500,
                   help="samples of the colour ribbon, uniform in path length")
    p.add_argument("--no-gates", action="store_true")
    args = p.parse_args()

    for f in (args.mcap, args.csv):
        if not os.path.exists(f):
            sys.exit(f"missing {f}")

    m = pm.analyse(args.mcap, args.csv, bootstrap=False, series=True)
    s = m["_series"]
    t, pos, e_c, speed, ref_pos = s["t"], s["pos"], s["e_c"], s["speed"], s["ref_pos"]

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    matplotlib.rc("font", size=18)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    v0 = 0.0
    v1 = args.vmax if args.vmax else float(np.ceil(speed.max()))
    cmap = LinearSegmentedColormap.from_list("", [C_SLOW, C_FAST])

    # ── 1. top-down view ────────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    if not args.no_gates:
        plot_gates(ax)
    rp, rs = resample_by_arclength(pos, speed, args.resample)
    lc = plot_trajectory(ax, ref_pos, rp, rs, v0, v1, cmap)

    lo = np.minimum(pos[:, :2].min(0), ref_pos[:, :2].min(0)) - 1.0
    hi = np.maximum(pos[:, :2].max(0), ref_pos[:, :2].max(0)) + 1.0
    ax.set_xlim(lo[0], hi[0] + CBAR_PAD_M)
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.grid(alpha=0.4)

    axins = inset_axes(ax, width="5%", height="45%", loc="lower left",
                       bbox_to_anchor=(0.94, 0.03, 1, 1),
                       bbox_transform=ax.transAxes, borderpad=0)
    cb = fig1.colorbar(lc, cax=axins)
    cb.set_ticks(np.arange(0, v1 + 0.1, 2))
    label = ax.annotate("Speed\n[m/s]", (0.995, 0.50), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=16)
    label.set_bbox(dict(facecolor="white", alpha=1, edgecolor="white"))
    ax.plot(np.nan, np.nan, "--", color=C_PLANNED, lw=2, label="Reference")
    ax.plot(np.nan, np.nan, "-", color="#F40", lw=4, label="Executed")
    ax.legend(loc="upper left", fontsize=15, framealpha=0.9)

    # ── 2. geometric tracking error over time ───────────────────────────
    rmse = m["contour_rmse_m"]
    run = running_rmse(e_c)
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    ax2.plot(t, run, color="#A20", lw=2.2, zorder=2, label="running RMSE")
    ax2.axhline(rmse, color="k", ls=":", lw=1.5, zorder=3,
                label=f"whole run: {rmse * 100:.1f} cm")
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(0, 1.25 * float(run.max()))
    ax2.set_xlabel("$t$ [s]")
    ax2.set_ylabel("Geometric RMSE [m]")
    ax2.grid(alpha=0.4)
    ax2.legend(fontsize=15, loc="lower right", framealpha=0.9)
    fig2.tight_layout()

    print(f"{os.path.basename(args.mcap)}: {m['duration_s']:.2f} s "
          f"(plan {m['ref_duration_s']:.2f} s), peak {speed.max():.2f} m/s, "
          f"contour RMSE {rmse * 100:.1f} cm, p95 {m['contour_p95_m'] * 100:.1f} cm, "
          f"max {m['contour_max_m'] * 100:.1f} cm")

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
        for fig, tag in ((fig1, "xy"), (fig2, "rmse")):
            for ext in ("pdf", "png"):
                out = f"{args.save}_{tag}.{ext}"
                fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
                print("wrote", out)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
