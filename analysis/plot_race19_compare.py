#!/usr/bin/env python3
"""Comparison of the time-optimal plans on the 19-gate splits race.

Same visual language as `plot_splits_aos_style.py` — gates, equal-aspect
view, trajectories drawn as speed-coloured ribbons — but with one ribbon
(and one colour bar) per planner, so the plans are told apart by hue while
the shared speed axis keeps them quantitatively comparable.

    python3 analysis/plot_race19_compare.py                        # top-down
    python3 analysis/plot_race19_compare.py --view 3d              # 3-D
    python3 analysis/plot_race19_compare.py --view both --no-show \
        --save docs/img/race19_compare                # PREFIX{,_3d}.{pdf,png}
    python3 analysis/plot_race19_compare.py --series ours_wgate cpc --vmin 4

Inputs are planner CSVs (`t, p_*, v_*, …`) — the same format as the mission
references — from `datasets/comparison/`. All of them cross the seven gates
of the splits track (the 19 waypoints are ~2.7 laps of it), so the gate poses
of `plot_splits_aos_style` are reused: short bars seen from above, 0.7 m
square frames in 3-D (`--gate-inner`), both in the same orientation.
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_splits_aos_style as aos  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARISON = os.path.join(HERE, "datasets", "comparison")

# One ramp per plan: dark end = fast. `flat` is the solid legend colour.
SERIES = {
    "ours": {"label": "Ours", "file": "ours_race19.csv",
             "ramp": [(0.0, 1.0, 0.6), (0.0, 0.0, 1.0)], "flat": "#1F6FD0"},
    "ours_wgate": {"label": "Ours (w/ gate)", "file": "ours_race19_wgate.csv",
                   "ramp": [(1.0, 1.0, 0.0), (1.0, 0.0, 0.0)], "flat": "#F08000"},
    "cpc": {"label": "CPC", "file": "cpc_race19.csv",
            "ramp": [(0.75, 0.75, 0.75), (0.0, 0.0, 0.0)], "flat": "#4D4D4D"},
}

# Gate frame: 0.7 m square opening, drawn as four bars of `GATE_BAR` width
# and `GATE_DEPTH` depth, in the gate plane of `aos.GATES`.
GATE_INNER, GATE_BAR, GATE_DEPTH = 0.7, 0.09, 0.05


def load_plan(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    g = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    return {"t": g("t"),
            "pos": np.stack([g("p_x"), g("p_y"), g("p_z")], 1),
            "speed": np.linalg.norm(
                np.stack([g("v_x"), g("v_y"), g("v_z")], 1), axis=1)}


def gate_bars(position, rpy, inner, bar, depth):
    """The four bars of one gate frame as (8, 3) vertex boxes.

    Built in the gate's own plane (local z = 0, the plane `aos.GATES` puts
    the square in) and mapped to the world by the same rotation the
    top-down bars use, so the 2-D and 3-D gates are the same object.
    """
    a, b, d = inner / 2, inner / 2 + bar, depth / 2
    spans = [((-b, b), (a, b)), ((-b, b), (-b, -a)),      # top, bottom
             ((-b, -a), (-a, a)), ((a, b), (-a, a))]      # left, right
    R = aos.rpy_to_rotation_matrix(rpy)
    out = []
    for (x0, x1), (y0, y1) in spans:
        box = np.array([[x, y, z] for z in (-d, d)
                        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])
        out.append(box @ R.T + np.array(position))
    return out


def box_faces(v):
    """The six quads of a box whose vertices are ordered bottom ring (0-3)
    then top ring (4-7), both counter-clockwise."""
    return [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[5], v[4]], [v[1], v[2], v[6], v[5]],
            [v[2], v[3], v[7], v[6]], [v[3], v[0], v[4], v[7]]]


def ribbon(pos, speed, cmap, norm, lw, zorder, three_d):
    """Speed-coloured LineCollection through `pos` (xy, or xyz in 3-D)."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    p = pos if three_d else pos[:, :2]
    seg = np.stack([p[:-1], p[1:]], 1)
    cls = Line3DCollection if three_d else LineCollection
    lc = cls(seg, cmap=cmap, norm=norm, linewidth=lw, zorder=zorder)
    lc.set_array((speed[:-1] + speed[1:]) / 2)
    return lc


def colorbars(fig, mappables, keys, x0, y0, w, h, fw, fh, gap=0.08):
    """Column of slim colour bars, ticks and label on the outermost only."""
    for i, (lc, key) in enumerate(zip(mappables, keys)):
        cax = fig.add_axes([(x0 + i * (w + gap)) / fw, y0 / fh, w / fw, h / fh])
        cb = fig.colorbar(lc, cax=cax)
        cb.outline.set_linewidth(0.8)
        if i == len(mappables) - 1:
            cb.set_label("Speed [m/s]", labelpad=8)
        else:
            cax.set_yticklabels([])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", nargs="+", default=list(SERIES),
                   choices=list(SERIES), metavar="NAME",
                   help=f"plans to draw, in order ({', '.join(SERIES)})")
    p.add_argument("--datasets", default=COMPARISON)
    p.add_argument("--view", choices=("2d", "3d", "both"), default="2d")
    p.add_argument("--save", metavar="PREFIX",
                   help="write PREFIX.{pdf,png} (2-D) and PREFIX_3d.{pdf,png}")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, help="default: peak of all, rounded up")
    p.add_argument("--resample", type=int, default=1500)
    p.add_argument("--lw", type=float, default=2.5, help="ribbon width")
    p.add_argument("--elev", type=float, default=76.0, help="3-D elevation [deg]")
    p.add_argument("--azim", type=float, default=40.0, help="3-D azimuth [deg]")
    p.add_argument("--z-stretch", type=float, default=1.0, dest="z_stretch",
                   help="vertical exaggeration of the 3-D box (1 = true scale)")
    p.add_argument("--z-axis", choices=("auto", "on", "off"), default="auto",
                   dest="z_axis", help="3-D z ticks and label; 'auto' drops "
                   "them above 60 deg elevation, where they overlap the box")
    p.add_argument("--zoom", type=float, default=1.10,
                   help="how much of the 3-D axes the projected box fills")
    p.add_argument("--gate-inner", type=float, default=GATE_INNER,
                   help="gate opening [m] (square)")
    p.add_argument("--no-gates", action="store_true")
    args = p.parse_args()

    plans = {}
    for key in args.series:
        path = os.path.join(args.datasets, SERIES[key]["file"])
        if not os.path.exists(path):
            sys.exit(f"missing {path}")
        plans[key] = load_plan(path)

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    matplotlib.rc("font", size=18)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    v1 = args.vmax if args.vmax else float(np.ceil(max(
        pl["speed"].max() for pl in plans.values())))
    norm = Normalize(args.vmin, v1)
    cmaps = {k: LinearSegmentedColormap.from_list(k, SERIES[k]["ramp"])
             for k in args.series}
    handles = [plt.Line2D([], [], color=SERIES[k]["flat"], lw=4,
                          label=SERIES[k]["label"]) for k in args.series]
    drawn = {k: aos.resample_by_arclength(plans[k]["pos"], plans[k]["speed"],
                                          args.resample) for k in args.series}
    frames = [] if args.no_gates else [
        gate_bars(pos, rpy, args.gate_inner, GATE_BAR, GATE_DEPTH)
        for pos, rpy in aos.GATES]

    cbw, cbgap, cbpad = 0.20, 0.08, 0.30       # colour-bar column geometry
    ncb = len(args.series)
    figs = {}

    # ── top-down ────────────────────────────────────────────────────────
    if args.view in ("2d", "both"):
        xy = np.concatenate([drawn[k][0][:, :2] for k in args.series])
        lo, hi = xy.min(0) - 0.8, xy.max(0) + 0.8
        dw, dh = hi - lo
        ah = 6.0
        aw = ah * dw / dh
        ml, mr, mb, mt = 1.05, 1.35, 0.95, 0.75
        fw = ml + aw + cbpad + ncb * (cbw + cbgap) + mr
        fh = mb + ah + mt
        fig = plt.figure(figsize=(fw, fh))
        ax = fig.add_axes([ml / fw, mb / fh, aw / fw, ah / fh])

        # Seen from above a gate is a bar: the projection of its opening.
        if not args.no_gates:
            r = args.gate_inner / 2
            square = np.array([[-r, r, 0], [-r, -r, 0], [r, -r, 0],
                               [r, r, 0], [-r, r, 0]])
            for position, rpy in aos.GATES:
                v = square @ aos.rpy_to_rotation_matrix(rpy).T + np.array(position)
                ax.plot(v[:, 0], v[:, 1], "-", lw=4.0, color="k", zorder=4,
                        solid_capstyle="butt")
        mappables = []
        for i, k in enumerate(args.series):
            lc = ribbon(*drawn[k], cmaps[k], norm, args.lw, 2 + i, False)
            ax.add_collection(lc)
            mappables.append(lc)

        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.grid(alpha=0.4)
        ax.legend(handles=handles, ncol=len(handles), loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1.4,
                  handlelength=1.6)
        colorbars(fig, mappables, args.series, ml + aw + cbpad, mb, cbw, ah,
                  fw, fh, cbgap)
        figs[""] = fig

    # ── 3-D ─────────────────────────────────────────────────────────────
    if args.view in ("3d", "both"):
        pts = [drawn[k][0] for k in args.series]
        pts += [box for bars in frames for box in bars]
        allp = np.concatenate(pts)
        lo, hi = allp.min(0) - 0.4, allp.max(0) + 0.4
        span = hi - lo

        zaxis = (abs(args.elev) <= 60.0 if args.z_axis == "auto"
                 else args.z_axis == "on")
        ah, aw = 6.4, 8.0
        ml, mr, mb, mt = 0.15, 1.45, 0.15, 0.55
        cbpad3 = cbpad + (0.55 if zaxis else 0.05)   # clear of the z tick labels
        fw = ml + aw + cbpad3 + ncb * (cbw + cbgap) + mr
        fh = mb + ah + mt
        fig = plt.figure(figsize=(fw, fh))
        ax = fig.add_axes([ml / fw, mb / fh, aw / fw, ah / fh],
                          projection="3d")

        for bars in frames:
            ax.add_collection3d(Poly3DCollection(
                [f for box in bars for f in box_faces(box)],
                facecolor="0.55", edgecolor="0.35", linewidths=0.4,
                alpha=0.45, zorder=1))
        mappables = []
        for i, k in enumerate(args.series):
            lc = ribbon(*drawn[k], cmaps[k], norm, args.lw, 10 + i, True)
            ax.add_collection3d(lc)
            mappables.append(lc)

        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((span[0], span[1], span[2] * args.z_stretch),
                          zoom=args.zoom)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_xlabel("$x$ [m]", labelpad=14)
        ax.set_ylabel("$y$ [m]", labelpad=14)
        if zaxis:
            ax.set_zlabel("$z$ [m]", labelpad=6)
        else:
            # Keep the ticks (an empty z axis breaks savefig's tight bbox),
            # just silence them — near top-down they land on the box.
            ax.set_zticklabels([])
            ax.tick_params(axis="z", length=0)
            for line in ax.zaxis.get_ticklines():
                line.set_visible(False)
        ax.tick_params(pad=4)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=handles, ncol=len(handles), loc="lower center",
                  bbox_to_anchor=(0.5, 0.97), frameon=False,
                  columnspacing=1.4, handlelength=1.6)
        colorbars(fig, mappables, args.series, ml + aw + cbpad3,
                  mb + 0.1 * ah, cbw, 0.8 * ah, fw, fh, cbgap)
        figs["_3d"] = fig

    for k in args.series:
        pl = plans[k]
        print(f'{SERIES[k]["label"]:>15}: {pl["t"][-1]:.3f} s, '
              f'{aos.pm.arclength(pl["pos"])[-1]:.1f} m, '
              f'mean {pl["speed"].mean():.2f} m/s, peak {pl["speed"].max():.2f} m/s')

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
        for suffix, fig in figs.items():
            for ext in ("pdf", "png"):
                out = f"{args.save}{suffix}.{ext}"
                fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
                print("wrote", out)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
