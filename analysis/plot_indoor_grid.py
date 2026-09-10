#!/usr/bin/env python3
"""2x4 grid of flown vs. reference XY trajectories for the indoor experiments.

Rows (top to bottom):  the --speeds profiles (default mid, timeopt)
Cols (left to right):  the --shapes trajectories (default circle, figure8,
                       slalom, splits)

Reads analysis/datasets/indoor_exp_<speed>/exp_<shape>_<speed>.{mcap,csv}
via the loaders in plot_dataset.py. Every panel uses the same fixed,
equal-aspect XY box so the trajectories are directly comparable, and
names its own profile in the top-right corner (low / mid / high) —
the rows are only labelled by position otherwise, which stops being
obvious the moment a panel is read on its own or the row order is
changed with --speeds.

    python3 analysis/plot_indoor_grid.py                    # show (2x4)
    python3 analysis/plot_indoor_grid.py --save grid.png    # also write PNG
    python3 analysis/plot_indoor_grid.py --speeds slow mid timeopt     # 3x4
    python3 analysis/plot_indoor_grid.py --speeds timeopt --shapes figure8 splits
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_dataset as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ALL_SPEEDS = ["slow", "mid", "fast", "timeopt"]
ALL_SHAPES = ["circle", "figure8", "slalom", "splits"]
SHAPE_LABEL = {
    "circle": "Circle",
    "figure8": "Figure-8",
    "slalom": "Slalom",
    "splits": "Splits",
}
# What each profile is called on the panel. The dataset directories keep
# their own names (indoor_exp_slow, indoor_exp_timeopt); only the label
# changes, so an unmapped profile prints as itself.
SPEED_LABEL = {"slow": "low", "timeopt": "high"}


def seat_legend(fig, axes, leg, gap_in=0.04):
    """Drop the legend to `gap_in` inches above the top row of panels.

    `bbox_to_anchor` is in figure coordinates, so anchoring at the top of
    the figure leaves whatever margin the layout engine happened to
    reserve above the axes — and `compact_height` then takes its trim out
    of the figure height, which widens that margin again. Measuring the
    top row once the panels have settled is what makes the gap the size
    it is asked to be.
    """
    fig.canvas.draw()
    # The panel geometry has converged; freeze it so the draw that
    # follows cannot move the axes out from under the measurement.
    fig.set_layout_engine("none")
    r = fig.canvas.get_renderer()
    top = max(a.get_tightbbox(r).y1 for a in axes[0]) / fig.dpi
    leg.set_bbox_to_anchor(
        (0.5, (top + gap_in) / fig.get_size_inches()[1]), transform=fig.transFigure
    )


def compact_height(fig, axes, gap_in=0.05, iters=5):
    """Squeeze the vertical whitespace between panel rows to `gap_in` inches.

    The panels are equal-aspect and their width is fixed by the column
    layout, so their height is fixed too — any surplus figure height turns
    into row gaps instead of bigger panels. Measure the real gaps (tight
    bboxes, so tick labels count) and take the surplus out of the figure
    height; one or two passes converge.
    """
    nrow = axes.shape[0]
    if nrow < 2:
        return
    for _ in range(iters):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        rows = [[a.get_tightbbox(r) for a in row] for row in axes]
        gaps = [
            min(b.y0 for b in rows[i]) - max(b.y1 for b in rows[i + 1])
            for i in range(nrow - 1)
        ]
        surplus = sum(g - gap_in * fig.dpi for g in gaps) / fig.dpi
        if abs(surplus) < 0.02:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w, max(h - surplus, 1.0), forward=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", default=os.path.join(HERE, "datasets"))
    ap.add_argument(
        "--speeds",
        nargs="+",
        metavar="SPEED",
        default=["mid", "timeopt"],
        help="speed profiles = rows, top to bottom "
        f"(known: {' '.join(ALL_SPEEDS)}; default: mid timeopt)",
    )
    ap.add_argument(
        "--shapes",
        nargs="+",
        metavar="SHAPE",
        default=list(ALL_SHAPES),
        help="trajectory shapes = columns, left to right "
        f"(default: {' '.join(ALL_SHAPES)})",
    )
    ap.add_argument("--save", metavar="PNG")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        help="fixed x range [m] (default: common bounds + margin)",
    )
    ap.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        help="fixed y range [m] (default: common bounds + margin)",
    )
    ap.add_argument("--margin", type=float, default=0.5, help="auto-box margin [m]")
    ap.add_argument(
        "--label-gutter",
        type=float,
        default=0.6,
        metavar="M",
        help="headroom added to the top of the common box for the "
        "per-panel profile label [m] (default: 0.6; 0 to "
        "let the label sit over the trajectory)",
    )
    ap.add_argument(
        "--row-gap",
        type=float,
        default=0.05,
        metavar="IN",
        help="vertical gap between panel rows [inch] (default: 0.05)",
    )
    ap.add_argument(
        "--legend-gap",
        type=float,
        default=0.04,
        metavar="IN",
        help="gap between the legend and the top row [inch] (default: 0.04)",
    )
    ap.add_argument("--euler", default="tilt")
    ap.add_argument("--mission-channel", type=int, default=pd.MISSION_CHANNEL)
    ap.add_argument("--mission-high", type=int, default=pd.MISSION_HIGH_US)
    ap.add_argument("--mission-low", type=int, default=pd.MISSION_LOW_US)
    ap.add_argument("--throttle-channel", type=int, default=pd.THROTTLE_CHANNEL)
    ap.add_argument("--throttle-deadband", type=float, default=pd.THROTTLE_DEADBAND)
    args = ap.parse_args()
    speeds, shapes = args.speeds, args.shapes

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {}
    for speed in speeds:
        for shape in shapes:
            base = os.path.join(
                args.datasets, f"indoor_exp_{speed}", f"exp_{shape}_{speed}"
            )
            if not (os.path.exists(base + ".mcap") and os.path.exists(base + ".csv")):
                print(f"note: missing {base}.{{mcap,csv}} — panel left empty")
                continue
            runs[(speed, shape)] = (
                pd.load_flight(base + ".mcap", args),
                pd.load_reference(base + ".csv", args.euler),
            )

    if not runs:
        sys.exit("no <shape>/<speed> pairs found — check --speeds/--shapes/--datasets")

    # One common box for all panels.
    if args.xlim and args.ylim:
        xlim, ylim = args.xlim, args.ylim
    else:
        allp = np.concatenate(
            [
                np.concatenate([f["pos"][:, :2], r["pos"][:, :2]])
                for f, r in runs.values()
            ]
        )
        lo, hi = allp.min(0) - args.margin, allp.max(0) + args.margin
        xlim = args.xlim or (lo[0], hi[0])
        ylim = args.ylim or (lo[1], hi[1])
    # Panels are drawn rotated 90 deg CCW (horizontal axis = -y, vertical
    # axis = x) so the tall trajectories lie flat and the figure is short.
    # The box also carries a strip of headroom along the top: the profile
    # label goes in that corner, and on the slalom and splits panels the
    # end marker sits right where the text would otherwise land.
    hlim, vlim = (ylim[1], ylim[0]), (xlim[0], xlim[1] + args.label_gutter)
    w, h = ylim[1] - ylim[0], vlim[1] - vlim[0]

    ncol, nrow = len(shapes), len(speeds)
    # Figure shape matches the panel grid so equal-aspect boxes fill it
    # without leaving horizontal gaps (extra height for labels + legend).
    pw = 2.4
    # (figsize below: width from the horizontal extent, height from the vertical)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(pw * ncol + 0.5, pw * nrow * h / w + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    for i, speed in enumerate(speeds):
        for j, shape in enumerate(shapes):
            ax = axes[i][j]
            if (speed, shape) in runs:
                flight, ref = runs[(speed, shape)]
                ax.plot(
                    ref["pos"][:, 1], ref["pos"][:, 0], "k--", lw=1.0, label="reference"
                )
                ax.plot(
                    flight["pos"][:, 1],
                    flight["pos"][:, 0],
                    color="C0",
                    lw=1.0,
                    label="executed",
                )
                ax.scatter(
                    flight["pos"][0, 1],
                    flight["pos"][0, 0],
                    c="green",
                    s=25,
                    zorder=3,
                    label="start",
                )
                ax.scatter(
                    flight["pos"][-1, 1],
                    flight["pos"][-1, 0],
                    c="red",
                    s=25,
                    zorder=3,
                    label="end",
                )
            ax.set_xlim(*hlim)
            ax.set_ylim(*vlim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            # Which profile this panel is, said on the panel: the rows
            # are otherwise identified by position alone, which stops
            # being obvious as soon as one panel is read on its own.
            ax.text(
                0.98,
                0.97,
                SPEED_LABEL.get(speed, speed),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="0.25",
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                    boxstyle="square,pad=0.15",
                ),
            )
            if i == nrow - 1:
                ax.set_xlabel("y [m]")
            if j == 0:
                ax.set_ylabel("x [m]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    # Anchored above the axes; savefig's tight bbox expands to include it.
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        # The legend's own padding sits between the anchor and the text —
        # ~9 pt of it by default, which is most of the gap being asked
        # about here. Zero it and `seat_legend`'s number is the gap.
        borderpad=0.0,
        borderaxespad=0.0,
        bbox_to_anchor=(0.5, 1.005),
    )
    # Trim the surplus figure height that equal-aspect panels leave as row gaps.
    compact_height(fig, axes, gap_in=args.row_gap)
    seat_legend(fig, axes, leg, gap_in=args.legend_gap)

    if args.save:
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"saved {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
