#!/usr/bin/env python3
"""The paper's result figures, from `paper_metrics.py`.

Four figures, each carrying one argument that a table cannot:

  **tradeoff**   contour error vs. time lag, one marker per configuration.
      The single most compact statement of the contribution: the nominal
      controller and the time-sampled ablation sit on opposite sides of
      the same trade-off, at both speed profiles. A table of two numbers
      makes the reader do this comparison in their head; the plot does it
      for them.

  **ablation**   contour error along the path for every ablation variant,
      with actuator saturation shaded. Shows *where* on the trajectory
      the error is spent — the ablations do not degrade uniformly, they
      degrade in the corners — and lines up the degradation with the
      saturated intervals.

  **solvetime**  empirical CDF of solve time against the control-period
      budget. The plot that makes an embedded-systems reviewer believe
      the real-time claim, and the plot that shows why two SQP
      iterations is not a free improvement.

  **errcdf**     CDF of contour error for the twelve nominal flights,
      grouped by speed profile. Supports the headline accuracy claim
      with a distribution rather than a single moment, and makes the
      p95/max columns of Table I legible at a glance.

Figures are written as PDF (vector, for LaTeX) and PNG. Sizes are set for
a two-column IEEE layout: `--width 3.5` for a single column, `--width 7.16`
for a full-width figure.

Usage:
    python3 analysis/paper_figures.py --outdir paper/figs
    python3 analysis/paper_figures.py --only tradeoff --show
    python3 analysis/paper_figures.py --outdir figs --format png

Requires: pip install mcap cbor2 numpy matplotlib
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_metrics as pm  # noqa: E402

SHAPE_LABEL = {"circle": "Circle", "figure8": "Figure-8",
               "slalom": "Slalom", "splits": "Splits"}
SPEED_LABEL = {"slow": "Slow", "mid": "Mid", "timeopt": "Time-optimal"}
VARIANT_LABEL = {"nominal": "Proposed", "timecost": "Time-sampled cost",
                 "iter2": "2 SQP iterations", "load": "Unknown payload"}
VARIANT_COLOR = {"nominal": "#1b6ca8", "timecost": "#d1495b",
                 "iter2": "#e5a000", "load": "#3f8f52"}
VARIANT_MARK = {"nominal": "o", "timecost": "s", "iter2": "^", "load": "D"}
SPEED_COLOR = {"slow": "#7fb3d5", "mid": "#1b6ca8", "timeopt": "#12354f"}

CONTROL_PERIOD_MS = 10.0   # mpc_rate_hz = 100 on the flown vehicle


def style():
    import matplotlib
    matplotlib.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.0,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def ecdf(x):
    xs = np.sort(np.asarray(x))
    return xs, np.arange(1, len(xs) + 1) / len(xs)


# ── figures ──────────────────────────────────────────────────────────────

def fig_tradeoff(plt, res, series, width):
    """Contour error vs. time lag — the contribution in one panel.

    An arrow runs from the nominal controller to each ablation, so the
    reader sees *what each change does* rather than having to diff two
    scatter points. There is deliberately no "better" direction marked:
    the whole claim is that this is a trade-off, and that the proposed
    cost picks the path-fidelity side of it.
    """
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(width, width * 0.78))

    def xy(m):
        return m["lag_rms_s"] * 1000, m["contour_rmse_m"] * 100

    for speed in ("mid", "timeopt"):
        base = res.get(f"figure8/{speed}/nominal")
        if base is None:
            continue
        bx, by = xy(base)
        for variant in ("timecost", "iter2", "load"):
            m = res.get(f"figure8/{speed}/{variant}")
            if m is None:
                continue
            x, y = xy(m)
            ax.annotate("", xy=(x, y), xytext=(bx, by),
                        arrowprops=dict(arrowstyle="-|>", color="0.72", lw=0.7,
                                        shrinkA=6, shrinkB=6))
        for variant in ("nominal", "timecost", "iter2", "load"):
            m = res.get(f"figure8/{speed}/{variant}")
            if m is None:
                continue
            x, y = xy(m)
            ax.scatter(x, y, s=44, marker=VARIANT_MARK[variant],
                       facecolor=VARIANT_COLOR[variant],
                       edgecolor="white", linewidths=0.7, zorder=3)
        ax.annotate(f"{SPEED_LABEL[speed]} profile", xy=(bx, by),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=7, color="0.35")

    ax.legend(handles=[Line2D([], [], marker=VARIANT_MARK[v], ls="",
                              color=VARIANT_COLOR[v], markeredgecolor="white",
                              markeredgewidth=0.7, label=VARIANT_LABEL[v])
                       for v in ("nominal", "timecost", "iter2", "load")],
              loc="upper right", frameon=False, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.3)
    ax.set_xlabel(r"time lag  $\tau_\mathrm{rms}$  [ms]")
    ax.set_ylabel("contour error RMS  [cm]")
    ax.margins(0.16)
    fig.tight_layout()
    return fig


def _smooth(x, y, frac=0.004):
    """Light moving average over `frac` of the path, to expose structure
    without flattening the peaks the p95/max columns report."""
    n = max(1, int(round(frac * len(y))))
    if n < 2:
        return x, y
    k = np.ones(n) / n
    return x[n - 1:], np.convolve(y, k, mode="valid")


def fig_ablation(plt, res, series, width, variants=("nominal", "timecost")):
    """Contour error along the path, per ablation variant, saturation shaded.

    Defaults to the nominal controller against the time-sampled-cost
    ablation only: four overlaid traces on one axis is unreadable at
    column width, and this pair carries the argument. Pass more variants
    for a full-width version.
    """
    fig, axes = plt.subplots(2, 1, figsize=(width, width * 0.82), sharex=True)
    for ax, speed in zip(axes, ("mid", "timeopt")):
        base = series.get(f"figure8/{speed}/nominal")
        if base is not None:
            s_ = base["_series"]
            sat, t_u = s_["sat"], s_["t_u"]
            if sat is not None and sat.any():
                prog_u = np.interp(t_u, s_["t"], s_["s_exec"] / s_["s_ref_total"])
                edges = np.diff(sat.astype(int))
                starts = list(prog_u[1:][edges == 1])
                stops = list(prog_u[1:][edges == -1])
                if sat[0]:
                    starts.insert(0, prog_u[0])
                if sat[-1]:
                    stops.append(prog_u[-1])
                for a, b in zip(starts, stops):
                    ax.axvspan(a, b, color="0.88", lw=0, zorder=0)
        for variant in variants:
            r = series.get(f"figure8/{speed}/{variant}")
            if r is None:
                continue
            s_ = r["_series"]
            prog = s_["s_exec"] / s_["s_ref_total"]
            px, py = _smooth(prog, s_["e_c"] * 100)
            ax.plot(px, py, color=VARIANT_COLOR[variant], lw=1.0,
                    label=VARIANT_LABEL[variant])
        ax.set_ylabel("contour error [cm]")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.text(0.985, 0.92, f"{SPEED_LABEL[speed]} profile",
                transform=ax.transAxes, va="top", ha="right", fontsize=7.5,
                color="0.35")
    from matplotlib.patches import Patch
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="0.88", edgecolor="none"))
    labels.append("actuator saturation")
    axes[0].legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 1.36), ncol=len(variants) + 1,
                   frameon=False, handletextpad=0.5, columnspacing=1.2)
    axes[-1].set_xlabel("progress along planned path")
    fig.tight_layout()
    return fig


def fig_solvetime(plt, res, series, width):
    """CDF of solve time against the control-period budget."""
    fig, ax = plt.subplots(figsize=(width, width * 0.62))
    groups = {
        "1 SQP iteration (proposed)": ("#1b6ca8", "-", [
            "figure8/timeopt/nominal", "figure8/mid/load",
            "figure8/timeopt/load", "figure8/mid/timecost",
            "figure8/timeopt/timecost"]),
        "2 SQP iterations": ("#e5a000", "--", [
            "figure8/mid/iter2", "figure8/timeopt/iter2"]),
    }
    for label, (color, ls, keys) in groups.items():
        vals = np.concatenate([series[k]["_series"]["solve_ms"] for k in keys
                               if k in series
                               and series[k]["_series"]["solve_ms"] is not None])
        xs, ys = ecdf(vals)
        ax.plot(xs, ys * 100, ls, color=color, label=f"{label}  ($n$={len(vals)})")
        # left of each (near-vertical) CDF, so the labels clear the curves,
        # each other, and the control-period line
        ax.annotate(f"max\n{xs[-1]:.2f} ms", xy=(xs[0], 72), xytext=(-4, 0),
                    textcoords="offset points", fontsize=6.5, color=color,
                    va="center", ha="right")
    ax.axvspan(CONTROL_PERIOD_MS, CONTROL_PERIOD_MS * 2, color="#d1495b",
               alpha=0.07, lw=0, zorder=0)
    ax.axvline(CONTROL_PERIOD_MS, color="#d1495b", lw=1.0)
    ax.annotate(f"{CONTROL_PERIOD_MS:.0f} ms control period",
                xy=(CONTROL_PERIOD_MS, 50), xytext=(-3, 0),
                textcoords="offset points", fontsize=7, color="#d1495b",
                ha="center", va="center", rotation=90)
    ax.set_xlim(0, CONTROL_PERIOD_MS * 1.25)
    ax.set_ylim(0, 103)
    ax.set_xlabel("solve time [ms]")
    ax.set_ylabel("solves below [\\%]" if plt.rcParams["text.usetex"]
                  else "solves below [%]")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.30), frameon=False,
              handletextpad=0.5, labelspacing=0.3)
    fig.tight_layout()
    return fig


def fig_errcdf(plt, res, series, width):
    """CDF of contour error for the nominal flights, grouped by profile."""
    fig, ax = plt.subplots(figsize=(width, width * 0.62))
    for speed in pm.SPEEDS:
        vals = np.concatenate([series[f"{sh}/{speed}/nominal"]["_series"]["e_c"]
                               for sh in pm.SHAPES
                               if f"{sh}/{speed}/nominal" in series])
        xs, ys = ecdf(vals * 100)
        ax.plot(xs, ys * 100, color=SPEED_COLOR[speed],
                label=f"{SPEED_LABEL[speed]}  (RMS {np.sqrt(np.mean((xs)**2)):.1f} cm)")
        p95 = np.percentile(xs, 95)
        ax.plot([p95], [95], marker="o", ms=3.5, color=SPEED_COLOR[speed])
    ax.axhline(95, color="0.6", lw=0.6, ls=":")
    ax.annotate("p95", xy=(0.99, 95), xycoords=("axes fraction", "data"),
                fontsize=6.5, color="0.45", ha="right", va="bottom")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.set_xlabel("contour error [cm]")
    ax.set_ylabel("samples below [%]")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.5)
    fig.tight_layout()
    return fig


FIGURES = {"tradeoff": fig_tradeoff, "ablation": fig_ablation,
           "solvetime": fig_solvetime, "errcdf": fig_errcdf}

# Which figures need the (expensive) per-sample series kept in memory.
NEEDS_SERIES = {"ablation", "solvetime", "errcdf"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", default=pm.DATASETS)
    ap.add_argument("--outdir", metavar="DIR", default=None,
                    help="write fig_<name>.<fmt> here")
    ap.add_argument("--only", choices=sorted(FIGURES), action="append",
                    help="render only this figure (repeatable)")
    ap.add_argument("--format", default="pdf,png",
                    help="comma-separated output formats (default pdf,png)")
    ap.add_argument("--width", type=float, default=3.5,
                    help="figure width in inches (3.5 = IEEE column, "
                         "7.16 = full width)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()

    wanted = args.only or list(FIGURES)
    keep = any(w in NEEDS_SERIES for w in wanted)

    res, series = {}, {}
    for key, shape, speed, variant, mcap, csv in pm.catalogue(args.datasets):
        try:
            m = pm.analyse(mcap, csv, bootstrap=False, series=keep)
        except Exception as e:                       # noqa: BLE001
            print(f"ERR {key}: {e}")
            continue
        m.update(shape=shape, speed=speed, variant=variant)
        if keep:
            series[key] = m
        res[key] = {k: v for k, v in m.items() if k != "_series"}
    print(f"loaded {len(res)} runs")

    for name in wanted:
        fig = FIGURES[name](plt, res, series, args.width)
        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            for fmt in args.format.split(","):
                out = os.path.join(args.outdir, f"fig_{name}.{fmt.strip()}")
                fig.savefig(out)
                print(f"wrote {out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
