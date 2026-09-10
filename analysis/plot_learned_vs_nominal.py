#!/usr/bin/env python3
"""Compare a flown track under the nominal MPC cost against the learned one.

The two datasets are flights of the *same* planned trajectory — the
reference CSV is byte-identical in both directories — so the only thing
that differs is the cost the MPC ran with:

    datasets/indoor_exp_timeopt/exp_<track>_timeopt.mcap          nominal
    datasets/indoor_exp_timeopt_learned/exp_<track>_timeopt_learned.mcap

Pick a track by shape name and nothing else:

    python3 analysis/plot_learned_vs_nominal.py splits
    python3 analysis/plot_learned_vs_nominal.py figure8 --save out --no-show

Run it with no track to list the shapes that have both recordings on
disk; `circle` and `slalom` join the list by themselves the moment a
learned `.mcap` for them is dropped into the learned dataset directory.

Two figures, both showing the shared reference plus each run:

  <track>_2d.png   top-down XY path
  <track>_3d.png   the same paths in 3D

and, printed to stdout, the geometric tracking RMSE of each run — the
whole-mission RMS distance from the executed position to the reference
*path*, timing-free — alongside the contouring / progress split and the
temporal RMSE, all computed by the same helpers `plot_dataset.py` uses,
so the numbers are directly comparable with that script's output.

Requires: pip install mcap cbor2 numpy matplotlib
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_dataset as pdd  # noqa: E402
from paper_metrics import _Args  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(HERE, "datasets")

# (label, dataset directory, mcap/csv basename suffix, colour)
VARIANTS = [
    ("nominal", "indoor_exp_timeopt", "exp_{track}_timeopt", "#1f77b4"),
    ("learned", "indoor_exp_timeopt_learned", "exp_{track}_timeopt_learned", "#d62728"),
]


def paths_for(track, datasets=DATASETS):
    """{label: (mcap, csv)} for every variant that has both files."""
    found = {}
    for label, d, stem, _ in VARIANTS:
        base = os.path.join(datasets, d, stem.format(track=track))
        if os.path.exists(base + ".mcap") and os.path.exists(base + ".csv"):
            found[label] = (base + ".mcap", base + ".csv")
    return found


def available_tracks(datasets=DATASETS):
    """Shapes with a recording *and* a reference under every variant."""
    per_variant = []
    for _, d, stem, _ in VARIANTS:
        dd = os.path.join(datasets, d)
        if not os.path.isdir(dd):
            return []
        prefix, suffix = stem.split("{track}")
        per_variant.append({
            f[len(prefix):-len(suffix + ".mcap")]
            for f in os.listdir(dd)
            if f.startswith(prefix) and f.endswith(suffix + ".mcap")
            and os.path.exists(os.path.join(dd, f[:-5] + ".csv"))
        })
    return sorted(set.intersection(*per_variant))


def load_run(label, colour, mcap_path, csv_path, args):
    flight = pdd.load_flight(mcap_path, args)
    ref = pdd.load_reference(csv_path, args.euler)
    lag = args.lag if args.lag is not None else pdd.find_lag(flight, ref)
    return {
        "label": label,
        "colour": colour,
        "flight": flight,
        "ref": ref,
        "lag": lag,
        "geom_rmse": pdd.geometric_rmse(flight["pos"], ref["pos"]),
        "temp_rmse": pdd.temporal_rmse(flight, ref, lag),
        "path": pdd.path_metrics(flight, ref),
    }


def check_shared_reference(runs):
    """The comparison only means anything if both flew the same plan."""
    a = runs[0]["ref"]["pos"]
    for r in runs[1:]:
        b = r["ref"]["pos"]
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-9):
            print(f"warning: {runs[0]['label']} and {r['label']} do not share the"
                  " same reference trajectory — the RMSE comparison is not"
                  " apples-to-apples", file=sys.stderr)
            return False
    return True


# ── reporting ────────────────────────────────────────────────────────────

# Headline metrics quoted as a nominal → learned delta, most important first.
DELTA_METRICS = [
    ("geometric RMSE", lambda r: r["geom_rmse"], 100, "cm"),
    ("contouring RMSE", lambda r: r["path"]["contour_rmse"], 100, "cm"),
    ("progress τ RMS", lambda r: r["path"]["tau_rms"], 1000, "ms"),
]


def report(track, runs):
    print(f"── {track} ── nominal vs learned MPC cost, same reference "
          f"({runs[0]['path']['path_length']:.1f} m plan)\n")
    head = (f"{'run':<10}{'geom RMSE [cm]':>16}{'contour RMSE [cm]':>19}"
            f"{'contour p95 [cm]':>18}{'progress τ [ms]':>17}"
            f"{'temporal RMSE [cm]':>20}{'flown [s]':>11}{'completed':>11}")
    print(head)
    print("─" * len(head))
    for r in runs:
        p, f = r["path"], r["flight"]
        print(f"{r['label']:<10}{r['geom_rmse']*100:>16.2f}{p['contour_rmse']*100:>19.2f}"
              f"{p['contour_p95']*100:>18.2f}{p['tau_rms']*1000:>17.1f}"
              f"{r['temp_rmse']*100:>20.2f}{f['segment_s']:>11.2f}"
              f"{p['completion']*100:>10.1f}%")
    if len(runs) == 2:
        base, other = runs
        for name, get, scale, unit in DELTA_METRICS:
            a, b = get(base), get(other)
            delta = (b - a) / a * 100 if a else float("nan")
            print(f"\n  {name:<16} {base['label']} {a*scale:.2f} {unit}"
                  f"  →  {other['label']} {b*scale:.2f} {unit}"
                  f"   ({delta:+.1f} %, {other['label']} is"
                  f" {'better' if b < a else 'worse'})")
    print()


# ── plotting ─────────────────────────────────────────────────────────────

def plot_2d(plt, track, runs):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ref = runs[0]["ref"]["pos"]
    ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1.2, label="reference", zorder=1)
    for r in runs:
        p = r["flight"]["pos"]
        ax.plot(p[:, 0], p[:, 1], color=r["colour"], lw=1.1, alpha=0.9,
                label=f'{r["label"]}  (geom. RMSE {r["geom_rmse"]*100:.1f} cm)')
        ax.scatter(*p[0, :2], color=r["colour"], marker="o", s=28, zorder=3)
        ax.scatter(*p[-1, :2], color=r["colour"], marker="s", s=28, zorder=3)
    ax.set_title(f"{track} — nominal vs learned cost (○ start, □ end)")
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def plot_3d(plt, track, runs):
    from matplotlib.ticker import MaxNLocator

    fig = plt.figure(figsize=(9, 5.6))
    ax = fig.add_subplot(projection="3d")
    ref = runs[0]["ref"]["pos"]
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], "k--", lw=1.1, label="reference")
    for r in runs:
        p = r["flight"]["pos"]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=r["colour"], lw=1.0, alpha=0.9,
                label=f'{r["label"]}  (geom. RMSE {r["geom_rmse"]*100:.1f} cm)')
        ax.scatter(*p[0], color=r["colour"], marker="o", s=28)
        ax.scatter(*p[-1], color=r["colour"], marker="s", s=28)
    # True 1:1:1 scale on every axis: each axis keeps its own data limits
    # (no wasted range) and the box side lengths are set proportional to
    # those spans, so a metre is the same length in x, y and z. These
    # missions are ~6 m wide and ~1 m tall, so the box comes out flat —
    # that is the real shape of the flown volume, not a rendering choice.
    allp = np.vstack([ref] + [r["flight"]["pos"] for r in runs])
    lo, hi = allp.min(0), allp.max(0)
    pad = (hi - lo).max() * 0.04          # equal absolute pad keeps the scale equal
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo))
    # a short axis gets few ticks, or the labels overprint each other
    span = hi - lo
    for axis, d in zip((ax.xaxis, ax.yaxis, ax.zaxis), span):
        axis.set_major_locator(MaxNLocator(
            nbins=int(np.clip(round(7 * d / span.max()), 2, 7))))
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    ax.set_zlabel("z up (m)")
    ax.set_title(f"{track} — nominal vs learned cost (○ start, □ end)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[2:]))
    ap.add_argument("track", nargs="?", help="trajectory shape, e.g. splits or figure8")
    ap.add_argument("--datasets", default=DATASETS, help="dataset root (default: analysis/datasets)")
    ap.add_argument("--save", metavar="PREFIX",
                    help="write PREFIX_<track>_2d.png and PREFIX_<track>_3d.png")
    ap.add_argument("--no-show", action="store_true", help="skip the interactive show")
    ap.add_argument("--lag", type=float, default=None,
                    help="fixed reference lag after the trigger [s] (default: auto per run)")
    args = ap.parse_args()

    tracks = available_tracks(args.datasets)
    if not args.track:
        print("specify a track to compare. available (both recordings present):")
        for t in tracks:
            print(f"  {t}")
        if not tracks:
            print("  (none — no shape has an .mcap in both dataset directories)")
        return 2
    if args.track not in tracks:
        sys.exit(f"no comparable recordings for {args.track!r}; available: "
                 + (", ".join(tracks) or "(none)"))

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    load_args = _Args()
    load_args.lag = args.lag

    found = paths_for(args.track, args.datasets)
    runs = [load_run(label, colour, *found[label], load_args)
            for label, _, _, colour in VARIANTS if label in found]
    check_shared_reference(runs)
    report(args.track, runs)

    figs = {"2d": plot_2d(plt, args.track, runs), "3d": plot_3d(plt, args.track, runs)}
    if args.save:
        for tag, fig in figs.items():
            out = f"{args.save}_{args.track}_{tag}.png"
            fig.savefig(out, dpi=130)
            print(f"saved {out}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
