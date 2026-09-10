#!/usr/bin/env python3
"""Render tracking-vs-reference trajectory plots for the learned-cost study.

Reads target/sim-out/learned_cost/traces/<method>/trace__<mission>.json
(written by `tests/learned_cost_eval.rs` runs, one directory per method —
see docs/learned_mpc_cost.md) and writes, into
target/sim-out/learned_cost/plots/:

  xy_<mission>.png   - XY path: reference (dashed black) + every method,
                       with an altitude-vs-time inset and per-method
                       whole-run geometric RMS in the legend
  grid_overview.png  - one XY panel per mission, all methods

Methods and colors follow the study's naming:
  cur_r20   current flown tune (w_rate 20, no drag model)
  r10       w_rate 10, no drag model
  r10_drag  w_rate 10 + rotor-drag prediction model
  v13       + learned cost (unified envelope, no DR)
  v14       + learned cost (unified envelope + domain randomization)
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "target" / "sim-out" / "learned_cost"
TR = BASE / "traces"
OUT = BASE / "plots"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    ("cur_r20", "current tune (w_rate 20)", "#7f7f7f"),
    ("r10", "w_rate 10", "#9467bd"),
    ("r10_drag", "w_rate 10 + drag model", "#1f77b4"),
    ("v13", "+ learned cost v13", "#2ca02c"),
    ("v14", "+ learned cost v14 (DR)", "#d62728"),
]


def load(method, mission):
    p = TR / method / f"trace__{mission}.json"
    if not p.exists():
        return None
    h = json.load(open(p))["history"]
    h = [r for r in h if r.get("in_mission", True)]
    return {
        "t": np.array([r["t"] for r in h]),
        "pos": np.array([r["pos"] for r in h]),
        "ref": np.array([r["sp_pos"] for r in h]),
    }


def geom_rms(tr):
    """Whole-run RMS distance to the closest reference sample (the
    reference here is the densely sampled sp_pos series, so nearest-sample
    distance ≈ closest-point distance)."""
    ref = tr["ref"]
    d = np.empty(len(tr["pos"]))
    # chunked nearest-neighbour to keep memory bounded on the 70 s runs
    for i in range(0, len(tr["pos"]), 512):
        blk = tr["pos"][i:i + 512]
        d[i:i + len(blk)] = np.sqrt(
            ((blk[:, None, :] - ref[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.sqrt((d ** 2).mean()))


def plot_mission(mission):
    data = [(m, lab, c, load(m, mission)) for m, lab, c in METHODS]
    data = [(m, lab, c, tr) for m, lab, c, tr in data if tr is not None]
    if not data:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 8))
    ref = max((tr for _, _, _, tr in data), key=lambda tr: len(tr["ref"]))["ref"]
    ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1.2, label="reference")
    ax.plot(ref[0, 0], ref[0, 1], "ko", ms=6)
    for m, lab, c, tr in data:
        ax.plot(tr["pos"][:, 0], tr["pos"][:, 1], color=c, lw=1.0, alpha=0.9,
                label=f"{lab}  (geom RMS {geom_rms(tr):.3f} m)")
    ax.set_xlabel("x [m] (East)")
    ax.set_ylabel("y [m] (North)")
    ax.set_title(mission)
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    # altitude inset
    ins = ax.inset_axes([0.02, 0.02, 0.42, 0.20])
    ins.plot(data[0][3]["t"], ref[: len(data[0][3]["t"]), 2], "k--", lw=0.8)
    for m, lab, c, tr in data:
        ins.plot(tr["t"], tr["pos"][:, 2], color=c, lw=0.7)
    ins.set_ylabel("z [m]", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"xy_{mission}.png", dpi=130)
    plt.close(fig)
    return True


def main():
    missions = sorted({p.stem.replace("trace__", "")
                       for d in TR.iterdir() if d.is_dir()
                       for p in d.glob("trace__*.json")})
    done = [m for m in missions if plot_mission(m)]
    print(f"wrote {len(done)} xy_*.png to {OUT}")
    # overview grid
    n = len(done)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for k, mission in enumerate(done):
        ax = axs[k // cols][k % cols]
        data = [(m, lab, c, load(m, mission)) for m, lab, c in METHODS]
        data = [(m, lab, c, tr) for m, lab, c, tr in data if tr is not None]
        ref = max((tr for _, _, _, tr in data), key=lambda tr: len(tr["ref"]))["ref"]
        ax.plot(ref[:, 0], ref[:, 1], "k--", lw=0.9)
        for m, lab, c, tr in data:
            ax.plot(tr["pos"][:, 0], tr["pos"][:, 1], color=c, lw=0.7, alpha=0.9)
        ax.set_title(mission, fontsize=8)
        ax.axis("equal")
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
    for k in range(n, rows * cols):
        axs[k // cols][k % cols].axis("off")
    handles = [plt.Line2D([], [], color="k", ls="--", label="reference")] + [
        plt.Line2D([], [], color=c, label=lab) for _, lab, c in METHODS]
    fig.legend(handles=handles, loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "grid_overview.png", dpi=110)
    print("wrote grid_overview.png")


if __name__ == "__main__":
    main()
