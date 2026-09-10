#!/usr/bin/env python3
"""Render the TinyMPC vs SQP-NMPC indoor-mission comparison.

Reads target/sim-out/figure8_tinympc/{summary.json,trace__*.json}
(written by crates/cybflight_sim/tests/figure8_tinympc_compare.rs) and
writes, next to them:

  <family>_paths.png    - XY path per mission of that family (circle,
                          figure8, slalom, splits): reference + both stacks
  <family>_errors.png   - position-error norm vs time per mission
  xy_<mission>.png      - standalone XY plot per mission (reference + both)
  summary.md            - markdown table of the summary metrics, plus the
                          geometric (closest-point-on-path) RMS / peak error
                          computed here from the traces
"""
import json
import math
import pathlib
import sys
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "target" / "sim-out" / "figure8_tinympc"
CONTROLLERS = [
    ("mpc_indi", "SQP-NMPC + INDI", "#1f77b4"),
    ("tinympc_indi", "TinyMPC + INDI", "#d62728"),
    ("tinympc_ff_indi", "TinyMPC + ref. feedforward + INDI", "#2ca02c"),
    ("geometric_indi", "Geometric tracking + INDI", "#9467bd"),
]
SPEED_ORDER = ["slow", "mid", "fast", "timeopt"]


def load_trace(mission, ctrl):
    p = OUT / f"trace__{mission}__{ctrl}.json"
    if not p.exists():
        return None
    return [r for r in json.load(open(p))["history"] if r["in_mission"]]


def geometric_error(h):
    """Per-sample distance from the flown position to the closest point on
    the reference *path* (polyline through the 100 Hz reference samples),
    independent of timing. Returns (rms, peak) in metres."""
    ref = np.array([r["sp_pos"] for r in h])
    pos = np.array([r["pos"] for r in h])
    a, b = ref[:-1], ref[1:]
    ab = b - a
    ab2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
    d = np.empty(len(pos))
    for i, p in enumerate(pos):
        t = np.clip(((p - a) * ab).sum(axis=1) / ab2, 0.0, 1.0)
        q = a + t[:, None] * ab
        d[i] = np.sqrt(((q - p) ** 2).sum(axis=1).min())
    return float(np.sqrt((d * d).mean())), float(d.max())


def families(summary):
    fams = OrderedDict()
    for r in summary:
        fam = r["mission"].split("_")[1]
        fams.setdefault(fam, set()).add(r["mission"])
    return {
        f: sorted(ms, key=lambda m: SPEED_ORDER.index(m.rsplit("_", 1)[1]))
        for f, ms in fams.items()
    }


def plot_family(fam, missions):
    n = len(missions)
    fig_p, ax_p = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    fig_e, ax_e = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
    for i, mission in enumerate(missions):
        ap, ae = ax_p[0][i], ax_e[i][0]
        ref_drawn = False
        for ctrl, label, color in CONTROLLERS:
            h = load_trace(mission, ctrl)
            if not h:
                continue
            if not ref_drawn:
                ap.plot([r["sp_pos"][0] for r in h], [r["sp_pos"][1] for r in h], "k--", lw=1, label="reference")
                ref_drawn = True
            ap.plot([r["pos"][0] for r in h], [r["pos"][1] for r in h], color=color, lw=1.2, label=label)
            err = [math.sqrt(sum((r["pos"][k] - r["sp_pos"][k]) ** 2 for k in range(3))) for r in h]
            ae.plot([r["t"] for r in h], err, color=color, lw=1.2, label=label)
        title = mission.replace("indoor_", "").replace("_", " ")
        ap.set_title(title)
        ap.set_xlabel("x [m]")
        ap.set_ylabel("y [m]")
        ap.set_aspect("equal")
        ap.set_xlim(-4.5, 4.5)
        ap.set_ylim(-4.5, 4.5)
        ap.grid(alpha=0.3)
        ap.legend(fontsize=8)
        ae.set_title(title)
        ae.set_ylabel("‖p − p_ref‖ [m]")
        ae.set_ylim(0, 1.0)
        ae.grid(alpha=0.3)
        ae.legend(fontsize=8)
    ax_e[-1][0].set_xlabel("t [s]")
    fig_p.tight_layout()
    fig_e.tight_layout()
    fig_p.savefig(OUT / f"{fam}_paths.png", dpi=130)
    fig_e.savefig(OUT / f"{fam}_errors.png", dpi=130)
    plt.close(fig_p)
    plt.close(fig_e)


def plot_mission_xy(mission):
    """Standalone XY plot: reference path + each controller's executed path."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ref_drawn = False
    for ctrl, label, color in CONTROLLERS:
        h = load_trace(mission, ctrl)
        if not h:
            continue
        if not ref_drawn:
            ax.plot([r["sp_pos"][0] for r in h], [r["sp_pos"][1] for r in h], "k--", lw=1.2, label="reference")
            ax.plot(h[0]["sp_pos"][0], h[0]["sp_pos"][1], "ko", ms=5, label="start")
            ref_drawn = True
        ax.plot([r["pos"][0] for r in h], [r["pos"][1] for r in h], color=color, lw=1.4, label=label)
    ax.set_title(mission.replace("indoor_", "").replace("_", " "))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"xy_{mission}.png", dpi=130)
    plt.close(fig)


def main():
    summary = json.load(open(OUT / "summary.json"))
    fams = families(summary)
    for fam, missions in fams.items():
        plot_family(fam, missions)
        for m in missions:
            plot_mission_xy(m)

    lines = [
        "| mission | controller | completed | temporal RMS [m] | temporal peak [m] | geometric RMS [m] | geometric peak [m] | terminal err [m] | peak tilt [°] | peak motor sat [%] | early exit |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = [m for ms in fams.values() for m in ms]
    for mission in order:
        for r in summary:
            if r["mission"] != mission:
                continue
            h = load_trace(r["mission"], r["controller"])
            g_rms, g_peak = geometric_error(h) if h else (float("nan"), float("nan"))
            r["geom_rms_err_m"], r["geom_peak_err_m"] = g_rms, g_peak
            lines.append(
                f"| {r['mission']} | {r['controller']} | {'yes' if r['completed'] else 'no'} | "
                f"{r['rms_pos_err_m']:.3f} | {r['peak_pos_err_m']:.3f} | {g_rms:.3f} | {g_peak:.3f} | "
                f"{r['terminal_pos_err_m']:.3f} | {r['peak_tilt_deg']:.1f} | "
                f"{r['peak_motor_saturation_pct']:.0f} | {r['early_exit'] or '—'} |"
            )
    (OUT / "summary_with_geometric.json").write_text(json.dumps(summary, indent=2))
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {', '.join(f'{f}_paths.png / {f}_errors.png' for f in fams)} and summary.md in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
