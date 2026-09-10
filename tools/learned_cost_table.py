#!/usr/bin/env python3
"""Summarize `target/sim-out/learned_cost/summary.json` (from
`tests/learned_cost_eval.rs`) as a nominal-vs-learned(-vs-blind) table,
and print the situation probes of `target/cost_policy/probe.json`.

    python3 tools/learned_cost_table.py [summary.json] [probe.json]
"""

import json
import sys

Z_NAMES = [
    "s.contour", "s.lag", "s.vel_x", "s.vel_y", "s.vel_z", "s.att_r", "s.att_p", "s.att_y",
    "t.contour", "t.lag", "t.vel_x", "t.vel_y", "t.vel_z", "t.att_r", "t.att_p", "t.att_y",
    "u.rate_x", "u.rate_y", "u.rate_z",
]


def table(path):
    rows = json.load(open(path))
    by = {}
    for r in rows:
        by.setdefault(r["mission"], {})[r["tag"]] = r
    tags = [t for t in ("nominal", "learned", "blind") if any(t in v for v in by.values())]
    print(f"{'mission':<24}" + "".join(f"{t + ' geom':>14}{t + ' time':>14}" for t in tags) + f"{'Δgeom':>9}{'Δtime':>9}")
    for m, v in by.items():
        line = f"{m:<24}"
        for t in tags:
            r = v.get(t)
            if r is None:
                line += f"{'-':>14}{'-':>14}"
            else:
                flag = "" if r["completed"] else "!"
                line += f"{r['rms_geom_m']:>13.4f}{flag:1}{r['rms_pos_err_m']:>13.4f} "
        if "nominal" in v and "learned" in v:
            n, l = v["nominal"], v["learned"]
            line += f"{100 * (l['rms_geom_m'] / n['rms_geom_m'] - 1):>+8.1f}%{100 * (l['rms_pos_err_m'] / n['rms_pos_err_m'] - 1):>+8.1f}%"
        print(line)
    print("('!' = did not complete)")


def probes(path):
    p = json.load(open(path))
    for name, rows in p.items():
        print(f"\nprobe: {name}")
        xs = [r["x"] for r in rows]
        print("  x     " + "".join(f"{x:>7.2f}" for x in xs))
        for i, zn in enumerate(Z_NAMES):
            zs = [r["z"][i] for r in rows]
            if max(zs) - min(zs) < 0.05:
                continue  # flat: not a function of this input
            print(f"  {zn:<8}" + "".join(f"{z:>7.2f}" for z in zs))


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "target/sim-out/learned_cost/summary.json"
    table(s)
    pr = sys.argv[2] if len(sys.argv) > 2 else "target/cost_policy/probe.json"
    try:
        probes(pr)
    except FileNotFoundError:
        pass
