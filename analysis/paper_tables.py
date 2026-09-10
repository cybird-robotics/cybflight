#!/usr/bin/env python3
"""Emit the paper's result tables as LaTeX, from `paper_metrics.py`.

Three tables, each answering one question a reviewer will ask:

  I   **tracking**   — how well does the stack fly the plan, across four
      trajectory shapes and three speed profiles? Contour RMSE is the
      headline; the peak speed / acceleration / duration columns are
      there so the RMSE can be read in context (a 25 cm error at
      12 m/s is not a 25 cm error at 5 m/s), and the lag column is
      there so nobody can object that the error was bought by flying
      slower.

  II  **ablation**   — figure-8 at both speed profiles, one row per
      single-flag change off the nominal controller. Contour *and* lag
      are both shown because the time-cost ablation trades one for the
      other; solve time and loop rate are shown because the
      two-iteration ablation is a real-time result, not a solver-quality
      one.

  III **payload**    — unknown added mass, four shapes at `mid` plus the
      figure-8 at `timeopt`. Effort and saturation carry this table:
      the tracking columns show the disturbance is rejected until the
      actuators run out of authority, and the effort columns show where
      that happens.

Preamble the tables need:

    \\usepackage{booktabs}
    \\usepackage{multirow}

No siunitx: units are written out, so the tables drop into any class.

Usage:
    python3 analysis/paper_tables.py                       # all three, to stdout
    python3 analysis/paper_tables.py --outdir paper/tables  # one .tex each
    python3 analysis/paper_tables.py --only ablation
    python3 analysis/paper_tables.py --metrics cached.json  # reuse a dump

Requires: pip install mcap cbor2 numpy
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_metrics as pm  # noqa: E402

SHAPE_LABEL = {"circle": "Circle", "figure8": "Figure-8",
               "slalom": "Slalom", "splits": "Splits"}
SPEED_LABEL = {"slow": "Slow", "mid": "Mid", "timeopt": "Time-opt."}
VARIANT_LABEL = {
    "nominal":  r"Proposed (contouring, RTI, nominal mass)",
    "timecost": r"\quad $\rightarrow$ time-sampled quadratic cost",
    "iter2":    r"\quad $\rightarrow$ 2 SQP iterations",
    "load":     r"\quad $\rightarrow$ unknown payload",
}

NA = r"\textemdash"


def f(x, digits=1):
    """Number or an em-dash when the log did not carry the field."""
    if x is None or (isinstance(x, float) and x != x):
        return NA
    return f"{x:.{digits}f}"


def get(res, shape, speed, variant="nominal"):
    return res.get(f"{shape}/{speed}/{variant}")


# ── Table I: tracking across shapes and speeds ───────────────────────────

def table_tracking(res):
    rows = []
    for shape in pm.SHAPES:
        first = True
        for speed in pm.SPEEDS:
            m = get(res, shape, speed)
            if m is None:
                continue
            name = (rf"\multirow{{3}}{{*}}{{{SHAPE_LABEL[shape]}}}" if first else "")
            first = False
            rows.append(
                f"{name} & {SPEED_LABEL[speed]} & "
                f"{f(m['ref_duration_s'], 2)} & {f(m['v_ref_max_m_s'])} & "
                f"{f(m['a_ref_max_m_s2'])} & "
                f"{f(m['contour_rmse_m'] * 100)} & {f(m['contour_p95_m'] * 100)} & "
                f"{f(m['contour_max_m'] * 100)} & "
                f"{f(m['lag_rms_s'] * 1000, 0)} & {f(m['tilt_max_deg'], 0)} \\\\"
            )
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows[-1] = r"\bottomrule"

    return r"""\begin{table}[t]
\centering
\caption{Trajectory-tracking accuracy over the mission-execution segment.
Contour error is the distance from the flown state to the nearest point on
the planned path; $\tau$ is the residual time lag against the plan after a
monotone alignment. Peak speed and acceleration are properties of the
\emph{plan}, given so the error can be read against the aggressiveness of
the manoeuvre. Single flight per cell.}
\label{tab:tracking}
\begin{tabular}{llrrrrrrrr}
\toprule
& & \multicolumn{3}{c}{Planned trajectory} & \multicolumn{3}{c}{Contour error [cm]} & & \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Path & Profile & $T$ [s] & $v_{\max}$ & $a_{\max}$ & RMS & p95 & max
     & $\tau_{\mathrm{rms}}$ [ms] & tilt$_{\max}$ [$^\circ$] \\
\midrule
""" + "\n".join(rows) + r"""
\end{tabular}
\end{table}"""


# ── Table II: controller ablation ────────────────────────────────────────

def table_ablation(res, shape="figure8"):
    rows = []
    for speed in ("mid", "timeopt"):
        base = get(res, shape, speed)
        if base is None:
            continue
        rows.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{SPEED_LABEL[speed]} profile"
                    rf" ($v_{{\max}}={base['v_ref_max_m_s']:.1f}$\,m/s)}}}} \\")
        for variant in ("nominal", "timecost", "iter2", "load"):
            m = get(res, shape, speed, variant)
            if m is None:
                continue
            rmse = m["contour_rmse_m"] * 100
            delta = NA if variant == "nominal" else \
                f"{100 * (rmse / (base['contour_rmse_m'] * 100) - 1):+.0f}\\%"
            rows.append(
                f"{VARIANT_LABEL[variant]} & "
                f"{f(rmse)} & {delta} & {f(m['contour_p95_m'] * 100)} & "
                f"{f(m['contour_max_m'] * 100)} & "
                f"{f(m['lag_rms_s'] * 1000, 0)} & "
                f"{f(m.get('solve_mean_ms'), 2)} & "
                f"{f(m.get('tick_rate_hz'), 0)} & "
                f"{f(m.get('sat_frac', float('nan')) * 100)} \\\\"
            )
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows[-1] = r"\bottomrule"

    return r"""\begin{table}[t]
\centering
\caption{Controller ablation on the figure-8, one single-flag change per row
off the nominal configuration ($N=20$, $\Delta t$ fixed, one real-time-iteration
SQP step per tick, contouring position cost, position-based reference sampler).
The planned trajectory is identical in every row; only the controller changes.
Note that the time-sampled quadratic cost does not simply track worse --- it
buys a lower time lag $\tau$ with a larger contour error, which is the
trade-off the contouring formulation is designed to make. The two-iteration
row is a real-time result: the solve no longer fits the 10\,ms control
period, and the achieved loop rate roughly halves. Single flight per row.
$^{\dagger}$Median cadence of the logged solver telemetry; the on-board
recorder drops messages under load, so this is a lower bound on the true
rate and is only compared across rows, never against the 100\,Hz
set point. Solver telemetry was not recorded in the nominal mid flight.}
\label{tab:ablation}
\begin{tabular}{lrrrrrrrr}
\toprule
& \multicolumn{4}{c}{Contour error [cm]} & & \multicolumn{2}{c}{Solver} & \\
\cmidrule(lr){2-5}\cmidrule(lr){7-8}
Configuration & RMS & $\Delta$ & p95 & max
 & $\tau_{\mathrm{rms}}$ [ms] & $t_{\mathrm{solve}}$ [ms] & rate$^{\dagger}$ [Hz]
 & sat.\ [\%] \\
\midrule
""" + "\n".join(rows) + r"""
\end{tabular}
\end{table}"""


# ── Table III: payload rejection ─────────────────────────────────────────

def table_payload(res):
    cells = [(s, "mid") for s in pm.SHAPES] + [("figure8", "timeopt")]
    rows = []
    for shape, speed in cells:
        n, l = get(res, shape, speed), get(res, shape, speed, "load")
        if n is None or l is None:
            continue
        rows.append(
            f"{SHAPE_LABEL[shape]} & {SPEED_LABEL[speed]} & "
            f"{f(n['contour_rmse_m'] * 100)} & {f(l['contour_rmse_m'] * 100)} & "
            f"{100 * (l['contour_rmse_m'] / n['contour_rmse_m'] - 1):+.0f}\\% & "
            f"{f(n.get('u_mean'), 3)} & {f(l.get('u_mean'), 3)} & "
            f"{f(n.get('u_trim_asym'), 3)} & {f(l.get('u_trim_asym'), 3)} & "
            f"{f(n.get('sat_frac', float('nan')) * 100)} & "
            f"{f(l.get('sat_frac', float('nan')) * 100)} \\\\"
        )

    return r"""\begin{table}[t]
\centering
\caption{Rejection of an unknown added mass. The controller is given no
knowledge of the payload: the model, the weights and the plan are unchanged
from the nominal flight. \textbf{TODO: state the payload mass and its
fraction of the 0.6\,kg vehicle mass, and where it was mounted.}
Tracking is essentially unchanged at the mid profile --- the disturbance is
absorbed into control effort, visible in $\bar{u}$ --- and degrades only at
the time-optimal profile, where the extra thrust demand pushes the vehicle
into actuator saturation for most of the mission. ``Trim spread'' is the
spread of the four per-motor mean commands: it roughly doubles under the
payload in every case, which says the added mass sits off the centre of
gravity, so the disturbance is a moment as well as a force. Single flight per
cell; differences below the run-to-run spread should not be read as
improvements, and the negative $\Delta$ entries are noise, not gains.}
\label{tab:payload}
\begin{tabular}{llrrrrrrrrr}
\toprule
& & \multicolumn{3}{c}{Contour RMS [cm]} & \multicolumn{2}{c}{Mean cmd.\ $\bar{u}$}
  & \multicolumn{2}{c}{Trim spread} & \multicolumn{2}{c}{Saturated [\%]} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
Path & Profile & nom. & payload & $\Delta$ & nom. & payload & nom. & payload
 & nom. & payload \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""


TABLES = {"tracking": table_tracking, "ablation": table_ablation,
          "payload": table_payload}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", default=pm.DATASETS)
    ap.add_argument("--metrics", metavar="JSON",
                    help="reuse a paper_metrics.py --dump instead of recomputing")
    ap.add_argument("--outdir", metavar="DIR",
                    help="write tab_<name>.tex here instead of stdout")
    ap.add_argument("--only", choices=sorted(TABLES), action="append",
                    help="emit only this table (repeatable)")
    args = ap.parse_args()

    if args.metrics:
        with open(args.metrics) as fh:
            res = json.load(fh)
    else:
        res = pm.compute_all(args.datasets, bootstrap=False, verbose=False)

    wanted = args.only or list(TABLES)
    for name in wanted:
        tex = TABLES[name](res)
        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            path = os.path.join(args.outdir, f"tab_{name}.tex")
            with open(path, "w") as fh:
                fh.write(tex + "\n")
            print(f"wrote {path}")
        else:
            print(tex, end="\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
