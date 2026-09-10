#!/usr/bin/env python3
"""SQP-NMPC horizon scaling on the STM32H743 (Cortex-M7 @ 480 MHz).

(a) measured solve time per outer-loop tick vs horizon length N
(b) static RAM of the MPC working set vs N

The memory numbers are `size_of` of the actual monomorphised types
(FullQuadModel, NX=13, NU=4), read out on the host:

    SqpSolver<13,4,N,N+1>                    = 2664*N + 1508 B
    x_refs / u_refs / u_warm reference set   =   84*N +   52 B

Nothing is heap-allocated; the whole working set is static.

Writes analysis/mpc_horizon_scaling.png
"""

import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent / "mpc_horizon_scaling.png"

# Measured on-target solve time [us]
N = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
T_US = np.array([579, 915, 1390, 1831, 2220, 2702, 3065, 3570, 3936, 4302])

# Static RAM [bytes]
MEM_B = (2664 * N + 1508) + (84 * N + 52)
MEM_KB = MEM_B / 1024.0

C_T = "#2a6f97"
C_M = "#bc4b51"

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)

fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="0.9", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("$N$")
    ax.set_xticks([4, 8, 12, 16, 20])
    ax.set_xticks(N, minor=True)
    ax.set_xlim(0.5, 21.5)


# ── (a) compute time ──────────────────────────────────────────────────────
slope, icept = np.polyfit(N, T_US, 1)
ax_t.fill_between(N, 0, T_US, color=C_T, alpha=0.10, lw=0)
ax_t.plot(N, T_US, "-", color=C_T, lw=1.6)
ax_t.plot(N, T_US, "o", color=C_T, ms=4, mfc="white", mew=1.3)
ax_t.set_ylabel("solve time  [$\\mu$s]")
ax_t.set_ylim(0, 4800)
ax_t.set_yticks([0, 1000, 2000, 3000, 4000])
style(ax_t)

# ── (b) memory ────────────────────────────────────────────────────────────
ax_m.fill_between(N, 0, MEM_KB, color=C_M, alpha=0.10, lw=0)
ax_m.plot(N, MEM_KB, "-", color=C_M, lw=1.6)
ax_m.plot(N, MEM_KB, "o", color=C_M, ms=4, mfc="white", mew=1.3)
ax_m.set_ylabel("static RAM  [kB]")
ax_m.set_ylim(0, 62)
ax_m.set_yticks([0, 15, 30, 45, 60])
style(ax_m)

fig.savefig(OUT, dpi=300)
print(f"wrote {OUT}")
print(
    f"  {slope:.1f} us/stage + {icept:.0f} us;  {(MEM_KB[-1] - MEM_KB[0]) / 18:.2f} kB/stage"
)
for n, t, m in zip(N, T_US, MEM_KB):
    print(f"  N={n:2d}  {t:5d} us  {m:6.2f} kB")
