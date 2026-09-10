#!/usr/bin/env python3
"""Compare flights carrying an unknown payload against the nominal flights.

The controller is given no knowledge of the added mass: model, weights
and planned trajectory are identical to the nominal run, so every
difference between the two flights is the disturbance being rejected (or
not). This script puts the two side by side for one speed profile.

Pick the profile with ``--speed``:

  ``mid``      four shapes (circle, figure-8, slalom, splits) — the
               operating point where the payload is expected to be
               absorbed into control effort with tracking essentially
               unchanged.
  ``timeopt``  figure-8 only — the operating point where the extra
               thrust demand runs the allocator out of authority, so the
               payload *does* show up in the tracking error.

Every run is cropped to the mission segment exactly like
``plot_mcap.py --from-mission --until-idle`` (first mission-switch rising
edge → first ``MISSION_IDLE`` event) and the metrics come from
``paper_metrics.py``, so the numbers here match the paper's tables.

Four figures, one column per shape:

  1. **xy**       top-down executed trajectories, both flights over the
                  planned path. The "does it still fly the shape" plot.
  2. **error**    contour error along the path, nominal vs payload, with
                  actuator-saturation intervals shaded. Shows *where* the
                  payload costs accuracy — in the corners, if anywhere.
  3. **effort**   the control-performance panel: per-motor command
                  envelope (min–max band + mean), speed against the plan,
                  and tilt angle. This is where the payload is visible at
                  the mid profile: the trace moves up, not sideways.
  4. **summary**  bar chart of the headline metrics, nominal vs payload,
                  over all shapes at once.

and one figure that spans *both* profiles instead of one:

  5. **xypair**   1×2, one shape (``--shape``, default figure-8) flown
                  with and without the payload at the mid profile (left)
                  and the time-optimal profile (right). Shared axes, no
                  titles, a single legend — the paper figure.
  6. **xysolo**   the same two panels as **xypair**, but written out as
                  two standalone figures — same camera, colours and
                  (pooled) limits, each with its own axis labels and
                  legend, larger text, and no z exaggeration. For a
                  layout that places the two profiles separately.

``--3d`` draws the two trajectory figures (**xy**, **xypair**) in 3D
instead of top-down, with ``--view ELEV,AZIM`` to override the camera
each figure picks for itself (**xysolo** uses an isometric one, which is
the only camera that draws a metre of x, of y and of z at the same length
on the page). Worth it
mainly at the time-optimal profile, where the plan itself descends
(z from 1.14 m down to 0.26 m on the figure-8) and a top-down view hides
that entirely; the mid-profile plans are flat at z = 1 m, so the third
axis there shows only the tracking error out of plane.

A nominal-vs-payload delta table is printed alongside.

Usage:
    python3 analysis/plot_payload_compare.py --speed mid
    python3 analysis/plot_payload_compare.py --speed timeopt --show
    python3 analysis/plot_payload_compare.py --speed mid --outdir analysis/paper/figs
    python3 analysis/plot_payload_compare.py --speed mid --only xy --only error
    python3 analysis/plot_payload_compare.py --only xypair --outdir analysis/paper/figs
    python3 analysis/plot_payload_compare.py --only xypair --3d --view 26,-52 --show
    python3 analysis/plot_payload_compare.py --only xysolo --3d --outdir analysis/paper/figs

Requires: pip install mcap cbor2 numpy matplotlib
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_figures as pf  # noqa: E402
import paper_metrics as pm  # noqa: E402

VARIANTS = ("nominal", "load")
LABEL = {"nominal": "nominal", "load": "with payload"}
COLOR = {"nominal": pf.VARIANT_COLOR["nominal"], "load": pf.VARIANT_COLOR["load"]}


# ── run discovery & loading ──────────────────────────────────────────────


def find_pairs(speed, datasets=pm.DATASETS):
    """[(shape, {variant: mcap}, csv)] for every shape flown both ways."""
    out = []
    for shape in pm.SHAPES:
        paths = {
            "nominal": os.path.join(
                datasets, f"indoor_exp_{speed}", f"exp_{shape}_{speed}.mcap"
            ),
            "load": os.path.join(
                datasets, "indoor_exp_ablation", f"exp_{shape}_{speed}_load.mcap"
            ),
        }
        csv = pm.reference_for(shape, speed)
        if all(os.path.exists(p) for p in paths.values()) and os.path.exists(csv):
            out.append((shape, paths, csv))
    return out


def load_all(speed, datasets=pm.DATASETS, verbose=True):
    """{shape: {variant: metric-dict-with-`_series`}} for one speed profile."""
    runs = {}
    for shape, paths, csv in find_pairs(speed, datasets):
        runs[shape] = {}
        for variant, mcap in paths.items():
            runs[shape][variant] = pm.analyse(mcap, csv, bootstrap=False, series=True)
            if verbose:
                m = runs[shape][variant]
                print(
                    f"ok  {shape}/{speed}/{variant:<8} "
                    f"contour RMS {m['contour_rmse_m'] * 100:5.1f} cm"
                )
    return runs


# ── shared plotting helpers ──────────────────────────────────────────────


def spans(x, mask):
    """[(x0, x1)] for each contiguous run of True in `mask`."""
    if mask is None or not mask.any():
        return []
    edges = np.flatnonzero(np.diff(mask.astype(int)))
    starts = np.r_[0 if mask[0] else [], edges[mask[edges + 1]] + 1].astype(int)
    ends = np.r_[edges[mask[edges]] + 1, len(mask) - 1 if mask[-1] else []].astype(int)
    return [(x[a], x[min(b, len(x) - 1)]) for a, b in zip(starts, ends)]


def merge(intervals, gap):
    """Join intervals closer than `gap`.

    Saturation chatters — the allocator clips for a few samples, backs
    off, clips again — so the raw intervals draw as a picket fence that
    reads as noise. What matters is the *region* of the manoeuvre in
    which the vehicle was out of authority, so intervals separated by
    less than `gap` are drawn as one.
    """
    out = []
    for a, b in intervals:
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def shade_saturation(ax, m, xkey="t", gap_frac=0.02):
    """Grey out the intervals where any motor was at/above saturation.

    `xkey` picks the abscissa: "t" for a time axis, "s" to place the
    intervals on the along-path progress axis of the error figure.
    """
    s = m["_series"]
    if s["t_u"] is None or s["sat"] is None:
        return False
    x = s["t_u"]
    if xkey == "s":
        x = np.interp(s["t_u"], s["t"], s["s_exec"])
    gap = gap_frac * (x[-1] - x[0]) if len(x) > 1 else 0.0
    for a, b in merge(spans(x, s["sat"]), gap):
        ax.axvspan(a, b, color="0.85", lw=0, zorder=0)
    return bool(s["sat"].any())


def smooth(y, frac=0.004):
    n = max(1, int(len(y) * frac))
    if n < 2:
        return y
    k = np.ones(n) / n
    return np.convolve(
        np.pad(y, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid"
    )


def draw_track(ax, runs, three_d=False):
    """Plan + both flights on one axes, top-down or in 3D.

    Returns the stacked point cloud so the caller can set limits that
    are common to several panels.
    """
    ref = runs["nominal"]["_series"]["ref_pos"]
    tracks = [("reference", ref, dict(ls="--", color="0.45"))]
    tracks += [
        (LABEL[v], runs[v]["_series"]["pos"], dict(color=COLOR[v])) for v in VARIANTS
    ]
    for label, p, kw in tracks:
        cols = (p[:, 0], p[:, 1], p[:, 2]) if three_d else (p[:, 0], p[:, 1])
        ax.plot(*cols, lw=2.0, label=label, **kw)
    if not three_d:
        ax.set_aspect("equal", adjustable="box")
    return np.vstack([p for _, p, _ in tracks])


def hide_z(ax):
    """Drop the z axis furniture from a panel that shares another's scale.

    With the default camera the z axis is drawn on a panel's right-hand
    side, i.e. straight into its neighbour, so only the rightmost panel
    of a shared-scale row keeps its ticks.
    """
    ax.set_zticklabels([])
    ax.set_zlabel("")


def limits_3d(ax, pts, z_floor=0.3, pad=1.04, zoom=1.0, zlim=None):
    """Common limits and box aspect for a 3D track panel.

    True equal aspect is useless here: these missions span metres
    horizontally and a fraction of a metre vertically, so an honest
    z-axis collapses to a line. The vertical axis is therefore given at
    least `z_floor` of the larger horizontal extent, and the
    exaggeration factor is returned so it can be written onto the axis
    label rather than left for the reader to discover.

    `zlim` pins the vertical axis to a chosen range instead of fitting
    it to the flights — a round range the reader can hold in their head,
    and one that shows where the flight sits between the floor and the
    net. The box follows it, so the proportions stay honest.
    """
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-6)
    ctr, r = (hi + lo) / 2, span / 2 * pad
    ax.set_xlim(ctr[0] - r[0], ctr[0] + r[0])
    ax.set_ylim(ctr[1] - r[1], ctr[1] + r[1])
    if zlim is None:
        ax.set_zlim(ctr[2] - r[2], ctr[2] + r[2])
    else:
        ax.set_zlim(*zlim)
        # x and y draw `span * pad` of range over `span` of box; z has
        # to divide out the same padding or it is quietly exaggerated.
        span[2] = (zlim[1] - zlim[0]) / pad
    box, exag = span.astype(float).copy(), 1.0
    z_min = z_floor * max(span[0], span[1])
    if box[2] < z_min:
        exag = z_min / box[2]
        box[2] = z_min
    # `zoom` fills the empty margin a 3D axes reserves around its cube.
    ax.set_box_aspect(tuple(box), zoom=zoom)
    # Orthographic. matplotlib's default 3D camera is a perspective one
    # with `focal_length=1` — a ~90 deg wide-angle lens — so the scale
    # varies with depth: a metre at the near corner of this box draws
    # 1.29x longer than a metre of the same axis at the far corner. The
    # z axis is drawn on the near edge and stands nearly upright, so it
    # is the one that gains, and the plot reads as though z were on a
    # bigger scale than x and y. Ortho gives one scale everywhere in the
    # box; what remains between the axes is the camera's foreshortening,
    # which no projection can remove.
    ax.set_proj_type("ortho")
    return exag


def label_axes(ax, three_d, exag=1.0, ylabel=True, size=None, pad=None):
    """Axis labels, optionally at an explicit size and standoff.

    `pad` is either one value or (x, y, z): the z label of a 3D panel is
    written along the outside edge of the figure, so it needs a shorter
    standoff than the x/y labels or it is pushed off the canvas.
    """
    kw = {} if size is None else {"fontsize": size}
    pads = (None, None, None) if pad is None else np.broadcast_to(pad, 3)
    ax.set_xlabel("x [m]", labelpad=pads[0], **kw)
    if ylabel or three_d:
        ax.set_ylabel("y [m]", labelpad=pads[1], **kw)
    if three_d:
        z = "z [m]" + (f"  ($\\times${exag:.1f})" if exag > 1.05 else "")
        ax.set_zlabel(z, labelpad=pads[2], **kw)


def scale_text(ax, scale):
    """Enlarge the tick labels of an existing axes by `scale`.

    The axis *labels* are sized when they are created, but a 3D axes
    builds its ticks lazily at draw time — after any `rc_context` has
    closed — so the tick size has to be pinned on the axes itself rather
    than left to the rcParams.
    """
    import matplotlib

    ax.tick_params(labelsize=matplotlib.rcParams["xtick.labelsize"] * scale)


def tick_budget(ax, three_d, nbins=(3, 3, 2)):
    """Cap the ticks per axis so enlarged labels cannot collide.

    The default locator picks a tick count for the axis' *data* range,
    with no idea how much room the labels need; at 1.5x text on a
    projected 3D axis that is reliably more labels than fit, and they
    overprint each other along the foreshortened edges. The z axis gets
    the smallest budget of the three: it is the shortest edge on screen
    once the box is no longer exaggerated.
    """
    from matplotlib.ticker import MaxNLocator

    axes = (ax.xaxis, ax.yaxis, ax.zaxis) if three_d else (ax.xaxis, ax.yaxis)
    for a, n in zip(axes, nbins):
        a.set_major_locator(MaxNLocator(nbins=n))


def ink_bounds(fig):
    """(x0, x1, y0, y1, w, h) in pixels of every pixel the figure draws on.

    Rows count from the top of the canvas, as the buffer does. Returns
    None if the canvas cannot be read back (a non-Agg backend).
    """
    fig.canvas.draw()
    try:
        buf = np.asarray(fig.canvas.buffer_rgba())
    except AttributeError:
        return None
    rows, cols = np.nonzero((buf[..., :3] < 250).any(-1) & (buf[..., 3] > 0))
    if not len(rows):
        return None
    h, w = buf.shape[:2]
    return cols.min(), cols.max(), rows.min(), rows.max(), w, h


def fit_panel(fig, ax, step=0.03, iters=10, pad_in=0.02):
    """Shrink the panel until every axis label is inside the canvas.

    A 3D axes puts its labels wherever the projection lands them, which
    at some cameras is clean off the canvas — never rendered, and so
    beyond the reach of a crop, which can only keep what was drawn.
    Reading the render back is no help either: ink that fell outside
    leaves no trace of itself. The labels' own extents are what to steer
    by — their *tick* labels are not, reporting positions that follow
    the canvas rather than the cube. Pulling the panel in shrinks the
    cube, and its labels come in with it.
    """
    p = fig.subplotpars
    box = dict(left=p.left, bottom=p.bottom, right=p.right, top=p.top)
    for _ in range(iters):
        fig.subplots_adjust(**box)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        w, h = fig.canvas.get_width_height()
        pad = pad_in * fig.dpi
        bbs = [a.label.get_window_extent(r) for a in (ax.xaxis, ax.yaxis, ax.zaxis)]
        out = (
            min(b.x0 for b in bbs) < pad,
            max(b.x1 for b in bbs) > w - pad,
            min(b.y0 for b in bbs) < pad,
            max(b.y1 for b in bbs) > h - pad,
        )
        if not any(out):
            return
        box["left"] += step * out[0]
        box["right"] -= step * out[1]
        box["bottom"] += step * out[2]
        box["top"] -= step * out[3]


def ink_bbox(fig, pad=0.02):
    """Bbox in inches of every pixel the figure actually draws on.

    `bbox_inches="tight"` never crops tighter than an axes' rectangle,
    and a 3D axes' rectangle is square by construction while its cube —
    flat, seen from a shallow angle — fills only part of it. The result
    is a band of empty canvas that no margin setting can reach. Reading
    the extent off the render instead crops to the drawing itself, with
    the labels and the legend included for free.

    Returns None if the canvas cannot be read back (a non-Agg backend),
    which leaves the caller to save the figure whole.
    """
    from matplotlib.transforms import Bbox

    b = ink_bounds(fig)
    if b is None:
        return None
    x0, x1, y0, y1, _w, h = b
    dpi = fig.dpi
    return Bbox(
        [
            [x0 / dpi - pad, (h - 1 - y1) / dpi - pad],
            [(x1 + 1) / dpi + pad, (h - y0) / dpi + pad],
        ]
    )


def grid(plt, rows, shapes, width, height_per_row, sharex=False):
    fig, axes = plt.subplots(
        rows,
        len(shapes),
        squeeze=False,
        sharex=sharex,
        figsize=(width * len(shapes), height_per_row * rows),
    )
    return fig, axes


# ── figures ──────────────────────────────────────────────────────────────


def fig_xy(plt, runs, speed, width, three_d=False, view=None):
    """Executed trajectories over the planned path, both flights."""
    view = view or DEFAULT_VIEW
    shapes = list(runs)
    kw = {"subplot_kw": {"projection": "3d"}} if three_d else {}
    fig, axes = plt.subplots(
        1, len(shapes), squeeze=False, figsize=(width * len(shapes), width * 0.95), **kw
    )
    for c, (ax, shape) in enumerate(zip(axes[0], shapes)):
        pts = draw_track(ax, runs[shape], three_d)
        exag = limits_3d(ax, pts) if three_d else 1.0
        if three_d:
            ax.view_init(*view)
        n, l = (runs[shape][v]["contour_rmse_m"] * 100 for v in VARIANTS)
        ax.set_title(f"{pf.SHAPE_LABEL[shape]}\ncontour RMS {n:.1f} → {l:.1f} cm")
        label_axes(ax, three_d, exag, ylabel=(c == 0))
        if three_d and c < len(shapes) - 1:
            hide_z(ax)
    axes[0][0].legend(loc="best", framealpha=0.9)
    fig.suptitle(f"Payload rejection — {pf.SPEED_LABEL[speed].lower()} profile", y=1.0)
    fig.tight_layout()
    return fig


def fig_error(plt, runs, speed, width):
    """Contour error along the path, with saturated intervals shaded.

    Plotted against distance travelled rather than time so the two
    flights line up geometrically even where the payload costs the
    vehicle a little progress.
    """
    from matplotlib.patches import Patch

    shapes = list(runs)
    fig, axes = grid(plt, 1, shapes, width, width * 0.72)
    saturated = False
    for ax, shape in zip(axes[0], shapes):
        for v in VARIANTS:
            s = runs[shape][v]["_series"]
            saturated |= shade_saturation(ax, runs[shape][v], xkey="s")
            ax.plot(
                s["s_exec"],
                smooth(s["e_c"]) * 100,
                lw=0.9,
                color=COLOR[v],
                label=LABEL[v],
            )
        ax.set_title(pf.SHAPE_LABEL[shape])
        ax.set_xlabel("distance along path [m]")
        ax.set_xlim(0, runs[shape]["nominal"]["_series"]["s_ref_total"])
    axes[0][0].set_ylabel("contour error [cm]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if saturated:
        handles.append(Patch(facecolor="0.82"))
        labels.append(r"motor $\geq$ 98 %")
    axes[0][0].legend(handles, labels, loc="upper left", framealpha=0.9)
    fig.suptitle(
        f"Contour error along the path — {pf.SPEED_LABEL[speed].lower()} profile", y=1.0
    )
    fig.tight_layout()
    return fig


def fig_effort(plt, runs, speed, width):
    """Control performance: motor envelope, speed vs plan, tilt angle.

    The motor row is the point of the figure. An added mass the
    controller does not model is rejected by the allocator holding more
    thrust, so at the mid profile the band moves *up* while the tracking
    rows stay put — until the band reaches 1.0, at which point there is
    nothing left to reject with and the error rows move instead.
    """
    shapes = list(runs)
    fig, axes = grid(plt, 3, shapes, width, width * 0.52, sharex="col")
    for c, shape in enumerate(shapes):
        for v in VARIANTS:
            s = runs[shape][v]["_series"]
            # row 0 — per-motor command envelope + mean
            ax = axes[0][c]
            if s["u"] is not None:
                ax.fill_between(
                    s["t_u"],
                    s["u"].min(1),
                    s["u"].max(1),
                    color=COLOR[v],
                    alpha=0.20,
                    lw=0,
                )
                ax.plot(
                    s["t_u"], s["u"].mean(1), lw=0.9, color=COLOR[v], label=LABEL[v]
                )
            # row 1 — speed against the plan
            axes[1][c].plot(s["t"], s["speed"], lw=0.8, color=COLOR[v], label=LABEL[v])
            # row 2 — tilt angle
            axes[2][c].plot(s["t"], s["tilt"], lw=0.8, color=COLOR[v], label=LABEL[v])
        axes[0][c].axhline(pm.SAT_THRESHOLD, color="0.4", ls=":", lw=0.8)
        axes[0][c].set_ylim(0, 1.05)
        axes[0][c].set_title(pf.SHAPE_LABEL[shape])
        axes[2][c].set_xlabel("t since mission trigger [s]")
    axes[0][0].set_ylabel("motor command [-]")
    axes[1][0].set_ylabel(r"$\|v\|$ [m/s]")
    axes[2][0].set_ylabel("tilt [deg]")
    fig.suptitle(
        f"Control effort and response — {pf.SPEED_LABEL[speed].lower()} profile", y=1.0
    )
    # figure-level legend: the motor band fills the top-left of its axes,
    # so an in-axes legend covers the very trace it labels.
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    return fig


# (key, plotted label, plain-text label, scale, digits, delta)
# `scale` takes the metric to the plotted unit. `delta` is how the
# nominal→payload change is quoted in the printed table: "rel" as a
# percentage, "abs" in the plotted unit — the right choice for the
# saturated fraction, where the nominal value is 0 or near it and a
# relative change is meaningless.
BARS = [
    ("contour_rmse_m", "contour RMS [cm]", "contour RMS [cm]", 100, 1, "rel"),
    ("contour_p95_m", "contour p95 [cm]", "contour p95 [cm]", 100, 1, "rel"),
    (
        "lag_rms_s",
        r"lag $\tau_{\mathrm{rms}}$ [ms]",
        "lag tau_rms [ms]",
        1000,
        0,
        "rel",
    ),
    ("u_mean", r"mean motor cmd $\bar{u}$ [-]", "mean motor cmd [-]", 1, 3, "rel"),
    ("u_trim_asym", "trim spread [-]", "trim spread [-]", 1, 3, "rel"),
    ("sat_frac", "saturated [% of mission]", "saturated [% of mission]", 100, 1, "abs"),
]


def fig_summary(plt, runs, speed, width):
    """The headline metrics side by side, nominal vs payload."""
    shapes = list(runs)
    ncol = 3
    nrow = int(np.ceil(len(BARS) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, squeeze=False, figsize=(width * ncol, width * 0.62 * nrow)
    )
    x = np.arange(len(shapes))
    for ax, (key, label, _plain, scale, digits, _d) in zip(axes.ravel(), BARS):
        for i, v in enumerate(VARIANTS):
            vals = [runs[s][v].get(key, np.nan) * scale for s in shapes]
            bars = ax.bar(
                x + (i - 0.5) * 0.36, vals, 0.34, color=COLOR[v], label=LABEL[v]
            )
            ax.bar_label(bars, fmt=f"%.{digits}f", fontsize=6, padding=1)
        ax.set_xticks(x, [pf.SHAPE_LABEL[s] for s in shapes], rotation=0)
        ax.set_title(label)
        ax.margins(y=0.18)
        ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[len(BARS) :]:
        ax.set_visible(False)
    fig.suptitle(f"Nominal vs payload — {pf.SPEED_LABEL[speed].lower()} profile", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.975),
    )
    return fig


def fig_xypair(plt, runs_by_speed, shape, width, three_d=False, view=None):
    """One shape, both speed profiles: mid on the left, time-optimal right.

    The paper figure. Deliberately bare — no titles, one legend, panels
    sharing both axes so the two profiles are read at the same scale and
    the reader compares the *tracks*, not the axis limits. Everything
    else (which profile, which shape, what the error was) belongs in the
    caption and the table, not in decoration on the plot.
    """
    view = view or DEFAULT_VIEW
    # the tracks are taller than they are wide, so a panel box near the
    # data's own aspect ratio is what keeps the figure compact — a square
    # panel would be mostly margin.
    if three_d:
        fig, axes = plt.subplots(
            1, 2, figsize=(width * 2, width * 1.0), subplot_kw={"projection": "3d"}
        )
        pts = [
            draw_track(ax, runs_by_speed[sp][shape], True)
            for ax, sp in zip(axes, PAIR_SPEEDS)
        ]
        # one point cloud for both panels: same limits, same box, so the
        # two profiles are read at the same scale.
        common = np.vstack(pts)
        for ax in axes:
            exag = limits_3d(ax, common)
            ax.view_init(*view)
            label_axes(ax, True, exag)
        hide_z(axes[0])
        axes[1].legend(
            loc="lower right",
            framealpha=0.9,
            borderpad=0.3,
            handlelength=1.4,
            labelspacing=0.3,
            bbox_to_anchor=(1.0, 1.0),
        )
        # `tight_layout` mismeasures 3D axis labels (they are drawn in the
        # projection, not as axes-level artists), so the margins are set
        # by hand — otherwise the x/y labels are clipped and the left
        # panel's y label lands inside the right panel.
        fig.subplots_adjust(left=0.01, right=0.90, bottom=0.09, top=1.0, wspace=0.10)
        return fig
    else:
        fig, axes = plt.subplots(
            1, 2, figsize=(width * 2, width * 1.45), sharex=True, sharey=True
        )
        for ax, speed in zip(axes, PAIR_SPEEDS):
            draw_track(ax, runs_by_speed[speed][shape], False)
            label_axes(ax, False, ylabel=(ax is axes[0]))
        legend_ax = axes[1]
    legend_ax.legend(
        loc="lower right",
        framealpha=0.9,
        borderpad=0.3,
        handlelength=1.4,
        labelspacing=0.3,
        bbox_to_anchor=(1.0, 1.0),
    )
    fig.tight_layout(pad=0.2, w_pad=0.6)
    return fig


# The camera that draws all three axes at one scale on the page: the
# isometric one, elev = asin(tan 30 deg), azim = -45. Every other angle
# foreshortens the three axes by different amounts — at (22, -60) a metre
# of z takes 1.56x the page length of a metre of y, which reads as z
# being plotted on a bigger scale even when the data scaling is equal.
ISO_VIEW = (float(np.degrees(np.arcsin(np.tan(np.radians(30))))), -45.0)
DEFAULT_VIEW = (22.0, -60.0)

SOLO_FONT_SCALE = 2.5
SOLO_ZLIM = (0.0, 2.0)  # floor to well above the plan, in metres


def fig_xysolo(
    plt,
    runs_by_speed,
    shape,
    width,
    three_d=False,
    view=None,
    font_scale=SOLO_FONT_SCALE,
    zlim=SOLO_ZLIM,
):
    """`xypair` split into standalone one-panel figures, one per profile.

    Same camera, same colours and — crucially — the same limits and box
    as `xypair`, computed from the two profiles' points pooled, so the
    panels stay comparable once they are no longer side by side and the
    reader can place them next to each other in any layout the paper
    wants. The differences are what a figure that stands on its own
    needs: every panel keeps its own axis furniture and legend, the text
    is `font_scale`x the shared style so it survives being printed at
    half the width, and the vertical axis is *not* exaggerated — a
    single panel has no neighbour to be consistent with, so an honest
    z axis costs nothing here. x, y and z are drawn at one common
    metres-per-unit: a metre of altitude is the same length on the page
    as a metre of north, and the only thing that shortens an axis is the
    camera's own foreshortening.

    Returns {speed: figure}.
    """
    # Isometric by default: this figure is the one that claims an honest
    # z axis, and a camera that foreshortens the three axes differently
    # undoes that claim on the page whatever the data scaling says.
    view = view or ISO_VIEW
    figs, axes, clouds = {}, {}, []
    kw = {"subplot_kw": {"projection": "3d"}} if three_d else {}
    # A 3D panel is square whatever canvas it is given, and its labels
    # land wherever the projection puts them — off the edge entirely at
    # some cameras, where they are never drawn and no crop can bring
    # them back. So the canvas is the square panel plus a margin wide
    # enough for any of them, and `ink_bbox` takes the surplus back at
    # save time: the margin only has to be big enough, never exact.
    side = width * 1.25
    margin = 0.45 * font_scale
    size = (side + 2 * margin, side + 2 * margin) if three_d else (side, side * 1.30)
    for speed in PAIR_SPEEDS:
        fig, ax = plt.subplots(figsize=size, **kw)
        clouds.append(draw_track(ax, runs_by_speed[speed][shape], three_d))
        figs[speed], axes[speed] = fig, ax
    common = np.vstack(clouds)

    label_size = plt.rcParams["axes.labelsize"] * font_scale
    for speed, ax in axes.items():
        scale_text(ax, font_scale)
        tick_budget(ax, three_d)
        if three_d:
            # z_floor=0 -> the box keeps the mission's true proportions.
            limits_3d(ax, common, z_floor=0.0, zoom=1.05, zlim=zlim)
            ax.view_init(*view)
            # A 3D axes already offsets each label by the size of that
            # axis' tick labels, so `labelpad` is a top-up, not the whole
            # standoff — a little is enough, and the z label sits along
            # the outside edge where less still is.
            label_axes(ax, True, size=label_size, pad=font_scale * np.r_[8, 8, 0])
        else:
            lo, hi = common[:, :2].min(0), common[:, :2].max(0)
            ctr, r = (hi + lo) / 2, (hi - lo) / 2 * 1.04
            ax.set_xlim(ctr[0] - r[0], ctr[0] + r[0])
            ax.set_ylim(ctr[1] - r[1], ctr[1] + r[1])
            label_axes(ax, False, size=label_size)
        legend_kw = dict(
            fontsize=plt.rcParams["legend.fontsize"] * font_scale,
            handlelength=1.3,
            handletextpad=0.4,
            labelspacing=0.3,
        )
        if three_d:
            m = margin / (side + 2 * margin)
            figs[speed].subplots_adjust(m, m, 1 - m, 1 - m)
            fit_panel(figs[speed], ax)
            # One legend for the pair, on the first panel. The two
            # figures are the same three lines in the same three colours
            # and are read together; repeating the key on both spends a
            # corner of each to say the same thing twice.
            if speed == PAIR_SPEEDS[0]:
                ax.legend(
                    loc="lower right",
                    frameon=True,
                    # borderpad=0.0,
                    # borderaxespad=0.0,
                    bbox_to_anchor=(1.25, 0.18),
                    **legend_kw,
                )
            figs[speed].crop_to_ink = True
        else:
            ax.legend(
                loc="lower right",
                framealpha=0.9,
                borderpad=0.3,
                bbox_to_anchor=(1.0, 1.0),
                **legend_kw,
            )
            figs[speed].tight_layout(pad=0.2)
    return figs


FIGURES = {
    "xy": fig_xy,
    "error": fig_error,
    "effort": fig_effort,
    "summary": fig_summary,
}
# The trajectory figures accept `three_d`/`view`; the rest do not.
THREE_D_CAPABLE = {"xy", "xypair", "xysolo"}
# Figures that span both speed profiles rather than one — driven by
# `--shape` instead of `--speed`.
PAIR_FIGURES = {"xypair": fig_xypair, "xysolo": fig_xysolo}
PAIR_SPEEDS = ("mid", "timeopt")


# ── text summary ─────────────────────────────────────────────────────────


def print_table(runs, speed):
    hdr = f"{'shape':<10}{'metric':<26}{'nominal':>10}{'payload':>10}{'delta':>11}"
    print(f"\n{pf.SPEED_LABEL[speed]} profile — payload vs nominal")
    print(hdr, "-" * len(hdr), sep="\n")
    for shape in runs:
        for j, (key, _label, plain, scale, digits, delta) in enumerate(BARS):
            n = runs[shape]["nominal"].get(key)
            l = runs[shape]["load"].get(key)
            if n is None or l is None:
                continue
            if delta == "abs":
                d = f"{(l - n) * scale:+.{digits}f} pt"
            else:
                d = f"{100 * (l / n - 1):+.0f} %" if n else "—"
            print(
                f"{shape if j == 0 else '':<10}{plain:<26}"
                f"{n * scale:>10.{digits}f}{l * scale:>10.{digits}f}{d:>11}"
            )
        print()


# ── main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--speed",
        choices=("slow", "mid", "timeopt"),
        default="mid",
        help="speed profile to compare (default mid; `timeopt` has "
        "only the figure-8 flown both ways)",
    )
    ap.add_argument(
        "--shape",
        choices=pm.SHAPES,
        default="figure8",
        help="shape for the paired `xypair` figure, which shows one "
        "shape at both profiles (default figure8 — the only one "
        "flown with a payload at the time-optimal profile)",
    )
    ap.add_argument("--datasets", default=pm.DATASETS)
    ap.add_argument("--outdir", metavar="DIR", help="write the figures here")
    ap.add_argument(
        "--only",
        choices=sorted(FIGURES) + sorted(PAIR_FIGURES),
        action="append",
        help="emit only this figure (repeatable)",
    )
    ap.add_argument("--show", action="store_true", help="open the figures")
    ap.add_argument(
        "--width",
        type=float,
        default=3.2,
        help="width of one column of panels, inches (default 3.2)",
    )
    ap.add_argument(
        "--format",
        default="pdf,png",
        help="comma-separated output formats (default pdf,png)",
    )
    ap.add_argument(
        "--3d",
        dest="three_d",
        action="store_true",
        help="draw the trajectory figures (xy, xypair) in 3D. The "
        "vertical axis is exaggerated when the mission is much "
        "wider than it is tall; the factor is written on the "
        "z label",
    )
    ap.add_argument(
        "--view",
        metavar="ELEV,AZIM",
        help="3D camera angle in degrees; overrides each figure's own "
        f"default ({DEFAULT_VIEW[0]:.0f},{DEFAULT_VIEW[1]:.0f} for xy and "
        f"xypair, isometric {ISO_VIEW[0]:.1f},{ISO_VIEW[1]:.0f} for "
        "xysolo, which draws all three axes at one scale)",
    )
    ap.add_argument("--no-table", action="store_true")
    args = ap.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pf.style()
    view = None
    if args.view:
        try:
            view = tuple(float(v) for v in args.view.split(","))
            assert len(view) == 2
        except (ValueError, AssertionError):
            sys.exit(f"--view: expected ELEV,AZIM in degrees, got {args.view!r}")
    extra = {"three_d": args.three_d, "view": view}

    cache = {}

    def runs_for(speed):
        if speed not in cache:
            cache[speed] = load_all(speed, args.datasets)
            if not cache[speed]:
                sys.exit(
                    f"no shape at the {speed} profile has both a nominal and a "
                    f"_load recording — see analysis/datasets/indoor_exp_ablation/"
                )
        return cache[speed]

    names = args.only or list(FIGURES) + list(PAIR_FIGURES)
    figs = {}
    for name in names:
        if name in PAIR_FIGURES:
            by_speed = {sp: runs_for(sp) for sp in PAIR_SPEEDS}
            missing = [sp for sp in PAIR_SPEEDS if args.shape not in by_speed[sp]]
            if missing:
                sys.exit(
                    f"{name}: {args.shape} was not flown with a payload at the "
                    f"{', '.join(missing)} profile — try --shape figure8"
                )
            made = PAIR_FIGURES[name](
                plt,
                by_speed,
                args.shape,
                args.width,
                **(extra if name in THREE_D_CAPABLE else {}),
            )
            # a pair figure may hand back one figure per profile instead
            # of a single multi-panel one; the profile joins the filename.
            if isinstance(made, dict):
                figs.update(((name, f"{args.shape}_{sp}"), f) for sp, f in made.items())
            else:
                figs[(name, args.shape)] = made
        else:
            figs[(name, args.speed)] = FIGURES[name](
                plt,
                runs_for(args.speed),
                args.speed,
                args.width,
                **(extra if name in THREE_D_CAPABLE else {}),
            )

    if not args.no_table:
        for speed in cache:
            print_table(cache[speed], speed)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        for (name, tag), fig in figs.items():
            for ext in args.format.split(","):
                three_d = args.three_d and name in THREE_D_CAPABLE
                dim = "_3d" if three_d else ""
                out = os.path.join(args.outdir, f"fig_payload_{tag}_{name}{dim}.{ext}")
                # matplotlib's tight bbox does not see 3D axis labels, so
                # it crops the z label off. A figure either asks to be
                # cropped to its own ink, or sets its own margins and is
                # saved with the bbox left alone.
                bbox = ink_bbox(fig) if getattr(fig, "crop_to_ink", False) else None
                if bbox is not None:
                    fig.savefig(out, bbox_inches=bbox)
                else:
                    ctx = {"savefig.bbox": None} if three_d else {}
                    with matplotlib.rc_context(ctx):
                        fig.savefig(out)
                print(f"wrote {out}")
    elif not args.show:
        print("\nnote: neither --outdir nor --show given — nothing written")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
