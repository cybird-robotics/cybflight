#!/usr/bin/env python3
"""Plot a cybflight blackbox MCAP recording (matplotlib).

Three figures, each produced only when its topic has data:

  1. **XY position** — top-down 2D trajectory from `/odometry`
     (`pos_x`/`pos_y`, ENU), start/end markers, equal aspect.
  2. **Attitude** — roll / pitch / yaw in degrees vs time, converted
     from the logged quaternions ([w,i,j,k]). Default is the quadrotor
     tilt–torsion split (continuous through 90° tilt in any direction;
     equals RPY for small angles); `--euler zyx|zxy` for classic sequences.
     Plots BOTH estimates when present: `/attitude` (Mahony IMU-only,
     ~100 Hz, solid) and `/odometry.pose.orientation` (ESKF, ~1 kHz,
     dashed) — divergence between them is diagnostic.
  3. **IMU** — accel [m/s²] and gyro [rad/s] xyz vs time from `/imu1`
     (post-LP; pass --raw to plot `/imu1_raw` instead, Large tier).

ARM / DISARM / FAILSAFE events overlay as vertical lines on the time
axes. Time is seconds since the first message.

``--from-mission`` trims every topic to the part of the log after the
mission trigger: the first rising edge of the AUX mission channel in
``/rc`` (Schmitt hysteresis, mirroring `control/rc_interpreter.rs`:
above ``rc_mission_high_us`` from a confirmed-low level). The channel
index and thresholds default to the firmware fallbacks (ch4, 1700/1300
µs) — pass ``--mission-channel/--mission-high/--mission-low`` if the
vehicle YAML pins different values. Three ways to trim the tail
(combinable; the earliest wins): ``--until-idle`` cuts at the first
``MISSION_IDLE`` event after the trigger — the firmware's own
"mission over" signal (natural end or abort); ``--until-abort`` cuts
at the next falling edge of the mission switch; ``--until-descent``
cuts when the pilot first commands a descent (throttle below
``988 + 512·(1 − rc_throttle_deadband)`` ≈ 1438 µs by default).

Usage:
    python3 analysis/plot_mcap.py logs/flight_0001.mcap
    python3 analysis/plot_mcap.py f.mcap --save out        # out_xy.png, out_attitude.png, out_imu.png
    python3 analysis/plot_mcap.py f.mcap --no-show --save out
    python3 analysis/plot_mcap.py f.mcap --raw             # imu figure uses /imu1_raw
    python3 analysis/plot_mcap.py f.mcap --from-mission    # only after the mission switch flips
    python3 analysis/plot_mcap.py f.mcap --from-mission --until-idle      # mission execution only

Decoding is schema-driven like analysis/read_mcap.py — the positional
`.v2` array topics are labeled from the prefixItems titles in the
file's own Schema records.

Requires: pip install mcap cbor2 numpy matplotlib
"""

import argparse
import collections
import json
import math
import sys

try:
    import cbor2
    import numpy as np
    from mcap.stream_reader import StreamReader
    from mcap import records as R
except ImportError:
    sys.exit("missing deps: pip install mcap cbor2 numpy matplotlib")

EVENT_KINDS = {  # mirror of topics::events::KIND_* (subset worth overlaying)
    0x01: ("ARM", "tab:green"),
    0x02: ("DISARM", "tab:red"),
    0x03: ("FAILSAFE", "tab:orange"),
    0x04: ("FAILSAFE_CLEAR", "tab:olive"),
    0x09: ("MISSION_PLANNING", "tab:purple"),
    0x0A: ("MISSION_EXECUTING", "tab:blue"),
    0x0B: ("MISSION_IDLE", "tab:brown"),
    0x0C: ("INNER_SILENT", "tab:red"),
    0x10: ("LOG_END", "gray"),
}


def array_labels(schema_json):
    try:
        s = json.loads(schema_json)
        if s.get("type") == "array" and "prefixItems" in s:
            return [it.get("title", f"[{i}]") for i, it in enumerate(s["prefixItems"])]
    except (ValueError, AttributeError):
        pass
    return None


def load(path):
    """Return {topic: list-of-dicts} with positional arrays label-mapped."""
    schemas, chans = {}, {}
    out = collections.defaultdict(list)
    for rec in StreamReader(open(path, "rb")).records:
        if isinstance(rec, R.Schema):
            schemas[rec.id] = array_labels(rec.data)
        elif isinstance(rec, R.Channel):
            chans[rec.id] = (rec.topic, schemas.get(rec.schema_id))
        elif isinstance(rec, R.Message):
            topic, labels = chans.get(rec.channel_id, (None, None))
            if topic is None:
                continue
            obj = cbor2.loads(rec.data)
            if isinstance(obj, list) and labels:
                obj = dict(zip(labels, obj))
            if isinstance(obj, dict):
                obj.setdefault("timestamp_ns", rec.log_time)
                out[topic].append(obj)
    return out


def col(msgs, key):
    return np.array([m[key] for m in msgs], dtype=np.float64)


def quat_to_rotm(w, i, j, k):
    """[w,i,j,k] quaternion arrays -> (N,3,3) rotation matrices."""
    R = np.empty((len(w), 3, 3))
    R[:, 0, 0] = 1 - 2 * (j * j + k * k); R[:, 0, 1] = 2 * (i * j - k * w); R[:, 0, 2] = 2 * (i * k + j * w)
    R[:, 1, 0] = 2 * (i * j + k * w);     R[:, 1, 1] = 1 - 2 * (i * i + k * k); R[:, 1, 2] = 2 * (j * k - i * w)
    R[:, 2, 0] = 2 * (i * k - j * w);     R[:, 2, 1] = 2 * (j * k + i * w);     R[:, 2, 2] = 1 - 2 * (i * i + j * j)
    return R


def quat_to_rpy_deg(w, i, j, k, seq="tilt"):
    """[w,i,j,k] -> (roll, pitch, yaw) in degrees.

    seq="tilt": tilt–torsion decomposition R = Rz(yaw) · R_tilt, where
      R_tilt is the shortest rotation taking world +Z onto the body +Z
      (thrust axis) and its rotation vector lies in the horizontal plane.
      roll/pitch are that vector's x/y components in the heading frame — identical to the
      classic angles for small tilt, and continuous all the way to a
      90° (and beyond) tilt in *any* direction. yaw is the remaining
      torsion about the thrust axis. The only singularity is a fully
      inverted vehicle (tilt = 180°). This is the standard quadrotor
      attitude split (Brescianini & D'Andrea) and the default here
      because the time-optimal missions tilt to ~88° both fore/aft and
      sideways, which gimbal-locks every 3-angle Euler sequence.
    seq="zyx": aerospace yaw-pitch-roll (R = Rz·Ry·Rx), pitch ∈ ±90°;
      locks when pitching near vertical.
    seq="zxy": yaw-roll-pitch (R = Rz·Rx·Ry), roll ∈ ±90°, pitch ∈ ±180°;
      locks when rolling near vertical.
    """
    w, i, j, k = (np.asarray(a, dtype=np.float64) for a in (w, i, j, k))
    if seq == "zyx":
        roll = np.degrees(np.arctan2(2 * (w * i + j * k), 1 - 2 * (i * i + j * j)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (w * j - i * k), -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(2 * (w * k + i * j), 1 - 2 * (j * j + k * k)))
        return roll, pitch, yaw
    R = quat_to_rotm(w, i, j, k)
    if seq == "zxy":
        roll = np.degrees(np.arcsin(np.clip(R[:, 2, 1], -1.0, 1.0)))
        pitch = np.degrees(np.arctan2(-R[:, 2, 0], R[:, 2, 2]))
        yaw = np.degrees(np.arctan2(-R[:, 0, 1], R[:, 1, 1]))
        return roll, pitch, yaw
    if seq != "tilt":
        raise ValueError(f"unknown euler sequence {seq!r}")
    bz = R[:, :, 2]  # body +Z in world frame
    theta = np.arccos(np.clip(bz[:, 2], -1.0, 1.0))
    sin_t = np.sin(theta)
    # axis = ez × bz / |…| = (-bz_y, bz_x, 0)/sin θ; rotation vector = θ·axis.
    # θ/sin θ → 1 as θ → 0, so guard the division and use that limit.
    scale = np.where(sin_t > 1e-9, theta / np.where(sin_t > 1e-9, sin_t, 1.0), 1.0)
    rx, ry = -bz[:, 1] * scale, bz[:, 0] * scale
    # R_tilt via Rodrigues from unit axis n and angle θ. Because n is
    # expressed in the world frame, the split is R = R_tilt · Rz(ψ)
    # (yaw first, about the body axis), so the torsion is Rz = R_tiltᵀ·R.
    nx = np.where(sin_t > 1e-9, -bz[:, 1] / np.where(sin_t > 1e-9, sin_t, 1.0), 0.0)
    ny = np.where(sin_t > 1e-9, bz[:, 0] / np.where(sin_t > 1e-9, sin_t, 1.0), 0.0)
    c, s_ = np.cos(theta), sin_t
    Rt = np.empty_like(R)
    Rt[:, 0, 0] = c + nx * nx * (1 - c); Rt[:, 0, 1] = nx * ny * (1 - c);     Rt[:, 0, 2] = ny * s_
    Rt[:, 1, 0] = nx * ny * (1 - c);     Rt[:, 1, 1] = c + ny * ny * (1 - c); Rt[:, 1, 2] = -nx * s_
    Rt[:, 2, 0] = -ny * s_;              Rt[:, 2, 1] = nx * s_;               Rt[:, 2, 2] = c
    Rz = np.einsum("nji,njk->nik", Rt, R)  # R_tiltᵀ · R
    yaw = np.arctan2(Rz[:, 1, 0], Rz[:, 0, 0])
    # R_tilt(n)·Rz(ψ) = Rz(ψ)·R_tilt(Rz(−ψ)n): rotate the tilt vector into
    # the heading frame so roll/pitch mean "about the vehicle's own x/y",
    # matching classic RPY for small angles regardless of heading.
    cy, sy = np.cos(yaw), np.sin(yaw)
    roll, pitch = cy * rx + sy * ry, -sy * rx + cy * ry
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def unwrap_deg(a):
    """Unwrap an euler-angle time series (degrees) so a ±180° crossing is
    a continuous line rather than a 360° jump."""
    return np.degrees(np.unwrap(np.radians(np.asarray(a, dtype=np.float64))))


def euler_deg(w, i, j, k, seq="tilt"):
    """Plot-ready (roll, pitch, yaw): `quat_to_rpy_deg(seq)` + unwrap."""
    return tuple(unwrap_deg(a) for a in quat_to_rpy_deg(w, i, j, k, seq))


def mission_edges(rc, channel, high_us, low_us):
    """Return (rise_ns, fall_ns) of the first mission-trigger rising edge
    and the first falling edge after it, or None for whichever is absent.

    Same hysteretic level detection as `rc_interpreter_task`: the first
    frame seeds the confirmed level (switch-high at boot is NOT an edge),
    then level goes high above `high_us` and low below `low_us`.
    """
    confirmed = None
    rise = fall = None
    for m in rc:
        ch = m.get("channels", [])
        if len(ch) <= channel:
            continue  # short frame: hold confirmed level
        v = ch[channel]
        if confirmed is None:
            confirmed = v > high_us
            continue
        level = v > low_us if confirmed else v > high_us
        if level == confirmed:
            continue
        confirmed = level
        if level and rise is None:
            rise = m["timestamp_ns"]
        elif not level and rise is not None:
            fall = m["timestamp_ns"]
            break
    return rise, fall


def first_descent(rc, after_ns, channel, threshold_us):
    """Timestamp of the first /rc frame after `after_ns` whose throttle
    is below `threshold_us` (pilot commanding a descent), or None."""
    for m in rc:
        if m["timestamp_ns"] < after_ns:
            continue
        ch = m.get("channels", [])
        if len(ch) > channel and ch[channel] < threshold_us:
            return m["timestamp_ns"]
    return None


KIND_DISARM = 0x02
KIND_MISSION_PLANNING = 0x09
KIND_MISSION_EXECUTING = 0x0A
KIND_MISSION_IDLE = 0x0B


def first_event(events, kinds, after_ns=None):
    """Timestamp of the first event of any kind in `kinds` at/after
    `after_ns` (unbounded when None), or None."""
    for ev in events:
        if ev.get("kind") in kinds and (after_ns is None
                                        or ev["timestamp_ns"] >= after_ns):
            return ev["timestamp_ns"]
    return None


def first_mission_idle(events, after_ns):
    """Timestamp of the first MISSION_IDLE event at/after `after_ns`, or None."""
    return first_event(events, (KIND_MISSION_IDLE,), after_ns)


def mission_start(data, channel, high_us, low_us):
    """`(start_ns, fall_ns, source)` of the mission trigger, or `(None,
    None, None)` if the recording carries neither cue.

    The preferred cue is the `/rc` mission switch: it is the pilot's
    actual trigger instant, and every recording under
    `analysis/datasets` was cropped by it. `/rc` is optional, though —
    recordings that mute the topic fall back to the firmware's own
    mission-state events, which mark the same transition from the other
    side: `MISSION_PLANNING` is emitted on the very Idle -> Planning
    step the switch causes, and `MISSION_EXECUTING` covers the case
    where the recorder's edge-detect poll coalesced Idle -> Planning ->
    Executing into a single observed change (so no PLANNING event
    exists). The event-based start is therefore at most one planner
    solve later than the switch flip; `paper_metrics.startup_lag`
    absorbs exactly that offset.

    `fall_ns` (the mission-abort falling edge) only exists for the `/rc`
    cue — there is no event for "switch released".
    """
    rc = data.get("/rc", [])
    if rc:
        rise, fall = mission_edges(rc, channel, high_us, low_us)
        if rise is not None:
            return rise, fall, f"/rc channel {channel}"
    events = data.get("/events", [])
    for kind, name in ((KIND_MISSION_PLANNING, "MISSION_PLANNING"),
                       (KIND_MISSION_EXECUTING, "MISSION_EXECUTING")):
        t = first_event(events, (kind,))
        if t is not None:
            return t, None, f"{name} event"
    return None, None, None


def mission_end(data, start_ns, throttle_channel, throttle_deadband):
    """`(end_ns, source)` of the mission segment that opened at
    `start_ns`. `end_ns` is None when nothing ends it (end of log).

    Preference order: the firmware's own MISSION_IDLE ("the trajectory
    is over"), then the pilot's first descent command on `/rc`, then
    DISARM — the last two being fallbacks for logs that lost the idle
    transition, and DISARM being the only one of the three that
    survives without `/rc`.
    """
    events = data.get("/events", [])
    idle = first_mission_idle(events, start_ns)
    if idle is not None:
        return idle, "MISSION_IDLE event"
    rc = data.get("/rc", [])
    if rc:
        thr = 988 + 512 * (1.0 - throttle_deadband)
        desc = first_descent(rc, start_ns, throttle_channel, thr)
        if desc is not None:
            return desc, "first descent command"
    disarm = first_event(events, (KIND_DISARM,), start_ns)
    if disarm is not None:
        return disarm, "DISARM event"
    return None, "end of log"


def mission_window(data, args):
    """`(start_ns, end_ns, start_src, end_src)` of the mission segment,
    using whichever cues the recording carries — the crop every dataset
    script shares. `start_ns` is None when the recording has no mission
    trigger at all; `end_ns` is None for "to the end of the log".

    `args` supplies the `/rc` mapping (`mission_channel`,
    `mission_high`, `mission_low`, `throttle_channel`,
    `throttle_deadband`) and is ignored where the fallbacks apply.
    """
    start, _, start_src = mission_start(data, args.mission_channel,
                                        args.mission_high, args.mission_low)
    if start is None:
        return None, None, None, None
    end, end_src = mission_end(data, start, args.throttle_channel,
                               args.throttle_deadband)
    return start, end, start_src, end_src


def crop(data, start_ns=None, end_ns=None):
    """Drop messages outside [start_ns, end_ns] on every topic, in place."""
    for topic, msgs in data.items():
        data[topic] = [m for m in msgs
                       if (start_ns is None or m["timestamp_ns"] >= start_ns)
                       and (end_ns is None or m["timestamp_ns"] <= end_ns)]


def overlay_events(ax, events, t0_ns):
    seen = set()
    for ev in events:
        kind = ev.get("kind")
        name, color = EVENT_KINDS.get(kind, (None, None))
        if name is None:
            continue
        t = (ev["timestamp_ns"] - t0_ns) / 1e9
        ax.axvline(t, color=color, ls="--", lw=0.8, alpha=0.6,
                   label=name if name not in seen else None)
        seen.add(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="path to a cybflight .mcap recording")
    ap.add_argument("--save", metavar="PREFIX",
                    help="write PREFIX_xy.png / PREFIX_attitude.png / PREFIX_imu.png")
    ap.add_argument("--no-show", action="store_true", help="skip interactive show")
    ap.add_argument("--raw", action="store_true",
                    help="IMU figure uses /imu1_raw (pre-filter) instead of /imu1")
    ap.add_argument("--euler", choices=("tilt", "zxy", "zyx"), default="tilt",
                    help="attitude split for the plot: tilt (default; tilt-torsion "
                         "roll/pitch + torsion yaw, no gimbal lock below 180° tilt), "
                         "zxy or zyx euler sequences")
    ap.add_argument("--from-mission", action="store_true",
                    help="plot only from the mission trigger onward: the first "
                         "rising edge on the /rc mission channel, or — for "
                         "recordings without /rc — the first MISSION_PLANNING / "
                         "MISSION_EXECUTING event")
    ap.add_argument("--until-abort", action="store_true",
                    help="with --from-mission: also cut at the next falling edge "
                         "(mission abort / switch released)")
    ap.add_argument("--until-idle", action="store_true",
                    help="with --from-mission: cut at the first MISSION_IDLE event "
                         "after the trigger (firmware's mission-over signal)")
    ap.add_argument("--until-descent", action="store_true",
                    help="with --from-mission: cut when the throttle stick first "
                         "commands a descent after the trigger (end of mission)")
    ap.add_argument("--throttle-channel", type=int, default=2,
                    help="0-based /rc throttle channel index (default 2)")
    ap.add_argument("--throttle-deadband", type=float, default=0.12,
                    help="rc_throttle_deadband; descent = throttle below "
                         "988 + 512*(1-deadband) µs (default 0.12 → ~1438 µs)")
    ap.add_argument("--mission-channel", type=int, default=4,
                    help="0-based /rc channel index of the mission switch "
                         "(rc_mission_channel, default 4)")
    ap.add_argument("--mission-high", type=int, default=1700,
                    help="rising threshold in µs (rc_mission_high_us, default 1700)")
    ap.add_argument("--mission-low", type=int, default=1300,
                    help="falling threshold in µs (rc_mission_low_us, default 1300)")
    args = ap.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(args.path)
    if args.from_mission:
        rc = data.get("/rc", [])
        rise, fall, src = mission_start(data, args.mission_channel,
                                        args.mission_high, args.mission_low)
        if rise is None:
            sys.exit("--from-mission: this recording has neither a rising edge "
                     f"on /rc channel {args.mission_channel} (> {args.mission_high} µs) "
                     "nor a MISSION_PLANNING/MISSION_EXECUTING event")
        end, why = None, "to end of log"
        if args.until_abort and fall is not None:
            end, why = fall, "until mission-switch falling edge"
        elif args.until_abort:
            print("note: --until-abort: no falling edge after the trigger"
                  + ("" if rc else " (no /rc in this recording)"))
        if args.until_idle:
            idle = first_mission_idle(data.get("/events", []), rise)
            if idle is None:
                print("note: --until-idle: no MISSION_IDLE event after the trigger")
            elif end is None or idle < end:
                end, why = idle, "until MISSION_IDLE event"
        if args.until_descent:
            thr = 988 + 512 * (1.0 - args.throttle_deadband)
            desc = first_descent(rc, rise, args.throttle_channel, thr) if rc else None
            if desc is None:
                print(f"note: --until-descent: throttle never below {thr:.0f} µs "
                      "after the trigger"
                      + ("" if rc else " (no /rc in this recording)"))
            elif end is None or desc < end:
                end, why = desc, f"until first descent command (< {thr:.0f} µs)"
        crop(data, rise, end)
        span = f"{why} at t={(end - rise) / 1e9:.2f}s" if end else why
        print(f"mission trigger at {rise / 1e9:.3f}s (log time, from {src}); "
              f"plotting from there {span}")
    events = data.get("/events", [])
    all_ts = [m["timestamp_ns"] for msgs in data.values() for m in msgs]
    if not all_ts:
        sys.exit(f"{args.path}: no messages")
    t0 = min(all_ts)
    figs = {}

    # ── 1. XY position ──────────────────────────────────────────────
    odom = data.get("/odometry", [])
    if odom:
        x, y = col(odom, "pos_x_m"), col(odom, "pos_y_m")
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(x, y, lw=1)
        ax.scatter(x[0], y[0], c="green", s=40, zorder=3, label="start")
        ax.scatter(x[-1], y[-1], c="red", s=40, zorder=3, label="end")
        ax.set_xlabel("x east (m)")
        ax.set_ylabel("y north (m)")
        ax.set_title(f"{args.path} — XY trajectory (/odometry)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        figs["xy"] = fig
    else:
        print("note: no /odometry messages — skipping XY plot "
              "(estimator never initialised, or tier < mid)")

    # ── 2. Roll / pitch / yaw ───────────────────────────────────────
    att = data.get("/attitude", [])
    sources = []
    if att:
        q = np.array([m["quaternion"] for m in att], dtype=np.float64)
        t = (col(att, "timestamp_ns") - t0) / 1e9
        sources.append(("/attitude (Mahony)", t, euler_deg(*q.T, seq=args.euler), "-"))
    if odom:
        t = (col(odom, "timestamp_ns") - t0) / 1e9
        rpy = euler_deg(col(odom, "q_w"), col(odom, "q_i"),
                        col(odom, "q_j"), col(odom, "q_k"), seq=args.euler)
        sources.append(("/odometry (ESKF)", t, rpy, "--"))
    if sources:
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for ax, name, idx in zip(axes, ("roll", "pitch", "yaw"), range(3)):
            for label, t, rpy, ls in sources:
                ax.plot(t, rpy[idx], ls, lw=1, label=label)
            ax.set_ylabel(f"{name} (deg)")
            ax.grid(True, alpha=0.3)
            overlay_events(ax, events, t0)
            ax.legend(fontsize=8, loc="best")
        axes[-1].set_xlabel("t (s)")
        split = "tilt–torsion" if args.euler == "tilt" else f"{args.euler.upper()} euler"
        axes[0].set_title(f"{args.path} — attitude (quaternion → {split}, unwrapped)")
        fig.tight_layout()
        figs["attitude"] = fig
    else:
        print("note: no /attitude or /odometry messages — skipping attitude plot")

    # ── 3. IMU ──────────────────────────────────────────────────────
    imu_topic = "/imu1_raw" if args.raw else "/imu1"
    imu = data.get(imu_topic, [])
    if not imu and args.raw:
        print("note: no /imu1_raw (Large tier only) — falling back to /imu1")
        imu_topic, imu = "/imu1", data.get("/imu1", [])
    if imu:
        t = (col(imu, "timestamp_ns") - t0) / 1e9
        fig, (ax_a, ax_g) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for ax, kind, unit in ((ax_a, "accel", "m/s²"), (ax_g, "gyro", "rad/s")):
            for axis in "xyz":
                key = f"{kind}_{axis}_{'m_s2' if kind == 'accel' else 'rad_s'}"
                ax.plot(t, col(imu, key), lw=0.6, label=axis)
            ax.set_ylabel(f"{kind} ({unit})")
            ax.grid(True, alpha=0.3)
            overlay_events(ax, events, t0)
            ax.legend(fontsize=8, loc="best")
        ax_g.set_xlabel("t (s)")
        ax_a.set_title(f"{args.path} — {imu_topic}")
        fig.tight_layout()
        figs["imu"] = fig
    else:
        print(f"note: no {imu_topic} messages — skipping IMU plot (tier < mid?)")

    if not figs:
        sys.exit("nothing to plot")
    if args.save:
        for tag, fig in figs.items():
            out = f"{args.save}_{tag}.png"
            fig.savefig(out, dpi=130)
            print(f"saved {out}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
