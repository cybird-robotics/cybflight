#!/usr/bin/env python3
"""Dump and sanity-check a cybflight blackbox MCAP recording.

The greppable companion to the firmware's SD-card blackbox
(docs/blackbox.md). Default mode prints one line per message:

    +0.123456s /topic       field=value field=value ...

so the doc's verification recipes work as written:

    python3 analysis/read_mcap.py flight_0001.mcap | head        # ARM up top
    python3 analysis/read_mcap.py flight_0001.mcap | grep events # event order
    python3 analysis/read_mcap.py flight_0001.mcap | grep /rc    # stick samples

Modes:
    (default)        pure line-per-message dump of every topic
    --summary        channel table (count, rate, seq-gaps) + event list, no dump
    --topic /imu1    dump only one topic (repeatable)

Decoding is schema-driven: cybflight's compact positional-array topics
(`/imu1`, `/imu1_raw`, `/odometry` — `*.v2` schemas) are labeled from
the `prefixItems` titles embedded in the file's own Schema records, so
this script does not go stale when a wire format gains fields. Map-
encoded topics print their keys as-is. `/events` kinds are named from
a mirror of `topics::events::KIND_*`.

Note: the firmware writes streamed MCAP with no summary section
(legal; `mcap doctor` passes), so this reads linearly via StreamReader
— indexed readers that require a summary will not work on these files.

For rich interactive plots use the sibling `cybflight-review` repo's
Bokeh report; for quick trajectory PNGs its (older) read_mcap.py.

Requires: pip install mcap cbor2
"""

import argparse
import collections
import json
import signal
import sys

# Don't traceback when piped into `head` / `grep -m`.
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

try:
    import cbor2
    from mcap.stream_reader import StreamReader
    from mcap import records as R
except ImportError:
    sys.exit("missing deps: pip install mcap cbor2")

# Mirror of `cybflight::blackbox::topics::events::KIND_*`.
EVENT_KINDS = {
    0x01: "ARM",
    0x02: "DISARM",
    0x03: "FAILSAFE",
    0x04: "FAILSAFE_CLEAR",
    0x05: "ESTIMATOR_DOWN",
    0x06: "ESTIMATOR_UP",
    0x07: "RC_LOSS",
    0x08: "RC_RECOVERED",
    0x09: "MISSION_PLANNING",
    0x0A: "MISSION_EXECUTING",
    0x0B: "MISSION_IDLE",
    0x0C: "INNER_SILENT",
    0x0D: "POWER_STALE",
    0x0E: "POWER_OK",
    0x0F: "RECORDER_OVERRUN",
    0x10: "LOG_END",
    0x20: "PANIC",
    0x21: "HARDFAULT",
    0x22: "BROWNOUT",
    0x23: "IWDG_RESET",
    0x24: "BOOT_POSTMORTEM",
}


def fmt_val(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return "[" + ",".join(fmt_val(x) for x in v) + "]"
    return str(v)


def array_labels(schema_json):
    """prefixItems titles for a positional-array schema, else None."""
    try:
        s = json.loads(schema_json)
        if s.get("type") == "array" and "prefixItems" in s:
            return [it.get("title", f"[{i}]") for i, it in enumerate(s["prefixItems"])]
    except (ValueError, AttributeError):
        pass
    return None


# What `data` means, per kind — mirror of the `data` description in
# `topics::events::SCHEMA`. Each entry renders the number into words;
# kinds absent from here carry no data (or a self-evident count).
_FAILSAFE_REASONS = {0: "commanded", 1: "ControllerTimeout", 2: "RcLoss"}
_MISSION_STATES = {0: "from Idle", 1: "from Planning", 2: "from Executing"}
EVENT_DATA = {
    0x02: lambda d: _FAILSAFE_REASONS.get(d, f"cause {d}"),
    0x03: lambda d: _FAILSAFE_REASONS.get(d, f"reason {d}"),
    0x0B: lambda d: _MISSION_STATES.get(d, f"from {d}"),
    0x0C: lambda d: {1: "voltage stale", 2: "WLS NaN"}.get(d, f"cause {d}"),
    0x0D: lambda d: f"episode {d}",
    0x0E: lambda d: f"held {d} ms",
    0x0F: lambda d: f"{d} records dropped",
}


def render(topic, obj, labels):
    """One message -> (timestamp_ns or None, 'k=v k=v ...')."""
    if topic == "/events" and isinstance(obj, dict):
        kind = obj.get("kind")
        name = EVENT_KINDS.get(kind, f"KIND_{kind}")
        data = obj.get("data", 0)
        decode = EVENT_DATA.get(kind)
        if decode is not None:
            # Rendered even when zero: `DISARM data=0` is a real
            # statement (the pilot disarmed), not an absent field.
            body = f"{name} ({decode(data)})"
        else:
            body = name + (f" data={data}" if data else "")
        return obj.get("timestamp_ns"), body
    if isinstance(obj, list) and labels:
        ts = obj[0] if labels and labels[0] == "timestamp_ns" else None
        pairs = [f"{k}={fmt_val(v)}" for k, v in zip(labels, obj)]
        if ts is not None:
            pairs = pairs[1:]
        return ts, " ".join(pairs)
    if isinstance(obj, dict):
        ts = obj.get("timestamp_ns")
        pairs = [f"{k}={fmt_val(v)}" for k, v in obj.items() if k != "timestamp_ns"]
        return ts, " ".join(pairs)
    return None, fmt_val(obj)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[2:]),
    )
    ap.add_argument("path", help="path to a cybflight .mcap recording")
    ap.add_argument("--summary", action="store_true",
                    help="channel table + events only, no per-message dump")
    ap.add_argument("--topic", action="append", default=None, metavar="/name",
                    help="dump only this topic (repeatable)")
    args = ap.parse_args()

    schemas = {}          # schema id -> (name, labels-or-None)
    chans = {}            # channel id -> (topic, labels-or-None)
    counts = collections.Counter()
    first_ts, last_ts = {}, {}
    seq_gaps = collections.Counter()
    prev_seq = {}
    events = []
    t0 = None
    header = None
    got_footer = False

    for rec in StreamReader(open(args.path, "rb")).records:
        if isinstance(rec, R.Header):
            header = rec
        elif isinstance(rec, R.Schema):
            schemas[rec.id] = (rec.name, array_labels(rec.data))
        elif isinstance(rec, R.Channel):
            name, labels = schemas.get(rec.schema_id, ("?", None))
            chans[rec.id] = (rec.topic, labels)
        elif isinstance(rec, R.Footer):
            got_footer = True
        elif isinstance(rec, R.Message):
            topic, labels = chans.get(rec.channel_id, (f"ch{rec.channel_id}", None))
            counts[rec.channel_id] += 1
            first_ts.setdefault(rec.channel_id, rec.log_time)
            last_ts[rec.channel_id] = rec.log_time
            p = prev_seq.get(rec.channel_id)
            if p is not None and rec.sequence != p + 1:
                seq_gaps[rec.channel_id] += 1
            prev_seq[rec.channel_id] = rec.sequence
            try:
                obj = cbor2.loads(rec.data)
            except Exception as e:  # corrupt tail of a crashed session
                print(f"!! {topic}: undecodable CBOR ({e})", file=sys.stderr)
                continue
            ts, body = render(topic, obj, labels)
            t = ts if ts is not None else rec.log_time
            if t0 is None:
                t0 = t
            if topic == "/events":
                events.append((t, body))
            if args.summary:
                continue
            if args.topic and topic not in args.topic:
                continue
            print(f"+{(t - t0) / 1e9:10.6f}s {topic:18s} {body}")

    if not args.summary:
        return 0  # default mode is a pure dump — greppable, tail ends at LOG_END

    if header is not None:
        print(f"-- {args.path}  (library: {header.library}"
              + ("" if got_footer else "  !! NO FOOTER — truncated/crashed session")
              + ")")
    for cid in sorted(chans):
        topic, _ = chans[cid]
        n = counts.get(cid, 0)
        d = (last_ts.get(cid, 0) - first_ts.get(cid, 0)) / 1e9 if n > 1 else 0.0
        hz = f"{n / d:7.1f} Hz" if d > 0 else "      —"
        gap = f"  seq-gaps(drops): {seq_gaps[cid]}" if seq_gaps.get(cid) else ""
        print(f"   ch{cid:>2} {topic:18s} {n:7d} msgs {hz}{gap}")
    if events:
        print(f"   events ({len(events)}):")
        for t, body in events:
            print(f"     +{(t - t0) / 1e9:9.3f}s  {body}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
