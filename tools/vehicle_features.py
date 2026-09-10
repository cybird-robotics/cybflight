#!/usr/bin/env python3
"""Compose the cargo feature list for a firmware build.

Single source of truth for the vehicle's compile-time hardware selections
is the ``build:`` section of ``vehicles/$VEHICLE.yaml``; environment
variables (BOARD, RC_PROTOCOL, OUTER_LOOP, POS_SOURCE, GPS_MODEL, ROLE,
IMU_RATE) remain as *dev overrides* on top of it (a divergence is warned on stderr,
never fatal). Precedence per knob: env > YAML > hardcoded default.
DEFMT_UART and ESTIMATOR are env-only: bench knobs, not vehicle facts.

Deliberately stdlib-only (no PyYAML): the ``build:`` section is a flat
two-space-indented ``key: value`` block, parsed line-wise — the same
approach as tools/param_sync.py. Full structural validation happens in
the shared ``vehicle_yaml`` Rust loader at bake time; this script only
needs the strings.

Invoked *per recipe* by the Justfile (``just build [<vehicle>]``), not as
a top-level backtick on every ``just`` invocation, so it is free to fail:
a VEHICLE naming a nonexistent ``vehicles/*.yaml`` exits non-zero listing
the available names, rather than composing a confident feature list out
of hardcoded defaults for a vehicle that does not exist.

Unrecognised knob *values* still degrade to a poisoned feature name
(``outer_INVALID_x``) that cargo rejects — the vehicle FILE is the one
thing worth failing on before cargo starts. The build.rs feature↔YAML
guard remains the consistency enforcement point.

Usage:
  vehicle_features.py            print the feature list (one line, stdout)
  vehicle_features.py --explain  print per-knob provenance to stderr too
  vehicle_features.py --list     table of every vehicle and its build facts
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# knob -> (env var, hardcoded default)
KNOBS = {
    "board": ("BOARD", "sakurah743"),
    "rc_protocol": ("RC_PROTOCOL", "crsf"),
    "outer_loop": ("OUTER_LOOP", "mpc"),
    "pos_source": ("POS_SOURCE", "gps"),
    "gps_model": ("GPS_MODEL", None),  # default depends on role, see below
    # ANT2 populated? Independent of the receiver model: a UM982 is
    # heading-capable but can be wired single-antenna.
    "gps_dual_antenna": ("GPS_DUAL_ANTENNA", "no"),
    "role": ("ROLE", ""),
    "imu_rate": ("IMU_RATE", "8khz"),
    # Run the incremental (INDI) inner loop, or degrade to a plain
    # proportional rate controller? `yes` is the flying default.
    "indi": ("INDI", "yes"),
    # Compile the online BFGS trajectory optimizer, or fly only the
    # missions baked from `missions/*.yaml`? `no` is the default: the
    # online path costs ~110 KiB of solver `.bss` plus task-future state
    # that a runtime flag cannot remove, which is exactly why this is a
    # build knob and not a parameter.
    "plan_online": ("PLAN_ONLINE", "no"),
}

OUTER_FEATURE = {
    "mpc": "outer_mpc",
    # Full-model NMPC + INDI alpha inner loop; cargo feature implies
    # outer_mpc so the shared mission/sampler infrastructure compiles.
    "mpc_full": "outer_mpc_full",
    "cascade": "outer_geometric",
    "rate": "outer_rate",
}

# Estimator is env-only, never a `build:` key: only est_eskf exists today,
# so it is a dev escape hatch rather than a vehicle fact.
ESTIMATOR_DEFAULT = "eskf"


def warn(msg: str) -> None:
    print(f"vehicle_features: {msg}", file=sys.stderr)


def available_vehicles() -> list:
    """Every `vehicles/*.yaml` stem — the set of legal VEHICLE values."""
    vdir = os.path.join(REPO_ROOT, "vehicles")
    try:
        return sorted(f[:-5] for f in os.listdir(vdir) if f.endswith(".yaml"))
    except OSError:
        return []


def parse_build_section(path: str, strict: bool = False) -> dict:
    """Extract the flat build: section from a vehicle YAML, best-effort.

    `strict` is for the build path, where a missing file means the caller
    named a vehicle that does not exist: composing knob defaults for it
    would print a plausible feature list for the wrong firmware. `--list`
    keeps the lenient behaviour, since it globs real files and wants a
    row even for a vehicle with no `build:` section.
    """
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        # FileNotFoundError, PermissionError and IsADirectoryError are all
        # OSError, so this one branch covers "named a vehicle that isn't
        # there" and "can't read the one that is".
        if strict:
            warn(f"cannot read {path} ({e})")
            names = available_vehicles()
            if names:
                warn(f"available vehicles: {', '.join(names)}")
            warn("run `just vehicles` for details, or fix VEHICLE in .env")
            sys.exit(2)
        warn(f"cannot read {path} ({e}); using defaults/env only")
        return out
    in_build = False
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        if not in_build:
            if stripped == "build:":
                in_build = True
            continue
        # Section ends at the first non-indented content line.
        if not stripped.startswith("  "):
            break
        if ":" in stripped:
            k, v = stripped.strip().split(":", 1)
            v = v.strip().strip("\"'")
            if v:  # `key:` with no value = knob absent (matches serde None)
                out[k.strip()] = v
    return out


def parse_scalar_under(path: str, block: str, key: str) -> str:
    """Read one 2-space-indented scalar `key:` under top-level `block:`.

    `parse_build_section` can't be reused for this: `airframe:` also holds
    `motors:` with 4-space `- { ... }` list items, so the depth guard below
    is what keeps a list entry from being mistaken for a scalar. Returns
    "" when absent.
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return ""
    in_block = False
    for raw in lines:
        stripped = raw.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        if not in_block:
            if stripped == f"{block}:":
                in_block = True
            continue
        if not stripped.startswith("  "):
            break  # section ended
        if stripped.startswith("    "):
            continue  # deeper nesting (motors list items) — not our key
        k, _, v = stripped.strip().partition(":")
        if k.strip() == key:
            return v.strip().strip("\"'")
    return ""


def list_vehicles() -> None:
    """Table of every vehicle and the build facts that distinguish it.

    Lives here rather than in its own script because the three build:
    columns come from `parse_build_section` — this file's non-trivial,
    convention-encoding parser. A second copy would be the thing that goes
    stale the next time `build:` gains a knob.
    """
    vdir = os.path.join(REPO_ROOT, "vehicles")
    current = os.environ.get("VEHICLE") or ""
    hdr = ("", "vehicle", "board", "pos_source", "role", "outer_loop", "airframe.name")
    rows = []
    for name in available_vehicles():
        path = os.path.join(vdir, f"{name}.yaml")
        b = parse_build_section(path)  # lenient: no build: section -> {}
        rows.append(
            (
                "*" if name == current else "",
                name,
                b.get("board", "-"),
                b.get("pos_source", "-"),
                b.get("role", "-"),
                b.get("outer_loop", "-"),
                parse_scalar_under(path, "airframe", "name") or "<unset>",
            )
        )
    widths = [max(len(r[i]) for r in [hdr, *rows]) for i in range(len(hdr))]
    for r in [hdr, *rows]:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip())
    print(
        "\n* = current VEHICLE (.env / environment). "
        "Override per build: just build <vehicle>"
    )


def main() -> None:
    args = sys.argv[1:]
    if "--list" in args:
        list_vehicles()
        return
    # Unknown flags were silently ignored, which is the same class of bug
    # as the silent vehicle fallback being fixed here.
    unknown = [a for a in args if a != "--explain"]
    if unknown:
        warn(f"unknown argument(s) {unknown}; want --explain | --list")
        sys.exit(2)
    explain = "--explain" in args
    # Fallback vehicle MUST match build.rs's fallback (foxeer_bench, which
    # also matches the crate's default feature set) — two different
    # fallbacks would compose features from one vehicle and bake params
    # from another.
    vehicle = os.environ.get("VEHICLE") or "foxeer_bench"
    yaml_path = os.path.join(REPO_ROOT, "vehicles", f"{vehicle}.yaml")
    declared = parse_build_section(yaml_path, strict=True)

    resolved = {}
    provenance = {}
    for knob, (env_name, default) in KNOBS.items():
        env_v = os.environ.get(env_name)
        yaml_v = declared.get(knob)
        if env_v is not None and env_v != "":
            resolved[knob] = env_v
            provenance[knob] = f"env {env_name}"
            if yaml_v is not None and yaml_v != env_v:
                warn(
                    f"{knob}: env {env_name}={env_v!r} overrides "
                    f"{vehicle}.yaml build.{knob}={yaml_v!r}"
                )
        elif yaml_v is not None:
            resolved[knob] = yaml_v
            provenance[knob] = f"{vehicle}.yaml"
        else:
            resolved[knob] = default
            provenance[knob] = "default"

    # GPS model default depends on the resolved role (chaser carries the
    # UM982) — mirrors the old Justfile conditional default.
    if resolved["gps_model"] is None:
        resolved["gps_model"] = "unicore" if resolved["role"] == "chaser" else "ublox"
        provenance["gps_model"] = f"default (role={resolved['role'] or 'none'})"

    estimator = os.environ.get("ESTIMATOR") or ESTIMATOR_DEFAULT
    feats = [
        f"board_{resolved['board']}",
        f"rx_{resolved['rc_protocol']}",
        f"est_{estimator}",
        f"est_pos_{resolved['pos_source']}",
    ]
    outer = OUTER_FEATURE.get(resolved["outer_loop"])
    if outer is None:
        # Do NOT silently fall back to mpc. `outer_geometric` is the
        # literal cargo feature name while the accepted key is
        # `cascade`, so OUTER_LOOP=geometric is the natural typo — and
        # falling back produced a working *MPC* firmware that build.rs
        # then saw as perfectly consistent (YAML says mpc, features say
        # outer_mpc), leaving one stderr line as the only signal.
        # Emit a feature cargo will reject instead, matching how a bad
        # POS_SOURCE already fails.
        warn(
            f"outer_loop={resolved['outer_loop']!r} unknown "
            f"(want mpc|cascade|rate; note 'cascade' is spelled "
            f"outer_geometric as a cargo feature)"
        )
        outer = f"outer_INVALID_{resolved['outer_loop']}"
    feats.append(outer)
    # ublox is the driver default — only unicore is a cargo feature.
    # Skip it entirely on a non-GPS build: the GPS init sites are all
    # #[cfg(est_pos_gps)], so the feature only pulls a driver the build
    # cannot use and claims hardware that isn't there.
    if resolved["gps_model"] == "unicore" and resolved["pos_source"] == "gps":
        feats.append("gps_unicore")
    # ANT2 fitted. The vehicle-yaml loader rejects `yes` without
    # `gps_model: unicore`, so this cannot enable a heading the driver
    # never emits; the pos_source gate mirrors gps_unicore's.
    if resolved["gps_dual_antenna"] == "yes" and resolved["pos_source"] == "gps":
        feats.append("gps_dual_antenna")
    if os.environ.get("DEFMT_UART", "false") == "true":
        feats.append("defmt_uart")
    if resolved["role"] in ("leader", "chaser"):
        feats.append(f"role_{resolved['role']}")
    elif resolved["role"]:
        # Loud, for the same reason as outer_loop: silently dropping the
        # role produces a firmware with the peer-pose wiring absent.
        warn(f"role={resolved['role']!r} unknown (want leader|chaser|empty)")
        feats.append(f"role_INVALID_{resolved['role']}")
    # 8khz is the no-feature default — only 1khz is a cargo feature.
    if resolved["imu_rate"] == "1khz":
        feats.append("imu_1khz")
    elif resolved["imu_rate"] != "8khz":
        warn(f"imu_rate={resolved['imu_rate']!r} unknown (want 8khz|1khz)")
        feats.append(f"imu_INVALID_{resolved['imu_rate']}")
    # INDI on is the no-feature default — only turning it off is a feature.
    if resolved["indi"] == "no":
        feats.append("indi_off")
        if resolved["outer_loop"] == "mpc_full":
            # Allowed on purpose (it is the paper's "NMPC w/o INDI"
            # ablation / A-B baseline) but never a flight default: the
            # inner loop degrades to model-based static inversion with NO
            # disturbance rejection below the NMPC.
            warn(
                "outer_loop=mpc_full with indi=no: inner loop is STATIC "
                "INVERSION (no INDI increments) — experimental/bench "
                "configuration, not for free flight"
            )
    elif resolved["indi"] != "yes":
        warn(f"indi={resolved['indi']!r} unknown (want yes|no)")
        feats.append(f"indi_INVALID_{resolved['indi']}")
    # Offline-only planning is the no-feature default; the online solver
    # is what costs memory, so it is the one that names a feature.
    if resolved["plan_online"] == "yes":
        feats.append("plan_online")
    elif resolved["plan_online"] != "no":
        warn(f"plan_online={resolved['plan_online']!r} unknown (want yes|no)")
        feats.append(f"plan_online_INVALID_{resolved['plan_online']}")

    if explain:
        for knob in KNOBS:
            warn(f"  {knob:12s} = {resolved[knob]!r:14} ({provenance[knob]})")
        warn(f"  {'estimator':12s} = {estimator!r:14} (env-only)")
    print(",".join(feats))


if __name__ == "__main__":
    main()
