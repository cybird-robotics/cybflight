#!/usr/bin/env python3
"""Emit one comparison-table row from a cybflight-sim JSON report.

Reads env vars SCEN, CTRL, REPORT (set by the `just sim-compare` recipe).
"""
import json
import os
import sys


def main() -> int:
    scen = os.environ["SCEN"]
    ctrl = os.environ["CTRL"]
    with open(os.environ["REPORT"]) as f:
        r = json.load(f)["summary"]
    print(
        f"{scen:<22} {ctrl:<11} "
        f"{r['rms_pos_err_m']:10.4f} "
        f"{r['peak_pos_err_m']:10.4f} "
        f"{r['terminal_pos_err_m']:10.4f} "
        f"{r['peak_tilt_rad'] * 57.2958:8.1f} "
        f"{r['peak_motor_saturation_pct']:7.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
