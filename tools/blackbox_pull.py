#!/usr/bin/env python3
"""Pull blackbox MCAP flight logs off the FC over the USB CDC shell.

Speaks the `blackbox get` wire protocol (see docs/blackbox.md):

    > blackbox get flight_0001.mcap
    OK <size>\r\n            <- or an error line, and no binary
    <size raw bytes>
    \r\nCRC <hex8>\r\n        <- CRC-32 of the raw bytes

Files land in --out (default logs/), written atomically (tmp + rename)
and only after the CRC-32 verifies. `--delete` removes each file from
the card (`blackbox rm`) after a *verified* pull.

Usage:
  tools/blackbox_pull.py                      # pull every *.mcap not already local
  tools/blackbox_pull.py flight_0003.mcap     # pull one file
  tools/blackbox_pull.py --list               # just show what's on the card
  tools/blackbox_pull.py --all --delete       # pull everything, then free the card

Requires pyserial (`pip install pyserial`).
"""

import argparse
import glob
import os
import re
import sys
import time
import zlib

# Matches the firmware's `blackbox ls` entry lines: "  /name  N bytes"
LS_ENTRY_RE = re.compile(r"^\s*/(\S+)\s+(\d+) bytes\s*$")
OK_RE = re.compile(rb"^OK (\d+)\s*$")
CRC_RE = re.compile(rb"^CRC ([0-9a-f]{8})\s*$")


def find_port() -> str:
    candidates = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/tty.usbmodem*"))
    if not candidates:
        sys.exit("no /dev/ttyACM* device found — is the FC plugged in? (use --port)")
    if len(candidates) > 1:
        print(f"note: multiple CDC devices, using {candidates[0]} (override with --port)")
    return candidates[0]


def open_port(port: str):
    try:
        import serial  # pyserial
    except ImportError:
        sys.exit("pyserial not installed: pip install pyserial")
    return serial.Serial(port, 115200, timeout=0.5)


def read_line(ser, deadline_s: float = 5.0) -> bytes:
    """Read one \\n-terminated line (returned without the terminator)."""
    deadline = time.monotonic() + deadline_s
    out = bytearray()
    while time.monotonic() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b == b"\n":
            return bytes(out).rstrip(b"\r")
        out += b
    raise TimeoutError(f"no line from FC within {deadline_s:.0f}s (got {bytes(out)!r})")


def command_response_line(ser, command: str, patterns, deadline_s: float = 12.0) -> re.Match:
    """Send a command and read lines until one matches any of `patterns`.

    Skips the local echo and any queued async stream lines; fails fast
    on the firmware's own error replies.
    """
    ser.reset_input_buffer()
    ser.write((command + "\n").encode())
    fail_prefixes = (
        command.split()[0].encode() + b" " + command.split()[1].encode() + b":",
        b"usage:",
        b"unknown command",
    )
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        line = read_line(ser, deadline - time.monotonic())
        for pat in patterns:
            m = pat.match(line)
            if m:
                return m
        stripped = line.strip()
        if any(stripped.startswith(p) for p in fail_prefixes):
            sys.exit(f"FC: {stripped.decode(errors='replace')}")
    raise TimeoutError(f"no reply to {command!r} within {deadline_s:.0f}s")


def list_files(ser) -> list[tuple[str, int]]:
    """`blackbox ls` → [(name, size)] for every FAT-root file."""
    ser.reset_input_buffer()
    ser.write(b"blackbox ls\n")
    entries: list[tuple[str, int]] = []
    header_seen = False
    # The op can take ~10 s on a cold mount; entries then stream fast.
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        try:
            line = read_line(ser, min(1.0 if header_seen else 12.0, deadline - time.monotonic()))
        except TimeoutError:
            if header_seen:
                break  # list is done, prompt has no newline
            raise
        text = line.decode(errors="replace")
        m = LS_ENTRY_RE.match(text)
        if m:
            entries.append((m.group(1), int(m.group(2))))
            continue
        if "blackbox ls:" in text:
            header_seen = True
            if "file(s)" not in text and "files" not in text:
                sys.exit(f"FC: {text.strip()}")
    return entries


def pull(ser, name: str, out_path: str, offset: int = 0) -> None:
    cmd = f"blackbox get {name}" if offset == 0 else f"blackbox get {name} {offset}"
    m = command_response_line(ser, cmd, [OK_RE])
    size = int(m.group(1))
    print(f"  /{name}: {size} bytes", flush=True)

    data = bytearray()
    t0 = time.monotonic()
    last_progress = t0
    last_report = t0
    while len(data) < size:
        chunk = ser.read(min(65536, size - len(data)))
        now = time.monotonic()
        if chunk:
            data.extend(chunk)
            last_progress = now
        elif now - last_progress > 5.0:
            sys.exit(f"  transfer stalled at {len(data)}/{size} bytes "
                     f"(resume with: blackbox get {name} {offset + len(data)})")
        if now - last_report > 1.0:
            rate = len(data) / max(now - t0, 1e-6) / 1024
            print(f"    {len(data)}/{size} bytes ({rate:.0f} KB/s)", flush=True)
            last_report = now

    # Trailer: blank line (the payload's "\r\n" separator) + CRC/ERR.
    while True:
        line = read_line(ser)
        if not line.strip():
            continue
        m = CRC_RE.match(line)
        if m:
            break
        if line.startswith(b"ERR"):
            sys.exit(f"  FC fault mid-transfer: {line.decode(errors='replace')}")
        # Anything else (queued stream line) — keep looking.

    want = int(m.group(1), 16)
    got = zlib.crc32(data) & 0xFFFFFFFF
    if got != want:
        sys.exit(f"  CRC mismatch on /{name}: FC says {want:08x}, host computed {got:08x}")

    rate = size / max(time.monotonic() - t0, 1e-6) / 1024
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out_path)
    print(f"  -> {out_path} (CRC ok, {rate:.0f} KB/s)")


def rm(ser, name: str) -> None:
    removed = re.compile(rb"^blackbox rm: removed /(\S+)\s*$")
    command_response_line(ser, f"blackbox rm {name}", [removed])
    print(f"  removed /{name} from the card")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*", help="file names on the card (default: every *.mcap)")
    ap.add_argument("--all", action="store_true",
                    help="pull every *.mcap on the card (default when no files named)")
    ap.add_argument("--list", action="store_true", help="list card contents and exit")
    ap.add_argument("--out", default="logs", help="output directory (default: logs/)")
    ap.add_argument("--port", default=None)
    ap.add_argument("--force", action="store_true", help="re-pull files that already exist locally")
    ap.add_argument("--delete", action="store_true",
                    help="blackbox rm each file after a CRC-verified pull")
    args = ap.parse_args()

    port = args.port or find_port()
    with open_port(port) as ser:
        if args.list or not args.files:
            print(f"listing card via {port} ...")
            entries = list_files(ser)
            if args.list:
                for name, size in entries:
                    print(f"  /{name}  {size} bytes")
                return
            targets = [n for n, _ in entries if n.lower().endswith(".mcap")]
            if not targets:
                print("no *.mcap files on the card.")
                return
        else:
            targets = args.files

        os.makedirs(args.out, exist_ok=True)
        pulled = 0
        for name in targets:
            out_path = os.path.join(args.out, os.path.basename(name))
            if os.path.exists(out_path) and not args.force:
                print(f"  /{name}: already at {out_path} (skip; --force to re-pull)")
            else:
                pull(ser, name, out_path)
                pulled += 1
            if args.delete:
                rm(ser, name)
        print(f"done: {pulled} file(s) pulled into {args.out}/")


if __name__ == "__main__":
    main()
