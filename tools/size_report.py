#!/usr/bin/env python3
"""Static RAM / flash report for a firmware ELF.

Prints the allocatable section sizes, the largest `.bss`/`.data` symbols,
and the headroom between the end of the statics and the initial stack
pointer. The STM32H743 linker script places `.data`, `.bss`, `.uninit`
AND the main stack in the single 512 KiB AXI SRAM region, so headroom is
the entire budget for stack growth — the number that matters when a
build boot-loops (crates/cybflight/src/blackbox/sdmmc_block.rs).

Usage: size_report.py ELF [--llvm-bin DIR] [--top N] [--memory-x FILE]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def ram_region(memory_x: Path):
    """(origin, length) of the RAM region from memory.x."""
    text = memory_x.read_text()
    m = re.search(r"RAM\s*:\s*ORIGIN\s*=\s*(0x[0-9A-Fa-f]+)\s*,\s*LENGTH\s*=\s*(\d+)K", text)
    if not m:
        sys.exit(f"cannot find RAM region in {memory_x}")
    return int(m.group(1), 16), int(m.group(2)) * 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("elf")
    ap.add_argument("--llvm-bin", default="", help="directory holding llvm-size / llvm-nm")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--memory-x", default="crates/cybflight/memory.x")
    a = ap.parse_args()

    size = str(Path(a.llvm_bin) / "llvm-size") if a.llvm_bin else "llvm-size"
    nm = str(Path(a.llvm_bin) / "llvm-nm") if a.llvm_bin else "llvm-nm"

    # ---- sections ----
    sections = {}
    for line in run([size, "-A", a.elf]).splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].startswith("."):
            try:
                sections[parts[0]] = (int(parts[1]), int(parts[2]))
            except ValueError:
                pass

    ram_origin, ram_len = ram_region(Path(a.memory_x))
    flash = sum(sz for name, (sz, addr) in sections.items() if name in (".vector_table", ".text", ".rodata", ".data"))
    ram_secs = [(n, sz, addr) for n, (sz, addr) in sections.items() if ram_origin <= addr < ram_origin + ram_len and sz > 0]
    ram_end = max(addr + sz for _, sz, addr in ram_secs)
    static_ram = ram_end - ram_origin
    headroom = ram_origin + ram_len - ram_end

    print(f"{a.elf}")
    print(f"  flash image      {flash:>9,} B   ({flash / 1024:7.1f} KiB)")
    for n, sz, _ in sorted(ram_secs, key=lambda t: t[2]):
        print(f"  {n:<16} {sz:>9,} B   ({sz / 1024:7.1f} KiB)")
    print(f"  static RAM       {static_ram:>9,} B   ({static_ram / 1024:7.1f} KiB)  = {100 * static_ram / ram_len:.1f} % of {ram_len // 1024} KiB")
    print(f"  headroom         {headroom:>9,} B   ({headroom / 1024:7.1f} KiB)  end of statics 0x{ram_end:08X} -> stack top 0x{ram_origin + ram_len:08X}")

    # ---- symbols ----
    rows = []
    for line in run([nm, "--print-size", "--size-sort", "--demangle", a.elf]).splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[2].lower() in ("b", "d"):
            rows.append((int(parts[1], 16), parts[2].lower(), parts[3]))
    rows.sort(reverse=True)
    print(f"\n  largest .bss (b) / .data (d) symbols:")
    for sz, kind, name in rows[: a.top]:
        print(f"  {sz:>9,} B  {kind}  {name}")


if __name__ == "__main__":
    main()
