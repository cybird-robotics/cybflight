#!/usr/bin/env bash
set -euo pipefail

ELF="$1"

# Output alongside the ELF, same name but .bin
BIN="${ELF%.*}.bin"

# Find llvm-objcopy from the rustup toolchain (provided by llvm-tools-preview)
OBJCOPY="$(rustc --print target-libdir)/../bin/llvm-objcopy"
if [ ! -x "$OBJCOPY" ]; then
  echo "llvm-objcopy not found. Run: rustup component add llvm-tools-preview"
  exit 1
fi

"$OBJCOPY" --output-target=binary "$ELF" "$BIN"
echo "Converted: $ELF -> $BIN"