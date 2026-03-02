#!/usr/bin/env bash
set -euo pipefail

ELF="$1"

# Output alongside the ELF, same name but .bin
BIN="${ELF%.*}.bin"

# Require rust-objcopy (cargo-binutils + llvm-tools-preview)
command -v rust-objcopy >/dev/null 2>&1 || {
  echo "rust-objcopy not found."
  exit 1
}

rust-objcopy --output-target=binary "$ELF" "$BIN"
echo "Converted: $ELF -> $BIN"