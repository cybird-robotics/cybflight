#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
host_target=$(rustc -vV | sed -n 's/^host: //p')
exec cargo run --quiet --locked -p vehicle-yaml --bin vehicle-features \
    --target "$host_target" -- "$@"
