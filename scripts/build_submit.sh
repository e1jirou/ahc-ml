#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: scripts/build_submit.sh <entry.rs> [output.rs]" >&2
    exit 2
fi

entry_path="$1"
output_path="${2:-dist/submit.rs}"
bundle_path="${output_path%.rs}.bundle.rs"

mkdir -p target "$(dirname -- "$output_path")"

rustc --edition=2024 -O scripts/bundle_submit.rs -o target/bundle_submit
target/bundle_submit \
    --lib rust/ahc-ml/src/lib.rs \
    --crate-name ahc_ml \
    --max-bytes 524288 \
    "$entry_path" "$bundle_path"

rustc --edition=2024 -O scripts/minify_submit.rs -o target/minify_submit
target/minify_submit --keep-names "$bundle_path" "$output_path"

wc -c "$bundle_path" "$output_path"
