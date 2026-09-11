#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/fsdd/raw"
ARCHIVE="$RAW/free-spoken-digit-dataset-master.tar.gz"
URL="https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/tar.gz/refs/heads/master"

mkdir -p "$RAW"
if [[ ! -s "$ARCHIVE" ]]; then
  wget --continue --tries=5 --timeout=30 -O "$ARCHIVE" "$URL"
fi

if [[ ! -d "$RAW/free-spoken-digit-dataset-master" ]]; then
  tar -xzf "$ARCHIVE" -C "$RAW"
fi

RECORDINGS="$RAW/free-spoken-digit-dataset-master/recordings"
test -d "$RECORDINGS"
count="$(find "$RECORDINGS" -type f -name '*.wav' | wc -l)"
test "$count" -eq 3000

sha256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
cat > "$ROOT/data/fsdd/PROVENANCE.md" <<EOF
# FSDD local dataset

- Source repository: https://github.com/Jakobovski/free-spoken-digit-dataset
- Download URL: $URL
- Archive: $ARCHIVE
- Archive SHA256: $sha256
- Verified WAV count: $count
- License: CC BY-SA 4.0, as declared by the upstream repository

The raw archive and extracted files are retained locally. The benchmark only
uses a deterministic subset described by the generated manifest.
EOF

printf 'FSDD ready: %s WAV files\\n' "$count"
printf 'Recordings: %s\\n' "$RECORDINGS"
