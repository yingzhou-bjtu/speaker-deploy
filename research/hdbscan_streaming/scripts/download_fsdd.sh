#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${RAW_ROOT:-$ROOT/data/fsdd/raw}"
UPSTREAM_COMMIT="26eb9aaf76e81b692f806f9140c2d2777410d7a1"
ARCHIVE="$RAW_ROOT/free-spoken-digit-dataset-${UPSTREAM_COMMIT}.tar.gz"
PART="$ARCHIVE.part"
EXTRACTED="$RAW_ROOT/free-spoken-digit-dataset-${UPSTREAM_COMMIT}"
URL="https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/tar.gz/${UPSTREAM_COMMIT}"
EXPECTED_SHA256="9a68686ad29274bd0affde81b244953e677c9d7048dfb8acaf1168d7758bafab"

mkdir -p "$RAW_ROOT"

if [[ -f "$ARCHIVE" ]]; then
  actual_sha256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
    printf 'archive checksum mismatch: %s\n' "$ARCHIVE" >&2
    printf 'expected: %s\nactual:   %s\n' "$EXPECTED_SHA256" "$actual_sha256" >&2
    exit 1
  fi
else
  wget --continue --tries=5 --timeout=30 --output-document="$PART" "$URL"
  actual_sha256="$(sha256sum "$PART" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
    printf 'download checksum mismatch: %s\n' "$PART" >&2
    printf 'expected: %s\nactual:   %s\n' "$EXPECTED_SHA256" "$actual_sha256" >&2
    exit 1
  fi
  mv "$PART" "$ARCHIVE"
fi

if [[ ! -d "$EXTRACTED" ]]; then
  tar -xzf "$ARCHIVE" -C "$RAW_ROOT"
fi

RECORDINGS="$EXTRACTED/recordings"
test -d "$RECORDINGS"
wav_count="$(find "$RECORDINGS" -type f -name '*.wav' | wc -l)"
test "$wav_count" -eq 3000

cat > "$ROOT/data/fsdd/PROVENANCE.md" <<EOF
# FSDD local dataset

- Source repository: https://github.com/Jakobovski/free-spoken-digit-dataset
- Pinned commit: $UPSTREAM_COMMIT
- Download URL: $URL
- Archive: data/fsdd/raw/free-spoken-digit-dataset-${UPSTREAM_COMMIT}.tar.gz
- Archive SHA256: $EXPECTED_SHA256
- Verified WAV count: $wav_count
- License: CC BY-SA 4.0, as declared by the upstream repository

The raw archive and extracted files are local inputs and are not committed.
The public benchmark uses committed derived embedding fixtures. The generated
manifest and embedding preparation path are separate from the release
reproduction command.
EOF

printf 'FSDD_READY wav_files=%s\n' "$wav_count"
printf 'FSDD_RECORDINGS=%s\n' "$RECORDINGS"
