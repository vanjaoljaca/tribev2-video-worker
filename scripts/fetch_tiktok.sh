#!/usr/bin/env bash
set -euo pipefail

COUNT="${1:-12}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/raw"
NOTES_DIR="$ROOT/notes"
ACCOUNT_URL="${ACCOUNT_URL:-https://www.tiktok.com/@some_account}"

mkdir -p "$RAW_DIR" "$NOTES_DIR"

yt-dlp \
  --flat-playlist \
  --dump-json \
  --playlist-end "$COUNT" \
  "$ACCOUNT_URL" > "$NOTES_DIR/tiktok-metadata.ndjson"

yt-dlp \
  --playlist-end "$COUNT" \
  --merge-output-format mp4 \
  --write-info-json \
  --no-overwrites \
  -o "$RAW_DIR/%(upload_date)s_%(id)s.%(ext)s" \
  "$ACCOUNT_URL"

python3 "$ROOT/scripts/metadata_manifest.py" "$NOTES_DIR/tiktok-metadata.ndjson" "$NOTES_DIR/tiktok-manifest.tsv"

echo "Wrote metadata: $NOTES_DIR/tiktok-metadata.ndjson"
echo "Wrote manifest: $NOTES_DIR/tiktok-manifest.tsv"
echo "Videos are in: $RAW_DIR"
