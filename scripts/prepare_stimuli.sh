#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/raw"
STIMULI_DIR="$ROOT/stimuli"

mkdir -p "$STIMULI_DIR"

shopt -s nullglob
for video in "$RAW_DIR"/*.mp4 "$RAW_DIR"/*.mov "$RAW_DIR"/*.m4v; do
  base="$(basename "$video")"
  stem="${base%.*}"
  out_mp4="$STIMULI_DIR/${stem}.mp4"
  out_wav="$STIMULI_DIR/${stem}.wav"

  ffmpeg -hide_banner -loglevel error -y \
    -i "$video" \
    -vf "scale=640:-2,fps=30,format=yuv420p" \
    -c:v libx264 -preset veryfast -crf 20 \
    -c:a aac -ar 48000 -ac 2 -b:a 128k \
    "$out_mp4"

  ffmpeg -hide_banner -loglevel error -y \
    -i "$video" \
    -vn -ar 16000 -ac 1 \
    "$out_wav"

  echo "$out_mp4"
done

python3 "$ROOT/scripts/stimuli_manifest.py" "$STIMULI_DIR" "$ROOT/notes/stimuli-manifest.tsv"
echo "Wrote stimuli manifest: $ROOT/notes/stimuli-manifest.tsv"

