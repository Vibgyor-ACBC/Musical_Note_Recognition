#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <filename_without_extension>"
  exit 1
fi

MIC="plughw:0,0"
RATE=44100
CH=1

# absolute path of this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# go to ../query
OUT_DIR="$SCRIPT_DIR/../query"
mkdir -p "$OUT_DIR"

OUT="$OUT_DIR/$1.mp3"

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg not found. Installing..."
  sudo apt update && sudo apt install -y ffmpeg
}

echo "Recording to $OUT"
echo "Press Ctrl+C to stop"

arecord -D $MIC -f S16_LE -c $CH -r $RATE | \
ffmpeg -y -f s16le -ar $RATE -ac $CH -i - "$OUT"

echo "Saved: $OUT"
