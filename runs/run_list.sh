#!/usr/bin/env bash
LIST_FILE=$1
OUT_DIR=$2
mkdir -p "$OUT_DIR"
while IFS= read -r pdf; do
  echo ">>> Processing $pdf"
  python extract_data.py "$pdf" "$OUT_DIR"
done < "$LIST_FILE"
