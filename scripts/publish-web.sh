#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="$ROOT_DIR/web"
DOCS_DIR="$ROOT_DIR/docs"

if ! command -v bun >/dev/null 2>&1; then
  echo "Error: bun is not installed or not on PATH." >&2
  exit 1
fi

mkdir -p "$DOCS_DIR/pkg"

(
  cd "$WEB_DIR"
  bun run build
)

rsync -a "$WEB_DIR/index.html" "$WEB_DIR/main.js" "$DOCS_DIR/"
rsync -a --delete "$WEB_DIR/pkg/" "$DOCS_DIR/pkg/"

BEST_SRC=$(ls -t "$ROOT_DIR"/checkpoints*/best.safetensors 2>/dev/null | head -n 1 || true)
if [[ -n "$BEST_SRC" ]]; then
  cp -f "$BEST_SRC" "$DOCS_DIR/best.safetensors"
else
  echo "Warning: no best.safetensors found under checkpoints*/" >&2
fi

echo "Published web assets to $DOCS_DIR"
