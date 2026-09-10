#!/usr/bin/env bash
# Setup Kaggle credentials for this project
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/.kaggle" "$HOME/.kaggle"
SRC="${1:-$HOME/Downloads/kaggle (1).json}"
if [[ ! -f "$SRC" ]]; then
  SRC="$HOME/Downloads/kaggle.json"
fi
cp "$SRC" "$ROOT/.kaggle/kaggle.json"
cp "$SRC" "$HOME/.kaggle/kaggle.json"
chmod 600 "$ROOT/.kaggle/kaggle.json" "$HOME/.kaggle/kaggle.json"
echo "Kaggle credentials installed to $ROOT/.kaggle and $HOME/.kaggle"
