#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# unzip tiny-imagenet-200.zip -d ./
if [ ! -d ../data ]; then
  mkdir ../data
else
  echo "../data dir exists"
fi

mv ./tiny-imagenet-200 ../data/
echo "move tiny-imagenet-200 dir to ../data/tiny-imagenet-200"
python tinyimagenet_reformat.py

