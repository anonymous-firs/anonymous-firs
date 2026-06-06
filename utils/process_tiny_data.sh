#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# unzip tiny-imagenet-200.zip -d .
data_dir="../data"
if [ ! -d "$data_dir" ]; then
  mkdir "$data_dir"
else
  echo "$data_dir dir exists"
fi

mv ./tiny-imagenet-200 "$data_dir"
echo "move tiny-imagenet-200 dir to $data_dir"
python tinyimagenet_reformat.py

