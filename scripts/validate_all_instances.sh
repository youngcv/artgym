#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hand> <object> [--unique] [valid_grasp args ...]" >&2
  echo "Example: $0 sharpa knife --asset-dir knife_multi --pipeline cpu --num-envs 500 --episode-length 30 --rot-threshold 0.1 --pos-threshold 0.01 --headless --camera" >&2
  echo "Example (unique): $0 sharpa knife --unique --asset-dir knife_multi --pipeline cpu --num-envs 500 --episode-length 30 --unique-pos-threshold 0.005 --unique-rot-threshold 0.05 --headless" >&2
  exit 1
fi

HAND="$1"
OBJECT="$2"
shift 2

FILTERED_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--unique" ]]; then
    FILTERED_ARGS+=("$arg")
    continue
  fi
  FILTERED_ARGS+=("$arg")
done
set -- "${FILTERED_ARGS[@]}"

for arg in "$@"; do
  if [[ "$arg" == "--instance-id" ]] || [[ "$arg" == --instance-id=* ]]; then
    echo "Do not pass --instance-id to $0. This script loops over all instances automatically." >&2
    exit 1
  fi
done

ASSET_DIR="${OBJECT}"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; ++i)); do
  if [[ "${ARGS[$i]}" == "--asset-dir" ]]; then
    if (( i + 1 >= ${#ARGS[@]} )); then
      echo "--asset-dir requires a value." >&2
      exit 1
    fi
    ASSET_DIR="${ARGS[$((i + 1))]}"
    break
  fi
  if [[ "${ARGS[$i]}" == --asset-dir=* ]]; then
    ASSET_DIR="${ARGS[$i]#--asset-dir=}"
    break
  fi
done

OBJECT_ROOT="assets/objects/${ASSET_DIR}"

if [[ ! -d "${OBJECT_ROOT}" ]]; then
  echo "Object directory not found: ${OBJECT_ROOT}" >&2
  exit 1
fi

mapfile -t INSTANCE_IDS < <(
  find "${OBJECT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
)

if [[ ${#INSTANCE_IDS[@]} -eq 0 ]]; then
  echo "No instance directories found under ${OBJECT_ROOT}" >&2
  exit 1
fi

for INSTANCE_ID in "${INSTANCE_IDS[@]}"; do
  echo "=== Validating ${OBJECT}/${INSTANCE_ID} ==="
  python -m isaacgymenvs.valid_grasp \
    --hand "${HAND}" \
    --object "${OBJECT}" \
    --instance-id "${INSTANCE_ID}" \
    "$@"
done

SUMMARY_PATH="caches/initial_grasp/${HAND}/${ASSET_DIR}/valid_num_summary.jsonl"
python - "$SUMMARY_PATH" "caches/initial_grasp/${HAND}/${ASSET_DIR}" <<'PY'
import json
import os
import sys

summary_path = sys.argv[1]
asset_root = sys.argv[2]

rows = []
total = 0
for instance_id in sorted(
    name for name in os.listdir(asset_root)
    if os.path.isdir(os.path.join(asset_root, name))
):
    valid_num_path = os.path.join(asset_root, instance_id, "valid_num.jsonl")
    value = None
    if os.path.exists(valid_num_path):
        with open(valid_num_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if instance_id in data:
                    value = int(data[instance_id])
    rows.append({"instance_id": instance_id, "valid_num": value})
    if value is not None:
        total += value

with open(summary_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
    f.write(json.dumps({"instance_id": "TOTAL", "valid_num": total}) + "\n")

print(f"Saved valid_num summary to {os.path.abspath(summary_path)}")
PY
