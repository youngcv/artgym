#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hand> <object> [eval_consecutive.py args ...]" >&2
  echo "Example: $0 sharpa knife --asset-dir knife_30 --checkpoint runs/foo/best/model.pth --train artmanipSAPGPrivLSTMPPO --grasp-split train --episodes-per-grasp 2 --max-steps 1200 --deterministic --headless --randomize false --progress-interval-sec 1" >&2
  exit 1
fi

HAND="$1"
OBJECT="$2"
shift 2

ASSET_DIR=""
SAVE_SUCCESS_CYCLE_THRESHOLD=""
SAVE_SUCCESS_CYCLE_METRIC="mean"
IS_STUDENT_EVAL="false"
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset-dir)
      if [[ $# -lt 2 ]]; then
        echo "--asset-dir requires a value." >&2
        exit 1
      fi
      ASSET_DIR="$2"
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --asset-dir=*)
      ASSET_DIR="${1#--asset-dir=}"
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    --save-success-cycle-threshold)
      if [[ $# -lt 2 ]]; then
        echo "--save-success-cycle-threshold requires a value." >&2
        exit 1
      fi
      SAVE_SUCCESS_CYCLE_THRESHOLD="$2"
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --save-success-cycle-threshold=*)
      SAVE_SUCCESS_CYCLE_THRESHOLD="${1#--save-success-cycle-threshold=}"
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    --save-success-cycle-metric)
      if [[ $# -lt 2 ]]; then
        echo "--save-success-cycle-metric requires a value." >&2
        exit 1
      fi
      SAVE_SUCCESS_CYCLE_METRIC="$2"
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --save-success-cycle-metric=*)
      SAVE_SUCCESS_CYCLE_METRIC="${1#--save-success-cycle-metric=}"
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    --student-artifact)
      if [[ $# -lt 2 ]]; then
        echo "--student-artifact requires a value." >&2
        exit 1
      fi
      IS_STUDENT_EVAL="true"
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --student-artifact=*)
      IS_STUDENT_EVAL="true"
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

for arg in "${PASSTHROUGH_ARGS[@]}"; do
  if [[ "$arg" == "--instance-id" ]] || [[ "$arg" == --instance-id=* ]]; then
    echo "Do not pass --instance-id to $0. This script loops over all instances automatically." >&2
    exit 1
  fi
  if [[ "$arg" == "--hand" ]] || [[ "$arg" == --hand=* ]]; then
    echo "Do not pass --hand to $0. Use the first positional argument." >&2
    exit 1
  fi
  if [[ "$arg" == "--object" ]] || [[ "$arg" == --object=* ]]; then
    echo "Do not pass --object to $0. Use the second positional argument." >&2
    exit 1
  fi
done

OBJECT_ROOT="assets/objects/${ASSET_DIR:-$OBJECT}"
if [[ ! -d "${OBJECT_ROOT}" ]]; then
  echo "Object directory not found: ${OBJECT_ROOT}" >&2
  exit 1
fi

mapfile -t DIR_INSTANCE_IDS < <(
  find "${OBJECT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
)

if [[ ${#DIR_INSTANCE_IDS[@]} -eq 0 ]]; then
  echo "No instance directories found under ${OBJECT_ROOT}" >&2
  exit 1
fi

INSTANCE_IDS=("${DIR_INSTANCE_IDS[@]}")

for INSTANCE_ID in "${INSTANCE_IDS[@]}"; do
  echo "=== Consecutive evaluating ${OBJECT}/${INSTANCE_ID} ==="
  python -m isaacgymenvs.eval_consecutive \
    --hand "${HAND}" \
    --object "${OBJECT}" \
    --instance-id "${INSTANCE_ID}" \
    "${PASSTHROUGH_ARGS[@]}"
done

python3 - "${HAND}" "${OBJECT}" "${ASSET_DIR:-$OBJECT}" "${SAVE_SUCCESS_CYCLE_THRESHOLD}" "${SAVE_SUCCESS_CYCLE_METRIC}" "${OBJECT_ROOT}" "${IS_STUDENT_EVAL}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np


def safe_mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0

def safe_std(values):
    values = [float(v) for v in values]
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return float(var ** 0.5)


hand = sys.argv[1]
object_name = sys.argv[2]
asset_dir = sys.argv[3]
threshold_arg = sys.argv[4]
cycle_metric = sys.argv[5]
object_root = Path(sys.argv[6])
is_student_eval = sys.argv[7].lower() == "true"
summary_suffix = "_student" if is_student_eval else ""
asset_summary_path = Path("caches") / "initial_grasp" / hand / asset_dir / f"consecutive_eval_asset_summary{summary_suffix}.json"
success_cycle_threshold = float(threshold_arg) if threshold_arg != "" else None

instance_summaries = []
for instance_dir in sorted([p for p in object_root.iterdir() if p.is_dir()], key=lambda p: p.name):
    summary_path = Path("caches") / "initial_grasp" / hand / asset_dir / instance_dir.name / f"consecutive_eval_summary{summary_suffix}.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing per-instance summary after evaluation: {summary_path}")
    stats = json.loads(summary_path.read_text())
    instance_summaries.append(stats)

avg_ratio_metric_1 = safe_mean(item["ratio_grasps_max_cycles_gt_1"] for item in instance_summaries)
avg_metric_2 = safe_mean(item["mean_cycles"] for item in instance_summaries)
ratio_instances_metric_1_above_0 = safe_mean(
    1.0 if item["ratio_grasps_max_cycles_gt_1"] > 0.0 else 0.0
    for item in instance_summaries
)
if success_cycle_threshold is None:
    feasible_instance_num = None
    instance_coverage = None
    grasp_coverage = None
    consecutive_successful_cycles = None
    consecutive_successful_cycles_std = None
    consecutive_successful_max_cycles = None
    consecutive_successful_max_cycles_std = None
else:
    feasible_instance_num = int(sum(1 for item in instance_summaries if item["successful_grasp_num"] > 0))
    instance_coverage = safe_mean(
        1.0 if item["successful_grasp_num"] > 0 else 0.0
        for item in instance_summaries
    )
    grasp_coverage = safe_mean(item["grasp_coverage"] for item in instance_summaries)
    consecutive_successful_cycles = safe_mean(
        item["consecutive_successful_cycles"] for item in instance_summaries
    )
    consecutive_successful_cycles_std = safe_std(
        item["consecutive_successful_cycles"] for item in instance_summaries
    )
    consecutive_successful_max_cycles = safe_mean(
        item["consecutive_successful_max_cycles"] for item in instance_summaries
    )
    consecutive_successful_max_cycles_std = safe_std(
        item["consecutive_successful_max_cycles"] for item in instance_summaries
    )

output = {
    "hand": hand,
    "object": object_name,
    "asset_dir": asset_dir,
    "num_instances": len(instance_summaries),
    "success_cycle_threshold": success_cycle_threshold,
    "success_cycle_metric": cycle_metric,
    "feasible_instance_num": feasible_instance_num,
    "total_instance_num": len(instance_summaries),
    "instance_coverage": instance_coverage,
    "grasp_coverage": grasp_coverage,
    "consecutive_successful_cycles": consecutive_successful_cycles,
    "consecutive_successful_cycles_std": consecutive_successful_cycles_std,
    "consecutive_successful_max_cycles": consecutive_successful_max_cycles,
    "consecutive_successful_max_cycles_std": consecutive_successful_max_cycles_std,
    "average_ratio_grasps_max_cycles_gt_1": avg_ratio_metric_1,
    "average_mean_cycles": avg_metric_2,
    "ratio_instances_with_ratio_grasps_max_cycles_gt_1_above_0": ratio_instances_metric_1_above_0,
    "instances": instance_summaries,
}

asset_summary_path.parent.mkdir(parents=True, exist_ok=True)
asset_summary_path.write_text(json.dumps(output, indent=2))
print(f"Saved asset consecutive eval summary to {asset_summary_path}")
PY
