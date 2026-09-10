import json
from pathlib import Path

import numpy as np


def _stable_descending_indices(values):
    return [int(idx) for idx in sorted(range(len(values)), key=lambda idx: (-float(values[idx]), int(idx)))]


def _majority_vote_codes(code_trials):
    code_trials = np.asarray(code_trials, dtype=np.int64)
    if code_trials.ndim != 2:
        raise ValueError(f"Expected code_trials to be rank-2, got shape {code_trials.shape}")

    num_grasps = code_trials.shape[1]
    majority_codes = []
    for grasp_idx in range(num_grasps):
        values, counts = np.unique(code_trials[:, grasp_idx], return_counts=True)
        best_order = np.lexsort((values, -counts))
        majority_codes.append(int(values[best_order[0]]))
    return majority_codes


def aggregate_consecutive_stats(trial_stats_list):
    if not trial_stats_list:
        raise ValueError("trial_stats_list must contain at least one trial.")

    if len(trial_stats_list) == 1:
        stats = dict(trial_stats_list[0])
        cycles = np.asarray(stats["consecutive_success_cycles"], dtype=np.float32)
        sorted_indices = _stable_descending_indices(cycles.tolist())
        stats["episodes_per_grasp"] = 1
        stats["average_consecutive_success_cycles"] = float(cycles.mean()) if cycles.size > 0 else 0.0
        stats["consecutive_success_cycles_mean"] = cycles.tolist()
        stats["consecutive_success_cycles_std"] = np.zeros_like(cycles).tolist()
        stats["consecutive_success_cycles_trials"] = [stats["consecutive_success_cycles"]]
        stats["completion_goal_distance_mean"] = list(stats["completion_goal_distance"])
        stats["completion_goal_distance_trials"] = [stats["completion_goal_distance"]]
        stats["completion_reason_code_trials"] = [stats["completion_reason_code"]]
        stats["completion_stage_code_trials"] = [stats["completion_stage_code"]]
        stats["sorted_grasp_indices_by_mean_cycles"] = sorted_indices
        stats["best_grasp_index"] = int(sorted_indices[0]) if sorted_indices else -1
        stats["best_consecutive_success_cycles"] = float(cycles[sorted_indices[0]]) if sorted_indices else 0.0
        return stats

    reason_map = {
        0: "unfinished",
        1: "goal_timeout",
        2: "fall",
        3: "invalid",
        5: "reset_without_reason",
        6: "episode_timeout",
    }
    stage_map = {
        -1: "none",
        0: "open",
        1: "close",
    }

    cycles_trials = np.asarray(
        [trial_stats["consecutive_success_cycles"] for trial_stats in trial_stats_list],
        dtype=np.float32,
    )
    goal_distance_trials = np.asarray(
        [trial_stats["completion_goal_distance"] for trial_stats in trial_stats_list],
        dtype=np.float32,
    )
    reason_code_trials = np.asarray(
        [trial_stats["completion_reason_code"] for trial_stats in trial_stats_list],
        dtype=np.int64,
    )
    stage_code_trials = np.asarray(
        [trial_stats["completion_stage_code"] for trial_stats in trial_stats_list],
        dtype=np.int64,
    )

    mean_cycles = cycles_trials.mean(axis=0)
    std_cycles = cycles_trials.std(axis=0)
    mean_goal_distance = goal_distance_trials.mean(axis=0)
    majority_reason_codes = _majority_vote_codes(reason_code_trials)
    majority_stage_codes = _majority_vote_codes(stage_code_trials)
    sorted_indices = _stable_descending_indices(mean_cycles.tolist())
    best_grasp_index = int(sorted_indices[0]) if sorted_indices else -1

    aggregated = dict(trial_stats_list[0])
    aggregated["episodes_per_grasp"] = int(len(trial_stats_list))
    aggregated["average_consecutive_success_cycles"] = float(mean_cycles.mean()) if mean_cycles.size > 0 else 0.0
    aggregated["consecutive_success_cycles_mean"] = mean_cycles.tolist()
    aggregated["consecutive_success_cycles_std"] = std_cycles.tolist()
    aggregated["consecutive_success_cycles_trials"] = cycles_trials.tolist()
    aggregated["completion_goal_distance_mean"] = mean_goal_distance.tolist()
    aggregated["completion_goal_distance_trials"] = goal_distance_trials.tolist()
    aggregated["completion_reason_code"] = majority_reason_codes
    aggregated["completion_reason"] = [reason_map.get(int(code), "unknown") for code in majority_reason_codes]
    aggregated["completion_reason_code_trials"] = reason_code_trials.tolist()
    aggregated["completion_stage_code"] = majority_stage_codes
    aggregated["completion_stage"] = [stage_map.get(int(code), "unknown") for code in majority_stage_codes]
    aggregated["completion_stage_code_trials"] = stage_code_trials.tolist()
    aggregated["sorted_grasp_indices_by_mean_cycles"] = sorted_indices
    aggregated["best_grasp_index"] = best_grasp_index
    aggregated["best_consecutive_success_cycles"] = float(mean_cycles[best_grasp_index]) if best_grasp_index >= 0 else 0.0
    return aggregated


def compute_grasp_consecutive_metric_means(stats):
    cycles = np.asarray(
        stats.get("consecutive_success_cycles_mean", stats.get("consecutive_success_cycles", [])),
        dtype=np.float32,
    )
    goal_distance = np.asarray(
        stats.get("completion_goal_distance_mean", stats.get("completion_goal_distance", [])),
        dtype=np.float32,
    )

    def _safe_mean(values):
        return float(values.mean()) if values.size > 0 else 0.0

    return {
        "eval/average_consecutive_success_cycles": float(stats.get("average_consecutive_success_cycles", _safe_mean(cycles))),
        "eval/best_consecutive_success_cycles": float(stats.get("best_consecutive_success_cycles", 0.0)),
        "eval/mean_completion_goal_distance": _safe_mean(goal_distance),
    }


def print_grasp_consecutive_summary(stats, output_path=""):
    metrics = compute_grasp_consecutive_metric_means(stats)
    print(f"average consecutive success cycles: {metrics['eval/average_consecutive_success_cycles']:.4f}")
    print(f"best consecutive success cycles: {metrics['eval/best_consecutive_success_cycles']:.4f}")
    print(f"average completion goal distance: {metrics['eval/mean_completion_goal_distance']:.6f}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(stats, indent=2))
        print(f"Saved stats to {output_path}")
