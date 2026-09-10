import argparse
import json
from pathlib import Path
import shutil
from typing import List, Optional

import isaacgym  # noqa: F401
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import open_dict

from isaacgymenvs.consecutive_eval_utils import aggregate_consecutive_stats
from isaacgymenvs.eval_common import _infer_expl_num_blocks, preprocess_train_config
from isaacgymenvs.grasp_split_utils import GRASP_SPLIT_CHOICES, normalize_grasp_split, resolve_grasp_cache_path
from isaacgymenvs.student_eval_utils import run_grasp_evaluation_loop
from isaacgymenvs.utils.student_runtime_utils import build_student_cfg_overrides_from_meta, resolve_teacher_identity_from_meta
from isaacgymenvs.utils.distributed_runtime import force_single_process_env
from isaacgymenvs.utils.player_utils import parse_bool_arg


def _save_last_episode_actions(
    *,
    save_path: Path,
    actions: torch.Tensor,
    args,
    cfg,
    asset_dir_name: str,
    grasp_split: str,
    goal_sequence,
    stage_duration_secs,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actions": actions.detach().cpu(),
        "metadata": {
            "task": str(args.task),
            "train": str(args.train) if args.train is not None else str(cfg.train.get("name", "")),
            "hand": str(args.hand),
            "object": str(args.object),
            "asset_dir": str(asset_dir_name),
            "instance_id": str(args.instance_id),
            "grasp_split": str(grasp_split),
            "goal_sequence": [float(goal_sequence[0]), float(goal_sequence[1])],
            "stage_duration_secs": None if stage_duration_secs is None else float(stage_duration_secs),
            "goal_switch_interval_secs": None
            if args.goal_switch_interval_secs is None
            else float(args.goal_switch_interval_secs),
            "max_steps": int(args.max_steps),
            "randomize": bool(args.randomize),
            "seed": int(args.seed),
            "torch_deterministic": None
            if args.torch_deterministic is None
            else bool(args.torch_deterministic),
            "sim_device": str(args.sim_device),
            "rl_device": str(args.rl_device),
            "graphics_device_id": int(args.graphics_device_id),
            "num_actions": int(actions.shape[1]),
        },
    }
    torch.save(payload, save_path)
    print(
        f"Saved last trial action sequence with {int(actions.shape[0])} steps to {save_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate each grasp once and count how many full open-close success cycles it can complete "
            "before the streak breaks."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the policy checkpoint.")
    parser.add_argument("--student-artifact", default="", help="Optional distilled student artifact path.")
    parser.add_argument("--task", default="artmanip", help="Task config name.")
    parser.add_argument("--train", default=None, help="Train config name. Defaults to the config default.")
    parser.add_argument("--hand", default="sharpa", help="Hand config name.")
    parser.add_argument("--object", default="knife", help="Object config name.")
    parser.add_argument("--asset-dir", default="", help="Optional asset directory under assets/objects, e.g. knife_multi.")
    parser.add_argument("--instance-id", required=True, help="Single object instance id to evaluate.")
    parser.add_argument(
        "--grasp-split",
        choices=GRASP_SPLIT_CHOICES,
        default="test",
        help="Which grasp split to evaluate.",
    )
    parser.add_argument(
        "--episodes-per-grasp",
        type=int,
        default=1,
        help="How many independent consecutive-eval trials to run for each grasp before aggregating results.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1200,
        help="Maximum number of environment steps per trial. This now controls the consecutive rollout budget.",
    )
    parser.add_argument("--open-goal", type=float, default=None, help="Optional override for the first-stage goal value.")
    parser.add_argument("--close-goal", type=float, default=None, help="Optional override for the second-stage goal value.")
    parser.add_argument(
        "--goal-switch-interval-secs",
        type=float,
        default=None,
        help="Optional success-hold duration required before switching goals in consecutive eval mode.",
    )
    parser.add_argument("--stage-duration-secs", type=float, default=None, help="Per-stage timeout in seconds.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument("--save-video", default="", help="Optional video path (.mp4/.gif) recorded from artmanip.yaml camera.")
    parser.add_argument("--video-env-index", type=int, default=0, help="Which env camera to record when --save-video is used.")
    parser.add_argument("--video-fps", type=int, default=30, help="Output video fps when --save-video is used.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions.")
    parser.add_argument("--sim-device", default="cuda:0", help="Simulation device.")
    parser.add_argument("--rl-device", default="cuda:0", help="RL device.")
    parser.add_argument("--graphics-device-id", type=int, default=0, help="Graphics device id.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--torch-deterministic",
        type=parse_bool_arg,
        default=None,
        help="Optional override for cfg.torch_deterministic.",
    )
    parser.add_argument("--expl-block-idx", type=int, default=0, help="Teacher exploration block id when applicable.")
    parser.add_argument(
        "--save-success-cycle-threshold",
        type=float,
        default=None,
        help="If set, save grasps whose selected consecutive-cycle metric is strictly greater than this threshold.",
    )
    parser.add_argument(
        "--save-success-cycle-metric",
        choices=("mean", "max"),
        default="mean",
        help="Which per-grasp cycle metric to threshold when saving success grasps.",
    )
    parser.add_argument(
        "--save-success-output-split",
        default="success_consecutive",
        help="Output split folder name for saved consecutive-success grasps.",
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=5.0,
        help="How often to print progress in seconds. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--randomize",
        type=parse_bool_arg,
        default=False,
        help="Whether to enable task.randomize during evaluation.",
    )
    parser.add_argument(
        "--save-last-episode-actions",
        default="",
        help="Optional .pt path to save the last trial's action sequence plus replay metadata.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many top grasps to print in the summary.")
    return parser.parse_args()


def _resolve_effective_asset_root(cfg, asset_dir: str) -> Path:
    if asset_dir:
        return Path("assets/objects") / asset_dir
    return Path(str(cfg.object.asset.asset_root))


def _resolve_source_paths(instance_dir: Path, grasp_split: str):
    split = normalize_grasp_split(grasp_split)
    grasp_path = resolve_grasp_cache_path(instance_dir, split)
    if split == "valid":
        return grasp_path, instance_dir / "valid" / "grasp_visualization"
    if split == "train":
        return grasp_path, instance_dir / "train" / "grasp_visualization"
    if split == "test":
        return grasp_path, instance_dir / "test" / "grasp_visualization"
    return grasp_path, None


def _save_visualizations(source_vis_dir: Optional[Path], target_vis_dir: Path, selected_indices: np.ndarray) -> int:
    if source_vis_dir is None or not source_vis_dir.exists():
        return 0

    target_vis_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for new_index, source_index in enumerate(selected_indices.tolist()):
        source_png = source_vis_dir / f"{source_index:05d}.png"
        if not source_png.exists():
            continue
        target_png = target_vis_dir / f"{new_index:05d}.png"
        shutil.copy2(source_png, target_png)
        copied += 1
    return copied


def _save_consecutive_success_grasps(
    *,
    stats: dict,
    threshold: float,
    cycle_metric: str,
    output_split: str,
    hand: str,
    object_name: str,
    asset_dir_name: str,
):
    instance_id = str(stats["instance_id"])
    grasp_split = normalize_grasp_split(str(stats["grasp_split"]))
    repo_root = Path(__file__).resolve().parent.parent
    instance_dir = repo_root / "caches" / "initial_grasp" / hand / asset_dir_name / instance_id
    if not instance_dir.exists():
        raise FileNotFoundError(f"Instance cache directory not found: {instance_dir}")

    source_grasps_path, source_vis_dir = _resolve_source_paths(instance_dir, grasp_split)
    if not source_grasps_path.exists():
        raise FileNotFoundError(f"Source grasp cache not found: {source_grasps_path}")

    source_grasps = np.load(source_grasps_path, allow_pickle=False)
    mean_cycles = np.asarray(
        stats.get("consecutive_success_cycles_mean", stats["consecutive_success_cycles"]),
        dtype=np.float32,
    )
    if source_grasps.shape[0] != mean_cycles.shape[0]:
        raise ValueError(
            "Stats length does not match source grasp cache length: "
            f"stats={mean_cycles.shape[0]} vs grasps={source_grasps.shape[0]} from {source_grasps_path}"
        )

    if cycle_metric == "max":
        cycle_trials = stats.get("consecutive_success_cycles_trials")
        if cycle_trials is not None:
            selected_metric_values = np.asarray(cycle_trials, dtype=np.float32).max(axis=0)
        else:
            selected_metric_values = mean_cycles
    else:
        selected_metric_values = mean_cycles

    success_mask = selected_metric_values > float(threshold)
    ranking_order = list(
        stats.get("sorted_grasp_indices_by_mean_cycles", stats["sorted_grasp_indices_by_cycles"])
    )
    selected_indices = np.asarray(
        [int(idx) for idx in ranking_order if success_mask[int(idx)]],
        dtype=np.int64,
    )
    saved_grasps = source_grasps[selected_indices]

    output_dir = instance_dir / output_split
    output_vis_dir = output_dir / "grasp_visualization"
    output_grasps_path = output_dir / "valid_grasps.npy"
    output_summary_path = output_dir / "summary.json"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_grasps_path, saved_grasps)
    copied_pngs = _save_visualizations(source_vis_dir, output_vis_dir, selected_indices)

    summary = {
        "stats_type": "consecutive_eval",
        "hand": hand,
        "object": object_name,
        "asset_dir": asset_dir_name,
        "instance_id": instance_id,
        "source_split": grasp_split,
        "cycle_metric": cycle_metric,
        "cycle_threshold_strict_gt": float(threshold),
        "num_source_grasps": int(source_grasps.shape[0]),
        "num_success_grasps": int(saved_grasps.shape[0]),
        "selected_indices": selected_indices.tolist(),
        "copied_visualizations": int(copied_pngs),
        "consecutive_success_cycles_mean": mean_cycles[selected_indices].tolist(),
        "selected_cycle_metric_values": selected_metric_values[selected_indices].tolist(),
    }
    with output_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Saved {saved_grasps.shape[0]} / {source_grasps.shape[0]} consecutive-success grasps "
        f"({cycle_metric} cycles > {threshold:.3f}) to {output_grasps_path}"
    )


def _compute_instance_cycle_metrics(stats: dict) -> dict:
    mean_cycles = np.asarray(
        stats.get("consecutive_success_cycles_mean", stats["consecutive_success_cycles"]),
        dtype=np.float32,
    )
    cycle_trials = stats.get("consecutive_success_cycles_trials")
    if cycle_trials is not None:
        max_cycles = np.asarray(cycle_trials, dtype=np.float32).max(axis=0)
    else:
        max_cycles = mean_cycles

    ratio_grasps_max_cycles_gt_1 = float(np.mean(max_cycles > 1.0)) if max_cycles.size > 0 else 0.0
    mean_cycles_value = float(mean_cycles.mean()) if mean_cycles.size > 0 else 0.0

    return {
        "ratio_grasps_max_cycles_gt_1": ratio_grasps_max_cycles_gt_1,
        "mean_cycles": mean_cycles_value,
        "num_grasps_max_cycles_gt_1": int(np.sum(max_cycles > 1.0)),
    }


def _compute_threshold_success_metrics(stats: dict, *, threshold: float, cycle_metric: str) -> dict:
    mean_cycles = np.asarray(
        stats.get("consecutive_success_cycles_mean", stats["consecutive_success_cycles"]),
        dtype=np.float32,
    )
    cycle_trials = stats.get("consecutive_success_cycles_trials")
    if cycle_trials is not None:
        max_cycles = np.asarray(cycle_trials, dtype=np.float32).max(axis=0)
    else:
        max_cycles = mean_cycles

    selected_cycle_metric_values = max_cycles if cycle_metric == "max" else mean_cycles
    success_mask = selected_cycle_metric_values > float(threshold)
    successful_grasp_num = int(np.sum(success_mask))
    total_grasp_num = int(stats.get("num_grasps", mean_cycles.shape[0]))
    grasp_coverage = (
        float(successful_grasp_num) / float(total_grasp_num)
        if total_grasp_num > 0 else 0.0
    )
    consecutive_successful_cycles = (
        float(mean_cycles[success_mask].mean())
        if successful_grasp_num > 0 else 0.0
    )
    consecutive_successful_cycles_std = (
        float(mean_cycles[success_mask].std())
        if successful_grasp_num > 0 else 0.0
    )
    consecutive_successful_max_cycles = (
        float(max_cycles[success_mask].max())
        if successful_grasp_num > 0 else 0.0
    )
    consecutive_successful_max_cycles_std = 0.0
    feasible_instance = successful_grasp_num > 0

    return {
        "success_cycle_threshold": float(threshold),
        "success_cycle_metric": cycle_metric,
        "feasible_instance": feasible_instance,
        "instance_coverage": 1.0 if feasible_instance else 0.0,
        "successful_grasp_num": successful_grasp_num,
        "grasp_coverage": grasp_coverage,
        "consecutive_successful_cycles": consecutive_successful_cycles,
        "consecutive_successful_cycles_std": consecutive_successful_cycles_std,
        "consecutive_successful_max_cycles": consecutive_successful_max_cycles,
        "consecutive_successful_max_cycles_std": consecutive_successful_max_cycles_std,
    }


def _save_instance_cycle_metrics(
    *,
    stats: dict,
    hand: str,
    asset_dir_name: str,
    summary_suffix="",
    success_cycle_threshold=None,
    success_cycle_metric=None,
):
    repo_root = Path(__file__).resolve().parent.parent
    instance_dir = repo_root / "caches" / "initial_grasp" / hand / asset_dir_name / str(stats["instance_id"])
    instance_dir.mkdir(parents=True, exist_ok=True)
    summary_path = instance_dir / f"consecutive_eval_summary{summary_suffix}.json"
    mean_cycles = np.asarray(
        stats.get("consecutive_success_cycles_mean", stats["consecutive_success_cycles"]),
        dtype=np.float32,
    )
    cycle_trials = stats.get("consecutive_success_cycles_trials")
    if cycle_trials is not None:
        max_cycles = np.asarray(cycle_trials, dtype=np.float32).max(axis=0)
    else:
        max_cycles = mean_cycles
    sorted_indices = list(
        stats.get("sorted_grasp_indices_by_mean_cycles", stats["sorted_grasp_indices_by_cycles"])
    )
    top_grasps = []
    for rank, grasp_idx in enumerate(sorted_indices[:10], start=1):
        top_entry = {
            "rank": int(rank),
            "grasp_index": int(grasp_idx),
            "cycles_mean": float(mean_cycles[grasp_idx]),
            "cycles_max": float(max_cycles[grasp_idx]),
        }
        top_grasps.append(top_entry)

    summary = {
        "instance_id": stats["instance_id"],
        "grasp_split": stats["grasp_split"],
        "episodes_per_grasp": int(stats.get("episodes_per_grasp", 1)),
        "num_grasps": int(stats["num_grasps"]),
        "best_grasp_index": int(stats.get("best_grasp_index", -1)),
        "best_consecutive_success_cycles": float(stats.get("best_consecutive_success_cycles", 0.0)),
        "average_consecutive_success_cycles": float(stats.get("average_consecutive_success_cycles", 0.0)),
        "ratio_grasps_max_cycles_gt_1": float(stats["ratio_grasps_max_cycles_gt_1"]),
        "mean_cycles": float(stats["mean_cycles"]),
        "num_grasps_max_cycles_gt_1": int(stats["num_grasps_max_cycles_gt_1"]),
        "top_grasps_by_consecutive_success": top_grasps,
    }
    if success_cycle_threshold is not None:
        summary.update(
            _compute_threshold_success_metrics(
                stats,
                threshold=float(success_cycle_threshold),
                cycle_metric=str(success_cycle_metric or "mean"),
            )
        )
    else:
        summary.update({
            "success_cycle_threshold": None,
            "success_cycle_metric": None,
            "feasible_instance": None,
            "instance_coverage": None,
            "successful_grasp_num": None,
            "grasp_coverage": None,
            "consecutive_successful_cycles": None,
            "consecutive_successful_cycles_std": None,
            "consecutive_successful_max_cycles": None,
            "consecutive_successful_max_cycles_std": None,
        })
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved instance consecutive summary to {summary_path}")


def _split_parallel_trial_stats(raw_stats: dict) -> List[dict]:
    parallel_trials = int(raw_stats.get("parallel_trials_per_grasp", 1))
    base_num_grasps = int(raw_stats.get("base_num_grasps", raw_stats.get("num_grasps", 0)))
    if parallel_trials <= 1:
        return [dict(raw_stats)]

    total_num_grasps = int(raw_stats.get("num_grasps", 0))
    if total_num_grasps != parallel_trials * base_num_grasps:
        raise ValueError(
            "Parallel eval stats shape mismatch: "
            f"num_grasps={total_num_grasps}, parallel_trials_per_grasp={parallel_trials}, "
            f"base_num_grasps={base_num_grasps}"
        )

    vector_keys = [
        "consecutive_success_cycles",
        "completion_reason_code",
        "completion_reason",
        "completion_stage_code",
        "completion_stage",
        "completion_goal_distance",
    ]
    split_stats = []
    for trial_idx in range(parallel_trials):
        start = trial_idx * base_num_grasps
        end = start + base_num_grasps
        trial_stats = dict(raw_stats)
        trial_stats["trial_index"] = int(trial_idx)
        trial_stats["num_grasps"] = int(base_num_grasps)
        trial_stats["parallel_trials_per_grasp"] = 1
        trial_stats["base_num_grasps"] = int(base_num_grasps)
        for key in vector_keys:
            values = raw_stats.get(key)
            if values is None:
                continue
            trial_stats[key] = values[start:end]
        trial_cycles = np.asarray(trial_stats["consecutive_success_cycles"], dtype=np.float32)
        sorted_indices = np.argsort(-trial_cycles, kind="stable").tolist()
        trial_stats["sorted_grasp_indices_by_cycles"] = [int(idx) for idx in sorted_indices]
        trial_stats["best_grasp_index"] = int(sorted_indices[0]) if sorted_indices else -1
        trial_stats["best_consecutive_success_cycles"] = (
            int(trial_cycles[sorted_indices[0]]) if sorted_indices else 0
        )
        split_stats.append(trial_stats)
    return split_stats


def print_consecutive_summary(stats, top_k=10):
    mean_cycles = np.asarray(
        stats.get("consecutive_success_cycles_mean", stats["consecutive_success_cycles"]),
        dtype=np.float32,
    )
    std_cycles = np.asarray(stats.get("consecutive_success_cycles_std", np.zeros_like(mean_cycles)), dtype=np.float32)
    cycle_trials = stats.get("consecutive_success_cycles_trials")
    reasons = stats["completion_reason"]
    stages = stats["completion_stage"]
    goal_dists = np.asarray(
        stats.get("completion_goal_distance_mean", stats["completion_goal_distance"]),
        dtype=np.float32,
    )
    sorted_indices = list(
        stats.get("sorted_grasp_indices_by_mean_cycles", stats["sorted_grasp_indices_by_cycles"])
    )
    top_k = max(1, min(int(top_k), len(sorted_indices))) if len(sorted_indices) > 0 else 0

    print(f"Evaluated instance: {stats['instance_id']}")
    print(f"Grasp split: {stats['grasp_split']}")
    print(f"Num grasps: {stats['num_grasps']}")
    print(f"Episodes per grasp: {stats.get('episodes_per_grasp', 1)}")
    print(f"Goal sequence: {stats['goal_sequence']}")
    print(f"Stage duration: {stats['stage_duration']:.3f}s")
    if "episode_length_steps" in stats:
        print(f"Episode length budget: {stats['episode_length_steps']} steps")
    print(f"Best grasp index: {stats['best_grasp_index']}")
    print(f"Best consecutive success cycles: {stats['best_consecutive_success_cycles']:.4f}")
    print(f"Average consecutive success cycles: {stats.get('average_consecutive_success_cycles', 0.0):.4f}")

    if top_k > 0:
        print(f"Top {top_k} grasps by consecutive success:")
        for rank, grasp_idx in enumerate(sorted_indices[:top_k], start=1):
            cycle_trials_str = ""
            if cycle_trials is not None:
                grasp_trials = [int(trial[grasp_idx]) for trial in cycle_trials]
                cycle_trials_str = f" | cycles_trials={grasp_trials}"
            print(
                f"rank {rank:02d} | grasp {grasp_idx:04d} | "
                f"cycles_mean={float(mean_cycles[grasp_idx]):.4f} | "
                f"cycles_std={float(std_cycles[grasp_idx]):.4f} | "
                f"reason={reasons[grasp_idx]} | "
                f"stage={stages[grasp_idx]} | "
                f"goal_dist={float(goal_dists[grasp_idx]):.6f}"
                f"{cycle_trials_str}"
            )


def main():
    args = parse_args()
    args.grasp_split = normalize_grasp_split(args.grasp_split)
    if args.episodes_per_grasp <= 0:
        raise ValueError(f"--episodes-per-grasp must be positive, got {args.episodes_per_grasp}")
    force_single_process_env()
    checkpoint_path = Path(args.checkpoint).resolve()
    inferred_blocks = _infer_expl_num_blocks(checkpoint_path)

    distill_meta = {}
    if args.student_artifact:
        student_artifact_path = Path(args.student_artifact).resolve()
        if not student_artifact_path.exists():
            raise FileNotFoundError(f"Student artifact not found: {student_artifact_path}")
        student_artifact = torch.load(student_artifact_path, map_location="cpu")
        distill_meta = student_artifact.get("distill_meta", {})
    else:
        student_artifact = None

    import isaacgymenvs
    isaacgymenvs.register_omegaconf_resolvers()

    from isaacgymenvs.infer_student_impl import build_student_encoder_from_artifact
    from isaacgymenvs.learning import a2c_dict_network_builder, a2c_sapg_priv_network_builder
    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from rl_games.algos_torch import model_builder
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    resolved_identity = resolve_teacher_identity_from_meta(
        distill_meta=distill_meta,
        task=args.task,
        train=args.train or "",
        hand=args.hand,
        object_name=args.object,
    )

    overrides = [
        f"task={resolved_identity['task']}",
        f"hand={resolved_identity['hand']}",
        f"object={resolved_identity['object']}",
        f"asset_dir={args.asset_dir}",
        f"headless={args.headless}",
        f"sim_device={args.sim_device}",
        f"rl_device={args.rl_device}",
        f"graphics_device_id={args.graphics_device_id}",
        f"seed={args.seed}",
        "multi_gpu=False",
    ]
    if resolved_identity["train"]:
        overrides.append(f"train={resolved_identity['train']}")
    overrides.extend(build_student_cfg_overrides_from_meta(distill_meta))
    with initialize(version_base="1.1", config_path="./cfg"):
        cfg = compose(config_name="config", overrides=overrides)

    effective_asset_root = _resolve_effective_asset_root(cfg, args.asset_dir)
    repo_root = Path(__file__).resolve().parent.parent
    grasp_dir = repo_root / "caches" / "initial_grasp" / cfg.hand.type / effective_asset_root.name / args.instance_id
    grasp_cache = resolve_grasp_cache_path(grasp_dir, args.grasp_split)
    effective_grasp_split = args.grasp_split
    if not grasp_cache.exists():
        raise FileNotFoundError(f"Could not find grasp file: {grasp_cache}")

    grasp_count = int(torch.from_numpy(np.load(grasp_cache)).shape[0])
    instance_id_list = list(cfg.object.asset.instance_id_list)
    if args.instance_id not in instance_id_list and instance_id_list != [""]:
        raise ValueError(f"instance_id '{args.instance_id}' not found in object config list {instance_id_list}")

    with open_dict(cfg):
        cfg.task.env.numEnvs = grasp_count * int(args.episodes_per_grasp)
        cfg.task.env.episodeLength = int(args.max_steps)
        cfg.task.env.graspSplit = effective_grasp_split
        if args.goal_switch_interval_secs is not None:
            cfg.task.env.successHoldDurationSec = float(args.goal_switch_interval_secs)
            cfg.task.env.successHoldDurationRangeSec = [
                float(args.goal_switch_interval_secs),
                float(args.goal_switch_interval_secs),
            ]
        cfg.task.task.randomize = bool(args.randomize)
        if args.torch_deterministic is not None:
            cfg.torch_deterministic = bool(args.torch_deterministic)
        cfg.multi_gpu = False
        cfg.object.asset.asset_root = str(effective_asset_root)
        cfg.object.asset.instance_id_list = [args.instance_id]
        cfg.test = True
        cfg.checkpoint = str(checkpoint_path)
        if args.save_video:
            cfg.task.env.enableCameraSensors = True

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0)

    def create_env(**kwargs):
        return isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            False,
            cfg.force_render,
            cfg,
            **kwargs,
        )

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: create_env(**kwargs),
        },
    )
    vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    env_cls = isaacgym_task_map[cfg.task_name]
    if getattr(env_cls, "dict_obs_cls", False):
        raise RuntimeError("eval_consecutive.py currently supports flat observations only.")

    rlg_config_dict = preprocess_train_config(cfg, omegaconf_to_dict(cfg.train))
    player_cfg = rlg_config_dict["params"]["config"].setdefault("player", {})
    if inferred_blocks is not None:
        player_cfg["expl_num_blocks"] = inferred_blocks

    model_builder.register_network("actor_critic_dict", a2c_dict_network_builder.A2CBuilder)
    model_builder.register_network("actor_critic_sapg_priv", a2c_sapg_priv_network_builder.A2CSAPGPrivBuilder)
    runner = Runner()
    runner.load(rlg_config_dict)
    player = runner.create_player()
    player.restore(cfg.checkpoint)

    if inferred_blocks is not None and player.intr_reward_coef_embd is not None:
        block_idx = max(0, min(args.expl_block_idx, inferred_blocks - 1))
        coef_ids = torch.linspace(50.0, 0.0, inferred_blocks, device=player.device)
        player.intr_reward_coef_embd[:] = coef_ids[block_idx]
        print(f"Using teacher exploration block {block_idx} with id {coef_ids[block_idx].item():.4f}")

    task_env = player.env.env
    if student_artifact is not None:
        student_encoder, _, _, _ = build_student_encoder_from_artifact(
            player=player,
            cfg=cfg,
            rlg_config_dict=rlg_config_dict,
            student_artifact=student_artifact,
            distill_meta=distill_meta,
        )
        player.model.a2c_network.priv_encoder = student_encoder
        player.model.eval()
        if hasattr(task_env, "set_student_encoder_obs_enabled"):
            task_env.set_student_encoder_obs_enabled(True)

    object_goals = cfg.object.task.get("goals")
    if object_goals is None:
        raise ValueError("object.task.goals is required.")
    goal_sequence = list(object_goals)
    if len(goal_sequence) != 2:
        raise ValueError(f"object.task.goals must contain exactly two values, got {goal_sequence}")
    if args.open_goal is not None:
        goal_sequence[0] = float(args.open_goal)
    if args.close_goal is not None:
        goal_sequence[1] = float(args.close_goal)

    video_writer = None
    if args.save_video:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError("imageio is required for --save-video. Please install imageio in this environment.") from exc
        video_path = Path(args.save_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=args.video_fps)

    last_trial_action_sequence = None
    raw_stats = None
    try:
        task_env.configure_grasp_consecutive_evaluation(
            instance_id=args.instance_id,
            goal_sequence=tuple(goal_sequence),
            stage_duration=args.stage_duration_secs,
            grasp_split=effective_grasp_split,
            episodes_per_grasp=args.episodes_per_grasp,
        )

        run_grasp_evaluation_loop(
            player=player,
            task_env=task_env,
            deterministic=args.deterministic,
            progress_interval_sec=args.progress_interval_sec,
            use_student_encoder=student_artifact is not None,
            render_human=not args.headless,
            record_action_sequence=bool(args.save_last_episode_actions),
            video_writer=video_writer,
            video_env_index=args.video_env_index,
        )
        if args.save_last_episode_actions:
            last_trial_action_sequence = getattr(task_env, "last_eval_action_sequence", None)
        raw_stats = task_env.get_grasp_consecutive_evaluation_stats()
    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"Saved video to {Path(args.save_video)}")

    all_trial_stats = _split_parallel_trial_stats(raw_stats)
    stats = aggregate_consecutive_stats(all_trial_stats)
    if args.episodes_per_grasp > 1:
        stats["trial_stats"] = all_trial_stats
    stats.update(_compute_instance_cycle_metrics(stats))
    print_consecutive_summary(stats, top_k=args.top_k)

    if args.save_last_episode_actions:
        if task_env.num_envs != 1:
            raise ValueError("--save-last-episode-actions currently supports only num_envs=1.")
        if last_trial_action_sequence is None:
            raise RuntimeError("No action sequence was recorded to save.")
        _save_last_episode_actions(
            save_path=Path(args.save_last_episode_actions),
            actions=last_trial_action_sequence,
            args=args,
            cfg=cfg,
            asset_dir_name=effective_asset_root.name,
            grasp_split=effective_grasp_split,
            goal_sequence=goal_sequence,
            stage_duration_secs=args.stage_duration_secs,
        )

    _save_instance_cycle_metrics(
        stats=stats,
        hand=cfg.hand.type,
        asset_dir_name=effective_asset_root.name,
        summary_suffix="_student" if student_artifact is not None else "",
        success_cycle_threshold=args.save_success_cycle_threshold,
        success_cycle_metric=args.save_success_cycle_metric,
    )

    if args.save_success_cycle_threshold is not None:
        _save_consecutive_success_grasps(
            stats=stats,
            threshold=float(args.save_success_cycle_threshold),
            cycle_metric=str(args.save_success_cycle_metric),
            output_split=str(args.save_success_output_split),
            hand=cfg.hand.type,
            object_name=args.object,
            asset_dir_name=effective_asset_root.name,
        )


if __name__ == "__main__":
    main()
