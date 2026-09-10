import argparse
import json
from pathlib import Path

import isaacgym  # noqa: F401
import torch
from hydra import compose, initialize
from omegaconf import open_dict

import isaacgymenvs
from isaacgymenvs import register_omegaconf_resolvers
from isaacgymenvs.consecutive_eval_utils import aggregate_consecutive_stats
from isaacgymenvs.utils.player_utils import parse_bool_arg
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a saved eval-consecutive action sequence after warmup steps."
    )
    parser.add_argument("--actions-file", required=True, help="Path to a saved action-sequence .pt file.")
    parser.add_argument("--episodes", type=int, default=10, help="How many replay trials to run.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument("--save-video", default="", help="Optional video path (.mp4/.gif) recorded from artmanip.yaml camera.")
    parser.add_argument("--video-env-index", type=int, default=0, help="Which env camera to record when --save-video is used.")
    parser.add_argument("--video-fps", type=int, default=30, help="Output video fps when --save-video is used.")
    parser.add_argument("--sim-device", default="", help="Optional override for simulation device.")
    parser.add_argument("--rl-device", default="", help="Optional override for RL device.")
    parser.add_argument("--graphics-device-id", type=int, default=None, help="Optional override for graphics device id.")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for random seed.")
    parser.add_argument(
        "--torch-deterministic",
        type=parse_bool_arg,
        default=None,
        help="Optional override for cfg.torch_deterministic.",
    )
    parser.add_argument(
        "--randomize",
        type=parse_bool_arg,
        default=None,
        help="Optional override for task.randomize during replay.",
    )
    parser.add_argument(
        "--print-reset-snapshot",
        action="store_true",
        help="Print the post-warmup reset snapshot for each replay episode.",
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=5.0,
        help="How often to print replay progress in seconds. Set <= 0 to disable.",
    )
    return parser.parse_args()


def _print_consecutive_summary(stats: dict):
    cycles = stats["consecutive_success_cycles"]
    reasons = stats["completion_reason"]
    stages = stats["completion_stage"]
    goal_dist = stats["completion_goal_distance"]
    print(
        "Replay consecutive summary: "
        f"cycles={cycles} reason={reasons} stage={stages} goal_dist={goal_dist}"
    )


def _run_replay_trial(task_env, action_sequence, *, progress_interval_sec=0.0, video_writer=None, video_env_index=0):
    obses = task_env.reset()
    warmup_steps = int(getattr(task_env, "eval_policy_warmup_steps", 0))
    zero_actions = torch.zeros(
        (task_env.num_envs, task_env.cfg["env"]["numActions"]),
        dtype=torch.float32,
        device=task_env.device,
    )
    for _ in range(max(0, warmup_steps)):
        obses, _, _, _ = task_env.step(zero_actions)

    progress_interval_sec = float(progress_interval_sec)
    next_progress_time = None
    if progress_interval_sec > 0.0:
        import time
        next_progress_time = time.monotonic() + progress_interval_sec

    step_idx = 0
    while not task_env.is_grasp_evaluation_complete():
        action_index = min(step_idx, action_sequence.shape[0] - 1)
        action = action_sequence[action_index].unsqueeze(0).to(device=task_env.device, dtype=torch.float32)
        obses, _, _, _ = task_env.step(action)
        step_idx += 1

        if video_writer is not None:
            if not hasattr(task_env, "get_camera_frame"):
                raise RuntimeError("Task env does not expose get_camera_frame(), so --save-video cannot be used.")
            video_writer.append_data(task_env.get_camera_frame(env_id=video_env_index))

        if next_progress_time is not None:
            import time
            if time.monotonic() >= next_progress_time:
                completed_episodes = int(task_env.eval_episode_counts.sum().item())
                print(
                    f"[replay progress] steps={step_idx} | episodes={completed_episodes}/{task_env.eval_episodes_per_grasp}"
                )
                next_progress_time = time.monotonic() + progress_interval_sec

    return task_env.get_grasp_consecutive_evaluation_stats()


def main():
    args = parse_args()
    payload = torch.load(Path(args.actions_file), map_location="cpu")
    action_sequence = payload["actions"]
    metadata = payload["metadata"]

    register_omegaconf_resolvers()

    sim_device = args.sim_device or str(metadata["sim_device"])
    rl_device = args.rl_device or str(metadata["rl_device"])
    graphics_device_id = (
        int(metadata["graphics_device_id"])
        if args.graphics_device_id is None
        else int(args.graphics_device_id)
    )
    seed = int(metadata["seed"]) if args.seed is None else int(args.seed)
    randomize = bool(metadata["randomize"]) if args.randomize is None else bool(args.randomize)

    overrides = [
        f"task={metadata['task']}",
        f"hand={metadata['hand']}",
        f"object={metadata['object']}",
        f"asset_dir={metadata['asset_dir']}",
        f"headless={args.headless}",
        f"sim_device={sim_device}",
        f"rl_device={rl_device}",
        f"graphics_device_id={graphics_device_id}",
        f"seed={seed}",
        "multi_gpu=False",
    ]
    train_name = str(metadata.get("train", ""))
    if train_name:
        overrides.append(f"train={train_name}")

    with initialize(version_base="1.1", config_path="./cfg"):
        cfg = compose(config_name="config", overrides=overrides)

    with open_dict(cfg):
        cfg.task.env.numEnvs = 1
        cfg.task.env.episodeLength = int(metadata["max_steps"])
        if metadata.get("goal_switch_interval_secs") is not None:
            cfg.task.env.successHoldDurationSec = float(metadata["goal_switch_interval_secs"])
            cfg.task.env.successHoldDurationRangeSec = [
                float(metadata["goal_switch_interval_secs"]),
                float(metadata["goal_switch_interval_secs"]),
            ]
        cfg.task.task.randomize = randomize
        if args.torch_deterministic is not None:
            cfg.torch_deterministic = bool(args.torch_deterministic)
        elif metadata.get("torch_deterministic") is not None:
            cfg.torch_deterministic = bool(metadata["torch_deterministic"])
        cfg.multi_gpu = False
        cfg.object.asset.instance_id_list = [str(metadata["instance_id"])]
        cfg.test = True
        if args.save_video:
            cfg.task.env.enableCameraSensors = True

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0)

    env = isaacgymenvs.make(
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
    )
    task_env = env
    if hasattr(task_env, "eval_policy_warmup_steps"):
        task_env.eval_policy_warmup_steps = int(metadata.get("warmup_steps", task_env.eval_policy_warmup_steps))
    if hasattr(task_env, "set_eval_reset_snapshot_debug"):
        task_env.set_eval_reset_snapshot_debug(args.print_reset_snapshot)

    video_writer = None
    if args.save_video:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError("imageio is required for --save-video. Please install imageio in this environment.") from exc
        video_path = Path(args.save_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=args.video_fps)

    all_trial_stats = []
    try:
        for trial_idx in range(args.episodes):
            if args.episodes > 1:
                print(f"Running replay trial {trial_idx + 1}/{args.episodes}")
            task_env.configure_grasp_consecutive_evaluation(
                instance_id=str(metadata["instance_id"]),
                goal_sequence=tuple(metadata["goal_sequence"]),
                stage_duration=metadata.get("stage_duration_secs"),
                grasp_split=str(metadata["grasp_split"]),
            )
            trial_stats = _run_replay_trial(
                task_env,
                action_sequence=action_sequence,
                progress_interval_sec=args.progress_interval_sec,
                video_writer=video_writer,
                video_env_index=args.video_env_index,
            )
            trial_stats["trial_index"] = int(trial_idx)
            all_trial_stats.append(trial_stats)
    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"Saved video to {Path(args.save_video)}")

    stats = aggregate_consecutive_stats(all_trial_stats)
    if args.episodes > 1:
        stats["trial_stats"] = all_trial_stats
    print(json.dumps(stats, indent=2))
    _print_consecutive_summary(stats)


if __name__ == "__main__":
    main()
