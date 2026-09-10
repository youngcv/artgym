from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from isaacgymenvs.deploy.real_robot_policy_api import HandAPI
from isaacgymenvs.deploy.utils import filter_constructor_kwargs, load_class, resolve_hand_robot_class_spec


def parse_args():
    parser = argparse.ArgumentParser(description="Replay saved cur_targets.npy trajectories on a real hand.")
    parser.add_argument("--hand", default="sharpa", help="Hand backend name used for default class resolution.")
    parser.add_argument("--cur-targets-npy", required=True, help="Path to the saved cur_targets .npy file.")
    parser.add_argument("--env-index", type=int, default=0, help="Which env trajectory to replay when the file shape is [T, num_envs, num_dofs].")
    parser.add_argument("--replay-hz", type=float, default=30.0, help="Replay frequency in Hz.")
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum replay steps after frame 0. Use 0 to replay the full trajectory.")
    parser.add_argument("--robot-class", default="", help="Optional import path in module:Class format. Defaults to isaacgymenvs.deploy.<hand>.robot:<HandName>Robot.")
    parser.add_argument("--robot-kwargs-json", default="{}", help="JSON dict passed to the hand backend constructor.")
    parser.add_argument("--init-grasp-settle-sec", type=float, default=1.0, help="How long to wait after commanding the init grasp.")
    return parser.parse_args()


def load_replay_targets(path: Path, env_index: int, num_hand_dofs: int) -> np.ndarray:
    targets = np.load(path, allow_pickle=False)
    if targets.ndim == 2:
        if env_index != 0:
            raise ValueError(
                f"cur_targets file {path} has shape {targets.shape}, so only env_index=0 is valid."
            )
        selected = targets
    elif targets.ndim == 3:
        if env_index < 0 or env_index >= targets.shape[1]:
            raise ValueError(
                f"env_index={env_index} is out of range for cur_targets shape {targets.shape}."
            )
        selected = targets[:, env_index, :]
    else:
        raise ValueError(
            f"cur_targets file must have shape [T, num_dofs] or [T, num_envs, num_dofs], got {targets.shape}."
        )

    if selected.shape[0] <= 0:
        raise ValueError(f"cur_targets file {path} is empty.")
    if selected.shape[1] < num_hand_dofs:
        raise ValueError(
            f"cur_targets width must be at least {num_hand_dofs} to contain hand targets, got {selected.shape[1]}."
        )

    hand_targets = np.asarray(selected[:, :num_hand_dofs], dtype=np.float32)
    return hand_targets

def build_robot(args):
    robot_cls = load_class(resolve_hand_robot_class_spec(args.hand, args.robot_class), HandAPI)
    robot_kwargs = json.loads(args.robot_kwargs_json)
    return robot_cls(**filter_constructor_kwargs(robot_cls, robot_kwargs))


def replay_targets(
    robot: HandAPI,
    hand_targets: np.ndarray,
    replay_hz: float,
    max_steps: int,
) -> None:
    if replay_hz <= 0.0:
        raise ValueError(f"replay_hz must be positive, got {replay_hz}")

    replay_seq = hand_targets[1:]
    if max_steps > 0:
        replay_seq = replay_seq[:max_steps]

    if replay_seq.shape[0] == 0:
        print("No replay steps remain after the init grasp; nothing to execute.")
        return

    period = 1.0 / replay_hz
    sample_delay = 0.8 * period
    start_time = time.monotonic()
    for step_idx, joint_targets in enumerate(replay_seq):
        deadline = start_time + step_idx * period
        now = time.monotonic()
        if now < deadline:
            time.sleep(deadline - now)
        robot.command_joint_targets(joint_targets)
        sample_time = deadline + sample_delay
        now = time.monotonic()
        if now < sample_time:
            time.sleep(sample_time - now)


def main():
    args = parse_args()
    cur_targets_path = Path(args.cur_targets_npy).expanduser().resolve()
    if not cur_targets_path.exists():
        raise FileNotFoundError(f"cur_targets file not found: {cur_targets_path}")

    robot = build_robot(args)
    hand_targets = load_replay_targets(
        cur_targets_path,
        env_index=args.env_index,
        num_hand_dofs=robot.get_num_hand_dofs(),
    )
    print(f"Loaded replay targets from {cur_targets_path} with hand target shape {hand_targets.shape}.")
    init_grasp = hand_targets[0]
    print("Using cur_targets[0] as the init grasp and replaying cur_targets[1:].")

    try:
        robot.connect()
        input("Hand connected and zeroed. Press Enter to load the init grasp...")
        robot.command_init_grasp(init_grasp, final_settle_sec=args.init_grasp_settle_sec)
        input("Init grasp loaded. Insert the object, then press Enter to start replay...")
        replay_targets(
            robot,
            hand_targets,
            replay_hz=args.replay_hz,
            max_steps=args.max_steps,
        )
        print("Replay complete.")
    except KeyboardInterrupt:
        print("Replay interrupted by user.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
