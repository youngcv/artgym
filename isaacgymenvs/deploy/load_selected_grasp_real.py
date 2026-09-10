from __future__ import annotations

import argparse
import json
import os

import numpy as np

from isaacgymenvs.deploy.real_robot_policy_api import HandAPI
from isaacgymenvs.deploy.task_state_provider import (
    GraspCacheTaskStateProvider,
)
from isaacgymenvs.deploy.utils import filter_constructor_kwargs, load_class, resolve_hand_robot_class_spec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a cached selected_grasps.npy pose and command it on a real hand."
    )
    parser.add_argument("--robot-class", default="", help="Optional import path in module:Class format. Defaults to isaacgymenvs.deploy.<hand>.robot:<HandName>Robot.")
    parser.add_argument("--robot-kwargs-json", default="{}", help="JSON dict passed to the hand backend constructor.")
    parser.add_argument("--grasp-settle-sec", type=float, default=1.0, help="How long to wait after commanding the selected grasp.")
    parser.add_argument("--hand", default="sharpa", help="Hand type used to resolve selected grasps.")
    parser.add_argument("--object", default="knife", help="Object type used to resolve selected grasps.")
    parser.add_argument(
        "--asset-dir",
        default="",
        help="Optional asset/cache directory name under assets/objects and caches/initial_grasp, e.g. knife_30.",
    )
    parser.add_argument("--grasp-instance-id", default="000", help="Cache instance id, e.g. 000.")
    parser.add_argument("--cache-root", default="caches/initial_grasp", help="Root directory for cached grasp files.")
    return parser.parse_args()


def build_robot(args):
    robot_cls = load_class(resolve_hand_robot_class_spec(args.hand, args.robot_class), HandAPI)
    robot_kwargs = json.loads(args.robot_kwargs_json)
    return robot_cls(**filter_constructor_kwargs(robot_cls, robot_kwargs))

def load_selected_grasp(args, num_hand_dofs: int) -> np.ndarray:
    provider = GraspCacheTaskStateProvider(
        hand_type=args.hand,
        object_type=args.object,
        asset_dir=args.asset_dir,
        instance_id=args.grasp_instance_id,
        cache_root=args.cache_root,
        command_cached_grasp=False,
    )
    provider.context = type("SelectedGraspContext", (), {"hand_dof_dim": num_hand_dofs})()
    cache_path = provider._resolve_cache_path()
    grasp_states = np.load(cache_path)

    if grasp_states.ndim == 1:
        grasp_row = grasp_states
    elif grasp_states.ndim == 2:
        if grasp_states.shape[0] != 1:
            raise ValueError(
                f"selected_grasps file {cache_path} must contain exactly one row, got {grasp_states.shape[0]}."
            )
        grasp_row = grasp_states[0]
    else:
        raise ValueError(
            f"Expected selected grasp states to be 1D or 2D, got {grasp_states.shape}"
        )

    unpacked = provider._unpack_grasp_state_row(grasp_row, hand_dof_dim=num_hand_dofs)
    cached_qpos = np.asarray(unpacked["hand_dof_pos"], dtype=np.float32)
    if cached_qpos.shape != (num_hand_dofs,):
        raise ValueError(
            f"Cached grasp hand_dof_pos must have shape ({num_hand_dofs},), got {cached_qpos.shape}"
        )

    print(
        "Loaded selected grasp [0] from "
        f"{os.path.join(args.cache_root, args.hand, args.asset_dir if args.asset_dir else args.object, args.grasp_instance_id, 'selected_grasps.npy')}"
    )
    return cached_qpos


def main():
    args = parse_args()
    robot = build_robot(args)
    init_grasp = load_selected_grasp(args, num_hand_dofs=robot.get_num_hand_dofs())
    try:
        robot.connect()
        input("Hand connected and zeroed. Press Enter to load the selected grasp...")
        robot.command_init_grasp(init_grasp, final_settle_sec=args.grasp_settle_sec)
        input("Selected grasp loaded. Insert/evaluate the object, then press Enter to disconnect...")
    except KeyboardInterrupt:
        print("Selected grasp evaluation interrupted by user.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
