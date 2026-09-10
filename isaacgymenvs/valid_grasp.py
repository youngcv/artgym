import argparse
from pathlib import Path

import isaacgym  # noqa: F401
import torch
from hydra import compose, initialize
from omegaconf import open_dict

from isaacgymenvs.utils.distributed_runtime import force_single_process_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ArtGrasp as a standalone grasp-validation generator without rl_games."
    )
    parser.add_argument("--pipeline", default="gpu", help="sim pipeline")
    parser.add_argument("--hand", default="sharpa", help="Hand config name.")
    parser.add_argument("--object", default="knife", help="Object config name.")
    parser.add_argument("--asset-dir", default="", help="Optional asset directory under assets/objects, e.g. knife_multi.")
    parser.add_argument(
        "--instance-id",
        default=None,
        help="Validate exactly one object instance by overriding object.asset.instance_id_list to [instance_id].",
    )
    parser.add_argument("--num-envs", type=int, required=True, help="Batch size / number of envs.")
    parser.add_argument("--episode-length", type=int, default=30, help="Episode length for each batch.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--sim-device", default="cuda:0", help="Simulation device.")
    parser.add_argument("--rl-device", default="cuda:0", help="RL device.")
    parser.add_argument("--graphics-device-id", type=int, default=0, help="Graphics device id.")
    parser.add_argument("--rot-threshold", type=float, default=None, help="Override object.task.rot_devia_threshold.")
    parser.add_argument("--pos-threshold", type=float, default=None, help="Override object.task.pos_devia_threshold.")
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Filter near-duplicate object poses before saving the valid grasp pool.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Do not split grasps into train/test. Save the full pool under caches/.../<instance>/valid/.",
    )
    parser.add_argument(
        "--unique-pos-threshold",
        type=float,
        default=0.005,
        help="Treat object poses within this translation distance as duplicates.",
    )
    parser.add_argument(
        "--unique-rot-threshold",
        type=float,
        default=0.05,
        help="Treat object poses within this quaternion angular distance (radians) as duplicates.",
    )
    parser.add_argument("--camera", action="store_true", help="Enable camera sensors and save visualization PNGs.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override. Can be passed multiple times.",
    )
    return parser.parse_args()


def main(force_unique=False):
    args = parse_args()
    if force_unique:
        args.unique = True
    force_single_process_env()

    import isaacgymenvs
    isaacgymenvs.register_omegaconf_resolvers()

    overrides = [
        "task=artgrasp",
        "train=artmanipPrivLSTMPPO",
        f"pipeline={args.pipeline}",
        f"hand={args.hand}",
        f"object={args.object}",
        f"asset_dir={args.asset_dir}",
        f"task.env.numEnvs={args.num_envs}",
        f"task.env.episodeLength={args.episode_length}",
        f"headless={args.headless}",
        f"seed={args.seed}",
        f"sim_device={args.sim_device}",
        f"rl_device={args.rl_device}",
        f"graphics_device_id={args.graphics_device_id}",
        "multi_gpu=False",
        "hand.randomization.randomize=False",
        "object.randomization.randomize=False",
    ]

    if args.unique:
        overrides.extend(
            [
                "+task.env.filterUniqueGrasps=True",
                f"+task.env.uniqueGraspPosThreshold={args.unique_pos_threshold}",
                f"+task.env.uniqueGraspRotThreshold={args.unique_rot_threshold}",
            ]
        )
    if args.no_split:
        overrides.append("+task.env.noSplitGrasps=True")

    if args.instance_id is not None:
        overrides.extend(
            [
                f"object.asset.instance_id_list=['{args.instance_id}']",
            ]
        )

    if args.camera:
        overrides.extend(
            [
                "task.env.enableCameraSensors=True",
            ]
        )
    else:
        overrides.extend(
            [
                "task.env.enableCameraSensors=False",
            ]
        )

    if args.rot_threshold is not None:
        overrides.append(f"object.task.rot_devia_threshold={args.rot_threshold}")
    if args.pos_threshold is not None:
        overrides.append(f"object.task.pos_devia_threshold={args.pos_threshold}")

    overrides.extend(args.override)

    with initialize(version_base="1.1", config_path="./cfg"):
        cfg = compose(config_name="config", overrides=overrides)

    with open_dict(cfg):
        cfg.test = False
        cfg.multi_gpu = False

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

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_idx(env_ids)

    actions = env.zero_actions()
    try:
        while True:
            env.step(actions)
    except SystemExit:
        task_env = env
        if args.unique:
            print(f"Saved unique valid grasps to {Path(task_env.valid_grasp_path).resolve()}")
        else:
            print(f"Saved valid grasps to {Path(task_env.valid_grasp_path).resolve()}")
        if args.no_split:
            print(f"Saved valid-only split grasps to {Path(task_env.valid_split_grasp_path).resolve()}")
        if task_env.enable_camera:
            if hasattr(task_env, "train_grasp_vis_dir") and hasattr(task_env, "test_grasp_vis_dir"):
                if args.no_split:
                    print(f"Saved valid grasp images to {Path(task_env.valid_grasp_vis_dir).resolve()}")
                else:
                    print(f"Saved training grasp images to {Path(task_env.train_grasp_vis_dir).resolve()}")
                    print(f"Saved test grasp images to {Path(task_env.test_grasp_vis_dir).resolve()}")
            else:
                print(f"Saved grasp images to {Path(task_env.grasp_vis_dir).resolve()}")


if __name__ == "__main__":
    main()
