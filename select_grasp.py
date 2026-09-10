import argparse
from pathlib import Path

import numpy as np

from isaacgymenvs.deploy.sharpa.robot import REAL_JOINT_NAMES, SIM_JOINT_NAMES


DEFAULT_HAND = "sharpa"
DEFAULT_OBJECT = "knife"
DEFAULT_INSTANCE_ID = "000"
DEFAULT_GROUP = "success"
DEFAULT_SELECT_IDX = [37]
DEFAULT_NUM_HAND_DOFS = 22


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select one or more rows from a grasp pool and write them to "
            "caches/initial_grasp/<hand>/<asset_dir_or_object>/<instance>/selected_grasps.npy."
        )
    )
    parser.add_argument("--hand", default=DEFAULT_HAND, help="Hand cache name.")
    parser.add_argument("--object", default=DEFAULT_OBJECT, help="Object cache name.")
    parser.add_argument(
        "--asset-dir",
        default="",
        help="Optional asset/cache directory name under assets/objects and caches/initial_grasp, e.g. knife_30.",
    )
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE_ID, help="Object instance id.")
    parser.add_argument(
        "--group",
        default=DEFAULT_GROUP,
        help="Source grasp pool folder name, for example train, test, valid, or success.",
    )
    parser.add_argument(
        "--select-idx",
        nargs="+",
        type=int,
        default=DEFAULT_SELECT_IDX,
        help="One or more source grasp row indices to save into selected_grasps.npy.",
    )
    parser.add_argument(
        "--cache-root",
        default="./caches/initial_grasp",
        help="Initial grasp cache root.",
    )
    parser.add_argument(
        "--num-hand-dofs",
        type=int,
        default=DEFAULT_NUM_HAND_DOFS,
        help="How many leading grasp values belong to the hand qpos.",
    )
    parser.add_argument(
        "--source-grasps",
        default="",
        help="Optional explicit source valid_grasps.npy path. Overrides --cache-root/--hand/--object/--instance-id/--group.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit selected_grasps.npy output path. Defaults to the instance root selected_grasps.npy.",
    )
    return parser.parse_args()


def sim_to_real_joint_order(values: np.ndarray) -> np.ndarray:
    joint_to_index = {name: idx for idx, name in enumerate(SIM_JOINT_NAMES)}
    return np.asarray([values[joint_to_index[name]] for name in REAL_JOINT_NAMES], dtype=np.float32)


def pretty_joint_name(name: str) -> str:
    return name.replace("left_", "")


def resolve_paths(args):
    cache_object_dir = args.asset_dir if args.asset_dir else args.object
    if args.source_grasps:
        source_path = Path(args.source_grasps).expanduser().resolve()
    else:
        source_path = (
            Path(args.cache_root).expanduser().resolve()
            / args.hand
            / cache_object_dir
            / args.instance_id
            / args.group
            / "valid_grasps.npy"
        )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = (
            Path(args.cache_root).expanduser().resolve()
            / args.hand
            / cache_object_dir
            / args.instance_id
            / "selected_grasps.npy"
        )

    return source_path, output_path


def main():
    args = parse_args()
    source_path, output_path = resolve_paths(args)

    if not source_path.exists():
        raise FileNotFoundError(f"Source grasp file not found: {source_path}")

    grasps = np.load(source_path, allow_pickle=False)
    select_idx = np.asarray(args.select_idx, dtype=np.int64)
    if select_idx.ndim != 1 or select_idx.size == 0:
        raise ValueError("--select-idx must contain at least one index.")
    if np.any(select_idx < 0) or np.any(select_idx >= grasps.shape[0]):
        raise IndexError(
            f"--select-idx contains out-of-range values for source grasp shape {grasps.shape}."
        )

    selected = grasps[select_idx]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, selected)

    print(f"Loaded source grasps from {source_path}")
    print(f"Saved {selected.shape[0]} selected grasps to {output_path}")

    for grasp_idx, grasp_row in zip(select_idx.tolist(), selected):
        hand_qpos_sim = np.asarray(grasp_row[: args.num_hand_dofs], dtype=np.float32)
        hand_qpos_real = sim_to_real_joint_order(hand_qpos_sim)
        print(f"selected grasp row {grasp_idx}")
        for joint_name, joint_value in zip(REAL_JOINT_NAMES, hand_qpos_real):
            print(f"{pretty_joint_name(joint_name):<16} {joint_value / np.pi * 180.0:8.3f} deg")


if __name__ == "__main__":
    main()
