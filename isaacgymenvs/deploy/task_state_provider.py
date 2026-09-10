from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from isaacgymenvs.deploy.real_robot_policy_api import HandAPI, TaskResetState, TaskStateProvider


DEFAULT_SIM_HAND_BASE_POS_WORLD = np.array([0.0, 0.0, 0.5], dtype=np.float32)
DEFAULT_SIM_HAND_BASE_ROT_WORLD = np.array([0.5, -0.5, 0.5, 0.5], dtype=np.float32)
DEFAULT_ROLLOUT_START_DELAY_SEC = 5.0


def _optional_vector(value: Any, dim: int, name: str):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {array.shape}")
    return array


def _vector_diff_stats(lhs, rhs):
    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    diff = lhs - rhs
    return {
        "l2_norm": float(np.linalg.norm(diff)),
        "max_abs": float(np.max(np.abs(diff))) if diff.size > 0 else 0.0,
    }


def _load_object_bbx(object_dir_name: str, instance_id: str):
    bbx_path = Path(__file__).resolve().parents[2] / "assets" / "objects" / str(object_dir_name) / "bbx.json"
    if not bbx_path.exists():
        return None
    data = json.loads(bbx_path.read_text())
    bbx_value = data.get(str(instance_id))
    if bbx_value is None:
        return None
    bbx = np.asarray(bbx_value, dtype=np.float32)
    if bbx.shape != (3,):
        raise ValueError(f"bbx.json entry for {object_dir_name}/{instance_id} must have shape (3,), got {bbx.shape}")
    return bbx


def _load_link_bbx(object_dir_name: str, instance_id: str):
    lbx_path = Path(__file__).resolve().parents[2] / "assets" / "objects" / str(object_dir_name) / "lbx.json"
    if not lbx_path.exists():
        return None, None
    data = json.loads(lbx_path.read_text())
    lbx_value = data.get(str(instance_id))
    if lbx_value is None:
        return None, None
    lbx = np.asarray(lbx_value, dtype=np.float32)
    if lbx.shape != (6,):
        raise ValueError(f"lbx.json entry for {object_dir_name}/{instance_id} must have shape (6,), got {lbx.shape}")
    return lbx[:3].copy(), lbx[3:].copy()


def _quat_conjugate(quat):
    quat = np.asarray(quat, dtype=np.float32)
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float32)


def _quat_multiply(lhs, rhs):
    x1, y1, z1, w1 = np.asarray(lhs, dtype=np.float32)
    x2, y2, z2, w2 = np.asarray(rhs, dtype=np.float32)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def _quat_apply(quat, vec):
    vec_quat = np.concatenate([np.asarray(vec, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    rotated = _quat_multiply(_quat_multiply(quat, vec_quat), _quat_conjugate(quat))
    return rotated[:3]


def _world_positions_to_hand_base(world_positions, hand_base_pos_world, hand_base_rot_world):
    world_positions = np.asarray(world_positions, dtype=np.float32)
    base_pos = np.asarray(hand_base_pos_world, dtype=np.float32)
    base_rot_inv = _quat_conjugate(hand_base_rot_world)
    flat = world_positions.reshape(-1, 3)
    local = np.stack([_quat_apply(base_rot_inv, pos - base_pos) for pos in flat], axis=0)
    return local.reshape(world_positions.shape)


def _world_pose_to_hand_base(world_pose, hand_base_pos_world, hand_base_rot_world):
    world_pose = np.asarray(world_pose, dtype=np.float32)
    base_pos = np.asarray(hand_base_pos_world, dtype=np.float32)
    base_rot_inv = _quat_conjugate(hand_base_rot_world)
    local_pos = _quat_apply(base_rot_inv, world_pose[:3] - base_pos)
    local_rot = _quat_multiply(base_rot_inv, world_pose[3:7])
    return np.concatenate([local_pos, local_rot], dtype=np.float32)


class StaticTaskStateProvider(TaskStateProvider):
    def __init__(
        self,
        init_object_pos=None,
        init_object_rot=None,
        goal_offset: float = 0.0,
        init_hand_qpos=None,
        init_fingertip_pos=None,
    ) -> None:
        self.init_object_pos = np.asarray(
            init_object_pos if init_object_pos is not None else [0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self.init_object_rot = np.asarray(
            init_object_rot if init_object_rot is not None else [0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        self.goal_offset = float(goal_offset)
        self.init_hand_qpos = _optional_vector(init_hand_qpos, 22, "init_hand_qpos")
        self.init_fingertip_pos = _optional_vector(init_fingertip_pos, 15, "init_fingertip_pos")

    def get_reset_state(self, robot: HandAPI) -> TaskResetState:
        return TaskResetState(
            init_object_pos=self.init_object_pos.copy(),
            init_object_rot=self.init_object_rot.copy(),
            goal_offset=self.goal_offset,
            init_hand_qpos=None if self.init_hand_qpos is None else self.init_hand_qpos.copy(),
            init_fingertip_pos=None if self.init_fingertip_pos is None else self.init_fingertip_pos.copy(),
        )


class ExampleTaskStateProvider(TaskStateProvider):
    """Edit or subclass this to define reset-time object pose, init grasp, and goal logic."""

    def get_reset_state(self, robot: HandAPI) -> TaskResetState:
        return TaskResetState(
            init_object_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            init_object_rot=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            goal_offset=0.0,
            init_hand_qpos=None,
            init_fingertip_pos=None,
            metadata={"note": "Fill these values with your real deployment setup."},
        )

    def get_goal_offset(self, robot: HandAPI, reset_state: TaskResetState) -> float:
        return float(reset_state.goal_offset)


class GoalOnlyTaskStateProvider(TaskStateProvider):
    """
    Real-world grasp workflow:
    1. Move the hand to a desired grasp pose.
    2. Insert the object manually.
    3. Press Enter to capture init hand/contact/fingertip observations.
    4. Press Enter again to start policy rollout.

    Override derive_init_object_pose/rot if you want init object pose to be
    estimated from the captured grasp state.
    """

    def __init__(
        self,
        goal_offset: float = 0.0,
        capture_prompt: str = "Place the hand in the training/test grasp, insert the object, then press Enter to capture init observations...",
        start_prompt: str = "Init observations captured. Press Enter to start rollout...",
        start_rollout_delay_sec: float = DEFAULT_ROLLOUT_START_DELAY_SEC,
        init_object_pos=None,
        init_object_rot=None,
        init_hand_qpos=None,
        init_fingertip_pos=None,
    ) -> None:
        self.goal_offset = float(goal_offset)
        self.capture_prompt = capture_prompt
        self.start_prompt = start_prompt
        self.start_rollout_delay_sec = float(start_rollout_delay_sec)
        self.init_object_pos_override = _optional_vector(init_object_pos, 3, "init_object_pos")
        self.init_object_rot_override = _optional_vector(init_object_rot, 4, "init_object_rot")
        self.init_hand_qpos_override = _optional_vector(init_hand_qpos, 22, "init_hand_qpos")
        self.init_fingertip_pos_override = _optional_vector(init_fingertip_pos, 15, "init_fingertip_pos")

    def before_reset_capture(self, robot: HandAPI) -> None:
        input(self.capture_prompt)

    def derive_init_object_pose(
        self,
        robot: HandAPI,
        measured_hand_qpos,
        measured_fingertip_pos,
    ) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def derive_init_object_rot(
        self,
        robot: HandAPI,
        measured_hand_qpos,
        measured_fingertip_pos,
    ) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def build_reset_state(
        self,
        robot: HandAPI,
        measured_hand_qpos,
        measured_fingertip_pos,
    ) -> TaskResetState:
        if self.init_object_pos_override is not None:
            init_object_pos = self.init_object_pos_override.copy()
        else:
            init_object_pos = np.asarray(
                self.derive_init_object_pose(robot, measured_hand_qpos, measured_fingertip_pos),
                dtype=np.float32,
            )
        if self.init_object_rot_override is not None:
            init_object_rot = self.init_object_rot_override.copy()
        else:
            init_object_rot = np.asarray(
                self.derive_init_object_rot(robot, measured_hand_qpos, measured_fingertip_pos),
                dtype=np.float32,
            )
        if init_object_pos.shape != (3,):
            raise ValueError(f"derive_init_object_pose must return shape (3,), got {init_object_pos.shape}")
        if init_object_rot.shape != (4,):
            raise ValueError(f"derive_init_object_rot must return shape (4,), got {init_object_rot.shape}")

        return TaskResetState(
            init_object_pos=init_object_pos,
            init_object_rot=init_object_rot,
            goal_offset=self.goal_offset,
            init_hand_qpos=(
                np.asarray(measured_hand_qpos, dtype=np.float32).copy()
                if self.init_hand_qpos_override is None
                else self.init_hand_qpos_override.copy()
            ),
            init_fingertip_pos=(
                np.asarray(measured_fingertip_pos, dtype=np.float32).copy()
                if self.init_fingertip_pos_override is None
                else self.init_fingertip_pos_override.copy()
            ),
        )

    def before_rollout_start(self, robot: HandAPI, reset_state: TaskResetState) -> None:
        input(self.start_prompt)
        if self.start_rollout_delay_sec > 0.0:
            print(f"Starting rollout in {self.start_rollout_delay_sec:.1f}s...")
            time.sleep(self.start_rollout_delay_sec)


class GraspCacheTaskStateProvider(TaskStateProvider):
    """
    Load init grasp/object/contact/fingertip state from the same cache files used by artmanip.py.

    Example:
      instance_id='000'
    selects the only row from:
      caches/initial_grasp/sharpa/<asset_dir_or_object>/000/selected_grasps.npy
    """

    def __init__(
        self,
        hand_type: str = "sharpa",
        object_type: str = "knife",
        asset_dir: str = "",
        instance_id: str = "000",
        cache_root: str = "caches/initial_grasp",
        goal_offset: float = 0.0,
        init_object_pos=None,
        init_object_rot=None,
        init_hand_qpos=None,
        init_fingertip_pos=None,
        command_cached_grasp: bool = True,
        cached_grasp_settle_sec: float = 1.0,
        cached_pose_frame: str = "auto",
        hand_base_pos_world=None,
        hand_base_rot_world=None,
        load_grasp_prompt: str = "Hand is ready. Press Enter to load the cached grasp pose...",
        capture_prompt: str = "Cached grasp loaded. Insert the object, then press Enter to continue...",
        start_prompt: str = "Cached init state loaded. Press Enter to start rollout...",
        start_rollout_delay_sec: float = DEFAULT_ROLLOUT_START_DELAY_SEC,
        use_measured_init_hand_qpos: bool = False,
        use_measured_init_fingertip_pos: bool = False,
    ) -> None:
        self.hand_type = str(hand_type)
        self.object_type = str(object_type)
        self.asset_dir = str(asset_dir).strip()
        self.object_dir_name = self.asset_dir if self.asset_dir else self.object_type
        self.instance_id = str(instance_id)
        self.cache_root = cache_root
        self.goal_offset = float(goal_offset)
        self.init_object_pos_override = _optional_vector(init_object_pos, 3, "init_object_pos")
        self.init_object_rot_override = _optional_vector(init_object_rot, 4, "init_object_rot")
        self.init_hand_qpos_override = _optional_vector(init_hand_qpos, 22, "init_hand_qpos")
        self.init_fingertip_pos_override = _optional_vector(init_fingertip_pos, 15, "init_fingertip_pos")
        self.command_cached_grasp = bool(command_cached_grasp)
        self.cached_grasp_settle_sec = float(cached_grasp_settle_sec)
        self.cached_pose_frame = str(cached_pose_frame)
        self.hand_base_pos_world = _optional_vector(
            hand_base_pos_world if hand_base_pos_world is not None else DEFAULT_SIM_HAND_BASE_POS_WORLD,
            3,
            "hand_base_pos_world",
        )
        self.hand_base_rot_world = _optional_vector(
            hand_base_rot_world if hand_base_rot_world is not None else DEFAULT_SIM_HAND_BASE_ROT_WORLD,
            4,
            "hand_base_rot_world",
        )
        self.load_grasp_prompt = load_grasp_prompt
        self.capture_prompt = capture_prompt
        self.start_prompt = start_prompt
        self.start_rollout_delay_sec = float(start_rollout_delay_sec)
        self.use_measured_init_hand_qpos = bool(use_measured_init_hand_qpos)
        self.use_measured_init_fingertip_pos = bool(use_measured_init_fingertip_pos)
        self._selected_grasp = None
        self.object_bbx = _load_object_bbx(self.object_dir_name, self.instance_id)
        self.link0_bbx, self.link1_bbx = _load_link_bbx(self.object_dir_name, self.instance_id)

    def _resolve_cache_path(self):
        base_dir = os.path.join(self.cache_root, self.hand_type, self.object_dir_name, self.instance_id)
        path = os.path.join(base_dir, "selected_grasps.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Grasp cache not found: {path}")
        return path

    def _load_selected_grasp(self):
        cache_path = self._resolve_cache_path()
        grasp_states = np.load(cache_path)
        if grasp_states.ndim != 2:
            raise ValueError(f"Expected cached grasp states to be 2D, got {grasp_states.shape}")
        if grasp_states.shape[0] != 1:
            raise ValueError(
                f"selected grasp cache {cache_path} must contain exactly one row, got {grasp_states.shape[0]}."
            )
        selected = self._unpack_grasp_state_row(grasp_states[0], hand_dof_dim=self.context.hand_dof_dim)
        self._selected_grasp = {
            "cache_path": cache_path,
            "grasp_pool": "selected",
            "state": self._convert_cached_state_to_policy_frame(selected),
        }
        return self._selected_grasp

    def _resolve_cached_pose_frame(self, cache_path):
        if self.cached_pose_frame != "auto":
            return self.cached_pose_frame
        metadata_path = Path(cache_path).resolve().parent / "grasp_state_metadata.json"
        if not metadata_path.exists():
            return "sim_world"
        metadata = json.loads(metadata_path.read_text())
        return str(metadata.get("pose_frame", "sim_world")).strip().lower()

    def _convert_cached_state_to_policy_frame(self, state):
        converted = {
            key: np.asarray(value, dtype=np.float32).copy() if isinstance(value, np.ndarray) else value
            for key, value in state.items()
        }
        pose_frame = self._resolve_cached_pose_frame(self._resolve_cache_path())
        if pose_frame == "hand_base":
            return converted
        if pose_frame != "sim_world":
            raise ValueError(
                f"Unsupported cached_pose_frame '{pose_frame}'. Use 'auto', 'sim_world', or 'hand_base'."
            )

        converted_object_pose = _world_pose_to_hand_base(
            converted["object_pose"],
            hand_base_pos_world=self.hand_base_pos_world,
            hand_base_rot_world=self.hand_base_rot_world,
        )
        converted["object_pose"] = converted_object_pose
        if "link1_pose" in converted:
            converted["link1_pose"] = _world_pose_to_hand_base(
                converted["link1_pose"],
                hand_base_pos_world=self.hand_base_pos_world,
                hand_base_rot_world=self.hand_base_rot_world,
            )

        fingertip_world = np.asarray(converted["fingertip_pos"], dtype=np.float32).reshape(-1, 3)
        converted["fingertip_pos"] = _world_positions_to_hand_base(
            fingertip_world,
            hand_base_pos_world=self.hand_base_pos_world,
            hand_base_rot_world=self.hand_base_rot_world,
        ).reshape(-1)
        return converted

    def before_reset_capture(self, robot: HandAPI) -> None:
        selected_grasp = self._load_selected_grasp()
        state = selected_grasp["state"]
        print(
            f"Using cached grasp {selected_grasp['grasp_pool']}[0] from "
            f"{self.hand_type}/{self.object_dir_name}/{self.instance_id}"
        )

        if self.command_cached_grasp:
            input(self.load_grasp_prompt)
            cached_qpos = state["hand_dof_pos"]
            robot.command_init_grasp(
                np.asarray(cached_qpos, dtype=np.float32),
                final_settle_sec=self.cached_grasp_settle_sec,
            )

        input(self.capture_prompt)

    @staticmethod
    def _unpack_grasp_state_row(grasp_row: np.ndarray, hand_dof_dim: int):
        grasp_row = np.asarray(grasp_row, dtype=np.float32).reshape(-1)
        fingertip_dim = 15
        contact_dim = 5
        min_full_dim = 2 * hand_dof_dim + 7 + 7 + fingertip_dim + contact_dim
        if grasp_row.shape[0] < hand_dof_dim + 7:
            raise ValueError(
                f"Invalid grasp row width {grasp_row.shape[0]}; expected at least {hand_dof_dim + 7}"
            )

        if grasp_row.shape[0] >= min_full_dim:
            object_dof_dim = grasp_row.shape[0] - min_full_dim
            if object_dof_dim < 0 or object_dof_dim > 8:
                raise ValueError(
                    f"Invalid selected grasp width {grasp_row.shape[0]}; expected current layout "
                    f"[hand_pos, hand_target, object_pose, link1_pose, object_dof, fingertip_pos, contact_bits]."
                )
            cursor = 0
            hand_dof_pos = grasp_row[cursor:cursor + hand_dof_dim]
            cursor += hand_dof_dim
            hand_dof_target = grasp_row[cursor:cursor + hand_dof_dim]
            cursor += hand_dof_dim
            object_pose = grasp_row[cursor:cursor + 7]
            cursor += 7
            link1_pose = grasp_row[cursor:cursor + 7]
            cursor += 7
            object_dof_pos = grasp_row[cursor:cursor + object_dof_dim]
            cursor += object_dof_dim
            fingertip_pos = grasp_row[cursor:cursor + fingertip_dim]
            cursor += fingertip_dim
            contact_info = grasp_row[cursor:cursor + contact_dim]
            return {
                "hand_dof_pos": hand_dof_pos,
                "hand_dof_target": hand_dof_target,
                "object_pose": object_pose,
                "link1_pose": link1_pose,
                "object_dof_pos": object_dof_pos,
                "fingertip_pos": fingertip_pos,
                "contact_info": contact_info,
            }

        raise ValueError(
            f"Invalid selected grasp width {grasp_row.shape[0]}; regenerate selected_grasps.npy with the current layout."
        )

    def build_reset_state(
        self,
        robot: HandAPI,
        measured_hand_qpos,
        measured_fingertip_pos,
    ) -> TaskResetState:
        selected_grasp = self._selected_grasp if self._selected_grasp is not None else self._load_selected_grasp()
        selected = selected_grasp["state"]
        measured_hand_qpos = np.asarray(measured_hand_qpos, dtype=np.float32).copy()
        measured_fingertip_pos = np.asarray(measured_fingertip_pos, dtype=np.float32).copy()
        cached_hand_qpos = np.asarray(selected["hand_dof_pos"], dtype=np.float32).copy()
        cached_fingertip_pos = np.asarray(selected["fingertip_pos"], dtype=np.float32).copy()

        if self.init_hand_qpos_override is not None:
            init_hand_qpos = self.init_hand_qpos_override.copy()
            init_hand_qpos_source = "override"
        elif self.use_measured_init_hand_qpos:
            init_hand_qpos = measured_hand_qpos
            init_hand_qpos_source = "measured"
        else:
            init_hand_qpos = cached_hand_qpos
            init_hand_qpos_source = "cached"

        if self.init_fingertip_pos_override is not None:
            init_fingertip_pos = self.init_fingertip_pos_override.copy()
            init_fingertip_pos_source = "override"
        elif self.use_measured_init_fingertip_pos:
            init_fingertip_pos = measured_fingertip_pos
            init_fingertip_pos_source = "measured"
        else:
            init_fingertip_pos = cached_fingertip_pos
            init_fingertip_pos_source = "cached"

        reset_debug = {
            "goal_offset": float(self.goal_offset),
            "init_sources": {
                "object_pos": "override" if self.init_object_pos_override is not None else "cached",
                "object_rot": "override" if self.init_object_rot_override is not None else "cached",
                "hand_qpos": init_hand_qpos_source,
                "fingertip_pos": init_fingertip_pos_source,
            },
            "cached_vs_measured": {
                "hand_qpos": _vector_diff_stats(cached_hand_qpos, measured_hand_qpos),
                "hand_qpos_values": {
                    "cached": cached_hand_qpos.astype(np.float32).tolist(),
                    "measured": measured_hand_qpos.astype(np.float32).tolist(),
                    "diff": (cached_hand_qpos - measured_hand_qpos).astype(np.float32).tolist(),
                },
                "fingertip_pos": _vector_diff_stats(cached_fingertip_pos, measured_fingertip_pos),
                "fingertip_pos_values": {
                    "cached": cached_fingertip_pos.astype(np.float32).tolist(),
                    "measured": measured_fingertip_pos.astype(np.float32).tolist(),
                    "diff": (cached_fingertip_pos - measured_fingertip_pos).astype(np.float32).tolist(),
                },
            },
        }
        return TaskResetState(
            init_object_pos=(
                np.asarray(selected["object_pose"][:3], dtype=np.float32).copy()
                if self.init_object_pos_override is None
                else self.init_object_pos_override.copy()
            ),
            init_object_rot=(
                np.asarray(selected["object_pose"][3:7], dtype=np.float32).copy()
                if self.init_object_rot_override is None
                else self.init_object_rot_override.copy()
            ),
            goal_offset=self.goal_offset,
            init_hand_qpos=init_hand_qpos,
            init_fingertip_pos=init_fingertip_pos,
            metadata={
                "cache_path": selected_grasp["cache_path"],
                "grasp_pool": selected_grasp["grasp_pool"],
                "object_bbx": None if self.object_bbx is None else self.object_bbx.astype(np.float32).tolist(),
                "init_link1_pose": np.asarray(selected["link1_pose"], dtype=np.float32).tolist(),
                "link0_bbx": None if self.link0_bbx is None else self.link0_bbx.astype(np.float32).tolist(),
                "link1_bbx": None if self.link1_bbx is None else self.link1_bbx.astype(np.float32).tolist(),
                "reset_debug": reset_debug,
            },
        )

    def get_goal_offset(self, robot: HandAPI, reset_state: TaskResetState) -> float:
        return float(self.goal_offset)

    def before_rollout_start(self, robot: HandAPI, reset_state: TaskResetState) -> None:
        input(self.start_prompt)
        if self.start_rollout_delay_sec > 0.0:
            print(f"Starting rollout in {self.start_rollout_delay_sec:.1f}s...")
            time.sleep(self.start_rollout_delay_sec)
