from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isaacgymenvs.deploy.real_robot_policy_api import (
    HandAPI,
    RobotObservations,
    TaskObservationProvider,
    TaskResetState,
)
from isaacgymenvs.deploy.obs_layout import (
    DEFAULT_STUDENT_INIT_OBS_DIM,
    STUDENT_TEMPORAL_OBS_MODE,
    get_student_temporal_obs_layout,
    infer_student_temporal_obs_layout,
    validate_student_temporal_obs_mode,
)
from isaacgymenvs.deploy.sharpa.robot import SIM_JOINT_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF_PATH = REPO_ROOT / "assets" / "hands" / "sharpa" / "left.urdf"
DEFAULT_FINGERTIP_LINKS = [
    "left_thumb_fingertip",
    "left_index_fingertip",
    "left_middle_fingertip",
    "left_ring_fingertip",
    "left_pinky_fingertip",
]
FINGERTIP_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def _rotation_from_rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def _transform_from_xyz_rpy(xyz, rpy):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = _rotation_from_rpy(rpy)
    transform[:3, 3] = xyz
    return transform


def _transform_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(4, dtype=np.float32)
    axis = axis / norm
    x, y, z = axis
    c, s = math.cos(float(angle)), math.sin(float(angle))
    one_c = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    return transform


@dataclass
class _URDFJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


class _SharpaForwardKinematics:
    def __init__(self, urdf_path: str | Path, fingertip_links: list[str] | None = None):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"Sharpa URDF not found: {self.urdf_path}")
        self.fingertip_links = fingertip_links or list(DEFAULT_FINGERTIP_LINKS)
        self.joints_by_child: dict[str, _URDFJoint] = {}
        self.revolute_joint_names: list[str] = []
        self._chains: dict[str, list[_URDFJoint]] = {}
        self._parse_urdf()

    def _parse_urdf(self):
        root = ET.parse(self.urdf_path).getroot()
        for joint_elem in root.findall("joint"):
            joint_type = joint_elem.attrib["type"]
            name = joint_elem.attrib["name"]
            parent = joint_elem.find("parent").attrib["link"]
            child = joint_elem.find("child").attrib["link"]
            origin_elem = joint_elem.find("origin")
            xyz = np.fromstring(origin_elem.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float32) if origin_elem is not None else np.zeros(3, dtype=np.float32)
            rpy = np.fromstring(origin_elem.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float32) if origin_elem is not None else np.zeros(3, dtype=np.float32)
            axis_elem = joint_elem.find("axis")
            axis = np.fromstring(axis_elem.attrib.get("xyz", "0 0 1"), sep=" ", dtype=np.float32) if axis_elem is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32)
            joint = _URDFJoint(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin_xyz=xyz,
                origin_rpy=rpy,
                axis=axis,
            )
            self.joints_by_child[child] = joint
            if joint_type != "fixed":
                self.revolute_joint_names.append(name)

        for fingertip_link in self.fingertip_links:
            chain = []
            current_link = fingertip_link
            while current_link in self.joints_by_child:
                joint = self.joints_by_child[current_link]
                chain.append(joint)
                current_link = joint.parent
            self._chains[fingertip_link] = list(reversed(chain))

    def fingertip_positions(self, joint_positions: np.ndarray) -> np.ndarray:
        joint_positions = np.asarray(joint_positions, dtype=np.float32)
        if joint_positions.shape != (len(self.revolute_joint_names),):
            raise ValueError(
                f"Expected {len(self.revolute_joint_names)} joint values for FK, got {joint_positions.shape}"
            )
        qpos_by_name = dict(zip(self.revolute_joint_names, joint_positions))
        fingertip_positions = []
        for fingertip_link in self.fingertip_links:
            transform = np.eye(4, dtype=np.float32)
            for joint in self._chains[fingertip_link]:
                transform = transform @ _transform_from_xyz_rpy(joint.origin_xyz, joint.origin_rpy)
                if joint.joint_type != "fixed":
                    transform = transform @ _transform_from_axis_angle(joint.axis, qpos_by_name[joint.name])
            fingertip_positions.append(transform[:3, 3].copy())
        return np.asarray(fingertip_positions, dtype=np.float32)

    def reorder_from_sim_joint_positions(self, sim_joint_positions: np.ndarray) -> np.ndarray:
        sim_joint_positions = np.asarray(sim_joint_positions, dtype=np.float32)
        if sim_joint_positions.shape != (len(SIM_JOINT_NAMES),):
            raise ValueError(
                f"Expected {len(SIM_JOINT_NAMES)} sim joint values for FK reorder, got {sim_joint_positions.shape}"
            )
        return np.asarray(
            [sim_joint_positions[SIM_JOINT_NAMES.index(joint_name)] for joint_name in self.revolute_joint_names],
            dtype=np.float32,
        )


class SharpaStaticTaskObservationProvider(TaskObservationProvider):
    uses_static_object_state = True

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_URDF_PATH,
        fingertip_links: list[str] | None = None,
        init_object_pos: list[float] | tuple[float, ...] | np.ndarray | None = None,
        init_object_rot: list[float] | tuple[float, ...] | np.ndarray | None = None,
        goal_offset: float = 0.0,
        object_size: list[float] | tuple[float, ...] | np.ndarray | None = None,
    ) -> None:
        self.default_init_object_pos = np.asarray(init_object_pos if init_object_pos is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self.default_init_object_rot = np.asarray(init_object_rot if init_object_rot is not None else [0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.default_goal_offset = float(goal_offset)
        self.object_size_override = None if object_size is None else np.asarray(object_size, dtype=np.float32)
        self.default_object_size = np.asarray(
            object_size if object_size is not None else [0.02, 0.016, 0.14],
            dtype=np.float32,
        )
        self.object_size = self.default_object_size.copy()
        self.default_link0_bbx = self.default_object_size.copy()
        self.default_link1_bbx = self.default_object_size.copy()
        self.fingertip_names = list(FINGERTIP_NAMES)
        self.fk = _SharpaForwardKinematics(urdf_path=urdf_path, fingertip_links=fingertip_links)
        self.history_len = None
        self.student_temporal_obs_mode = STUDENT_TEMPORAL_OBS_MODE
        self.proprio_dim_per_step = None
        self.contact_dim_per_step = None
        self.init_dim = None
        self.init_hand_qpos_norm = None
        self.init_link1_pose = None
        self.init_fingertip_pos = None
        self.init_link0_bbx = None
        self.init_link1_bbx = None
        self.proprio_history = None
        self.task_state_provider = None
        self.reset_state = None
        self.goal_offset = self.default_goal_offset
        self.reset_debug_info = {}
        self.reset_measurements = {}

    @staticmethod
    def _to_vector(value, dim: int, name: str):
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (dim,):
            raise ValueError(f"{name} must have shape ({dim},), got {array.shape}")
        return array

    def set_deployment_context(self, context) -> None:
        super().set_deployment_context(context)
        validate_student_temporal_obs_mode(context.student_temporal_obs_mode)
        if context.student_obs_dim:
            temporal_layout = infer_student_temporal_obs_layout(
                student_obs_dim=context.student_obs_dim,
                history_len=context.student_history_len,
                proprio_dim_per_step=context.student_proprio_dim_per_step,
                init_dim=context.student_init_dim or DEFAULT_STUDENT_INIT_OBS_DIM,
            )
        else:
            temporal_layout = get_student_temporal_obs_layout(
                history_len=context.student_history_len,
                proprio_dim_per_step=context.student_proprio_dim_per_step,
                init_dim=context.student_init_dim or DEFAULT_STUDENT_INIT_OBS_DIM,
            )
        self.history_len = int(temporal_layout["history_len"])
        self.student_temporal_obs_mode = STUDENT_TEMPORAL_OBS_MODE
        self.proprio_dim_per_step = int(temporal_layout["proprio_dim_per_step"])
        self.contact_dim_per_step = 0
        self.init_dim = int(temporal_layout["init_dim"])

    def _default_reset_state(self) -> TaskResetState:
        return TaskResetState(
            init_object_pos=self.default_init_object_pos.copy(),
            init_object_rot=self.default_init_object_rot.copy(),
            goal_offset=self.default_goal_offset,
        )

    @staticmethod
    def _normalize_joint_positions(qpos, lower, upper):
        denom = np.asarray(upper, dtype=np.float32) - np.asarray(lower, dtype=np.float32)
        denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
        return (2.0 * (np.asarray(qpos, dtype=np.float32) - np.asarray(lower, dtype=np.float32)) / denom) - 1.0

    def _build_init_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.init_hand_qpos_norm,
                self.reset_state.init_object_pos,
                self.reset_state.init_object_rot,
                self.init_link1_pose,
                self.init_fingertip_pos,
                self.init_link0_bbx,
                self.init_link1_bbx,
            ],
            dtype=np.float32,
        )

    def _fingertip_position_labels(self, prefix: str) -> list[str]:
        labels = []
        for finger_name in FINGERTIP_NAMES:
            labels.extend(
                [
                    f"{prefix}/{finger_name}_x",
                    f"{prefix}/{finger_name}_y",
                    f"{prefix}/{finger_name}_z",
                ]
            )
        return labels

    def _init_obs_labels(self) -> list[str]:
        return (
            [f"init_hand_qpos_norm/{joint_name}" for joint_name in SIM_JOINT_NAMES]
            + ["init_object_pos/x", "init_object_pos/y", "init_object_pos/z"]
            + ["init_object_rot/x", "init_object_rot/y", "init_object_rot/z", "init_object_rot/w"]
            + ["init_link1_pose/x", "init_link1_pose/y", "init_link1_pose/z", "init_link1_pose/qx", "init_link1_pose/qy", "init_link1_pose/qz", "init_link1_pose/qw"]
            + self._fingertip_position_labels("init_fingertip_pos")
            + ["init_link0_bbx/x", "init_link0_bbx/y", "init_link0_bbx/z"]
            + ["init_link1_bbx/x", "init_link1_bbx/y", "init_link1_bbx/z"]
        )

    def _proprio_step_labels(self) -> list[str]:
        return [f"hand_qpos_norm/{joint_name}" for joint_name in SIM_JOINT_NAMES] + [
            f"last_action/{joint_name}" for joint_name in SIM_JOINT_NAMES
        ]

    def export_reset_snapshot(self) -> dict:
        if self.reset_state is None or not self.reset_measurements:
            return {}

        init_obs = self._build_init_obs().astype(np.float32, copy=True)
        measured_qpos_norm = np.asarray(self.reset_measurements["measured_hand_qpos_norm"], dtype=np.float32)
        zero_action = np.zeros(self.context.action_dim, dtype=np.float32)
        measured_fingertip_pos = np.asarray(self.reset_measurements["measured_fingertip_pos"], dtype=np.float32)
        policy_obs = np.concatenate(
            [
                init_obs,
                measured_qpos_norm,
                zero_action,
                np.asarray([self.goal_offset], dtype=np.float32),
                measured_fingertip_pos,
            ],
            dtype=np.float32,
        )
        student_obs = np.concatenate(
            [self.proprio_history.reshape(-1).astype(np.float32, copy=True), init_obs],
            dtype=np.float32,
        )

        student_obs_labels = []
        proprio_labels = self._proprio_step_labels()
        for hist_idx in range(self.history_len):
            student_obs_labels.extend([label.replace("/", f"/{hist_idx}/", 1) for label in proprio_labels])
        init_obs_labels = self._init_obs_labels()
        student_obs_labels.extend(init_obs_labels)

        policy_obs_labels = (
            init_obs_labels
            + [f"hand_qpos_norm/{joint_name}" for joint_name in SIM_JOINT_NAMES]
            + [f"last_action/{joint_name}" for joint_name in SIM_JOINT_NAMES]
            + ["goal_offset"]
            + self._fingertip_position_labels("current_fingertip_pos")
        )

        return {
            "reset_debug": dict(self.reset_debug_info),
            "student_temporal_obs_mode": self.student_temporal_obs_mode,
            "history_len": int(self.history_len),
            "measurements": {
                "measured_hand_qpos": np.asarray(self.reset_measurements["measured_hand_qpos"], dtype=np.float32),
                "measured_hand_qpos_norm": measured_qpos_norm,
                "measured_fingertip_pos": measured_fingertip_pos,
            },
            "init_state": {
                "init_hand_qpos_norm": np.asarray(self.init_hand_qpos_norm, dtype=np.float32),
                "init_link1_pose": np.asarray(self.init_link1_pose, dtype=np.float32),
                "init_fingertip_pos": np.asarray(self.init_fingertip_pos, dtype=np.float32),
                "init_link0_bbx": np.asarray(self.init_link0_bbx, dtype=np.float32),
                "init_link1_bbx": np.asarray(self.init_link1_bbx, dtype=np.float32),
                "goal_offset": float(self.goal_offset),
            },
            "histories": {
                "proprio_history": np.asarray(self.proprio_history, dtype=np.float32),
            },
            "vectors": {
                "init_obs": init_obs,
                "policy_obs_reset": policy_obs,
                "student_obs_reset": student_obs,
            },
            "labels": {
                "init_obs": init_obs_labels,
                "policy_obs_reset": policy_obs_labels,
                "student_obs_reset": student_obs_labels,
            },
        }

    def _fingertip_positions_from_sim_qpos(self, sim_qpos: np.ndarray) -> np.ndarray:
        fk_joint_positions = self.fk.reorder_from_sim_joint_positions(sim_qpos)
        return self.fk.fingertip_positions(fk_joint_positions).reshape(-1).astype(np.float32)

    def reset(self, robot: HandAPI) -> None:
        if self.task_state_provider is not None:
            self.task_state_provider.reset(robot)
            self.task_state_provider.before_reset_capture(robot)

        measured_qpos = np.asarray(robot.get_hand_joint_positions(), dtype=np.float32)
        measured_fingertip_pos = self._fingertip_positions_from_sim_qpos(measured_qpos)

        if self.task_state_provider is not None:
            self.reset_state = self.task_state_provider.build_reset_state(
                robot,
                measured_qpos,
                measured_fingertip_pos,
            )
        else:
            self.reset_state = self._default_reset_state()
        self.reset_debug_info = dict(self.reset_state.metadata.get("reset_debug", {})) if self.reset_state.metadata else {}
        self.reset_debug_info.setdefault("goal_offset", float(self.reset_state.goal_offset))
        metadata_object_bbx = None if not self.reset_state.metadata else self.reset_state.metadata.get("object_bbx")
        metadata_link0_bbx = None if not self.reset_state.metadata else self.reset_state.metadata.get("link0_bbx")
        metadata_link1_bbx = None if not self.reset_state.metadata else self.reset_state.metadata.get("link1_bbx")
        metadata_init_link1_pose = None if not self.reset_state.metadata else self.reset_state.metadata.get("init_link1_pose")
        if self.object_size_override is not None:
            self.object_size = self.object_size_override.astype(np.float32, copy=True)
            self.reset_debug_info.setdefault("object_bbx_source", "override")
        elif metadata_object_bbx is not None:
            object_bbx = np.asarray(metadata_object_bbx, dtype=np.float32)
            if object_bbx.shape != (3,):
                raise ValueError(f"reset_state.metadata['object_bbx'] must have shape (3,), got {object_bbx.shape}")
            self.object_size = object_bbx
            self.reset_debug_info.setdefault("object_bbx_source", "instance_bbx")
        else:
            self.object_size = self.default_object_size.copy()
            self.reset_debug_info.setdefault("object_bbx_source", "default")
        if metadata_link0_bbx is not None:
            self.init_link0_bbx = self._to_vector(metadata_link0_bbx, 3, "metadata.link0_bbx")
        else:
            self.init_link0_bbx = self.object_size.astype(np.float32, copy=True)
        if metadata_link1_bbx is not None:
            self.init_link1_bbx = self._to_vector(metadata_link1_bbx, 3, "metadata.link1_bbx")
        else:
            self.init_link1_bbx = self.object_size.astype(np.float32, copy=True)
        if metadata_init_link1_pose is not None:
            self.init_link1_pose = self._to_vector(metadata_init_link1_pose, 7, "metadata.init_link1_pose")
        else:
            self.init_link1_pose = np.zeros(7, dtype=np.float32)

        init_hand_qpos = measured_qpos
        if self.reset_state.init_hand_qpos is not None:
            init_hand_qpos = self._to_vector(self.reset_state.init_hand_qpos, self.context.hand_dof_dim, "init_hand_qpos")

        self.init_hand_qpos_norm = self._normalize_joint_positions(
            init_hand_qpos,
            self.context.hand_dof_lower_limits,
            self.context.hand_dof_upper_limits,
        )
        if self.reset_state.init_fingertip_pos is not None:
            self.init_fingertip_pos = self._to_vector(self.reset_state.init_fingertip_pos, 15, "init_fingertip_pos")
        else:
            self.init_fingertip_pos = measured_fingertip_pos
        self.goal_offset = float(self.reset_state.goal_offset)
        zero_action = np.zeros(self.context.action_dim, dtype=np.float32)
        measured_qpos_norm = self._normalize_joint_positions(
            measured_qpos,
            self.context.hand_dof_lower_limits,
            self.context.hand_dof_upper_limits,
        )
        proprio = np.concatenate([measured_qpos_norm, zero_action], dtype=np.float32)
        self.reset_measurements = {
            "measured_hand_qpos": measured_qpos.astype(np.float32, copy=True),
            "measured_hand_qpos_norm": measured_qpos_norm.astype(np.float32, copy=True),
            "measured_fingertip_pos": measured_fingertip_pos.astype(np.float32, copy=True),
        }
        self.proprio_history = np.repeat(proprio[None, :], self.history_len, axis=0)
        if self.task_state_provider is not None:
            self.task_state_provider.before_rollout_start(robot, self.reset_state)

    def build_observations(self, robot: HandAPI, last_action) -> RobotObservations:
        if self.proprio_history is None:
            self.reset(robot)

        qpos = np.asarray(robot.get_hand_joint_positions(), dtype=np.float32)
        qpos_norm = self._normalize_joint_positions(
            qpos,
            self.context.hand_dof_lower_limits,
            self.context.hand_dof_upper_limits,
        )
        last_action = np.asarray(last_action, dtype=np.float32)
        if last_action.shape != (self.context.action_dim,):
            raise ValueError(f"Expected last_action shape ({self.context.action_dim},), got {last_action.shape}")

        fingertip_pos = self._fingertip_positions_from_sim_qpos(qpos)
        if self.task_state_provider is not None and self.reset_state is not None:
            self.goal_offset = float(self.task_state_provider.get_goal_offset(robot, self.reset_state))

        proprio = np.concatenate([qpos_norm, last_action], dtype=np.float32)
        self.proprio_history[:-1] = self.proprio_history[1:]
        self.proprio_history[-1] = proprio

        init_obs = self._build_init_obs()
        policy_obs = np.concatenate(
            [
                init_obs,
                qpos_norm,
                last_action,
                np.asarray([self.goal_offset], dtype=np.float32),
                fingertip_pos,
            ],
            dtype=np.float32,
        )
        student_obs = np.concatenate([self.proprio_history.reshape(-1), init_obs], dtype=np.float32)
        if policy_obs.shape != (self.context.policy_obs_dim,):
            raise ValueError(f"Expected policy_obs_dim={self.context.policy_obs_dim}, got {policy_obs.shape}")
        if student_obs.shape != (self.context.student_obs_dim,):
            raise ValueError(f"Expected student_obs_dim={self.context.student_obs_dim}, got {student_obs.shape}")

        expl_features = None
        if self.context.expl_feature_dim > 0:
            expl_features = np.zeros(self.context.expl_feature_dim, dtype=np.float32)

        debug_info = {
            "student_temporal_obs_mode": self.context.student_temporal_obs_mode,
            "reset_debug": self.reset_debug_info,
        }

        return RobotObservations(
            policy_obs=policy_obs,
            student_obs=student_obs,
            expl_features=expl_features,
            episode_done=False,
            debug_info=debug_info,
        )
