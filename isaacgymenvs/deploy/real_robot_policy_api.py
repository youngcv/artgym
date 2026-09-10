from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import time

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


if torch is None:
    ArrayLike = np.ndarray
else:
    ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class PolicyDeploymentContext:
    policy_obs_dim: int
    student_obs_dim: int
    privileged_obs_dim: int
    expl_feature_dim: int
    action_dim: int
    hand_dof_dim: int
    control_dt: float
    use_relative_control: bool
    action_moving_average: float
    hand_dof_speed_scale: float
    hand_dof_lower_limits: np.ndarray
    hand_dof_upper_limits: np.ndarray
    student_temporal_obs_mode: str = ""
    student_history_len: int = 1
    student_proprio_dim_per_step: int = 0
    student_contact_dim_per_step: int = 0
    student_init_dim: int = 0
    distill_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotObservations:
    policy_obs: ArrayLike
    student_obs: ArrayLike
    expl_features: Optional[ArrayLike] = None
    episode_done: bool = False
    debug_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResetState:
    init_object_pos: ArrayLike
    init_object_rot: ArrayLike
    goal_offset: float
    init_hand_qpos: Optional[ArrayLike] = None
    init_fingertip_pos: Optional[ArrayLike] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HandAPI(ABC):
    """Minimal interface each real hand backend must implement for deploy."""

    num_hand_dofs: Optional[int] = None

    def set_deployment_context(self, context: PolicyDeploymentContext) -> None:
        self.context = context

    def get_num_hand_dofs(self) -> int:
        if self.num_hand_dofs is not None:
            return int(self.num_hand_dofs)
        context = getattr(self, "context", None)
        if context is not None:
            return int(context.hand_dof_dim)
        raise ValueError("Hand backend must define num_hand_dofs or receive a deployment context.")

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def get_observations(self) -> RobotObservations:
        raise NotImplementedError("This robot backend expects a TaskObservationProvider to assemble policy observations.")

    @abstractmethod
    def get_hand_joint_positions(self) -> ArrayLike:
        pass

    @abstractmethod
    def command_joint_targets(self, joint_targets: ArrayLike) -> None:
        pass

    def command_init_grasp(self, init_grasp: ArrayLike, final_settle_sec: float = 1.0) -> None:
        init_grasp = np.asarray(init_grasp, dtype=np.float32)
        expected_shape = (self.get_num_hand_dofs(),)
        if init_grasp.shape != expected_shape:
            raise ValueError(f"Expected init grasp shape {expected_shape}, got {init_grasp.shape}")
        self.command_joint_targets(init_grasp)
        if final_settle_sec > 0.0:
            time.sleep(float(final_settle_sec))

    def should_stop(self) -> bool:
        return False


# Backward-compatible name used by older scripts and configs.
RobotAPI = HandAPI


class TaskObservationProvider(ABC):
    def set_deployment_context(self, context: PolicyDeploymentContext) -> None:
        self.context = context

    def set_task_state_provider(self, provider: "TaskStateProvider") -> None:
        self.task_state_provider = provider

    @abstractmethod
    def reset(self, robot: HandAPI) -> None:
        pass

    @abstractmethod
    def build_observations(self, robot: HandAPI, last_action: ArrayLike) -> RobotObservations:
        pass


class TaskStateProvider(ABC):
    def set_deployment_context(self, context: PolicyDeploymentContext) -> None:
        self.context = context

    def reset(self, robot: HandAPI) -> None:
        pass

    def before_reset_capture(self, robot: HandAPI) -> None:
        pass

    def build_reset_state(
        self,
        robot: HandAPI,
        measured_hand_qpos: ArrayLike,
        measured_fingertip_pos: ArrayLike,
    ) -> TaskResetState:
        return self.get_reset_state(robot)

    def get_reset_state(self, robot: HandAPI) -> TaskResetState:
        raise NotImplementedError("Implement build_reset_state(...) or get_reset_state(...).")

    def get_goal_offset(self, robot: HandAPI, reset_state: TaskResetState) -> float:
        return float(reset_state.goal_offset)

    def before_rollout_start(self, robot: HandAPI, reset_state: TaskResetState) -> None:
        pass
