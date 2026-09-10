from __future__ import annotations

import argparse
import ast
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from isaacgymenvs.deploy.policy_rpc import JsonLineSocketClient, PolicyRPCError
from isaacgymenvs.deploy.real_robot_policy_api import (
    PolicyDeploymentContext,
    HandAPI,
    TaskObservationProvider,
    TaskStateProvider,
)
from isaacgymenvs.deploy.utils import (
    filter_constructor_kwargs,
    get_optional_hand_joint_order,
    load_class,
    parse_bool_arg,
    resolve_hand_observation_provider_class_spec,
    resolve_hand_robot_class_spec,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Real-hand client for the remote student policy server.")
    parser.add_argument("--host", default="127.0.0.1", help="Policy server host.")
    parser.add_argument("--port", type=int, default=5555, help="Policy server port.")
    parser.add_argument("--session-id", default="default", help="Logical policy session id.")
    parser.add_argument("--request-timeout-sec", type=float, default=5.0, help="RPC socket timeout.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions on the server.")
    parser.add_argument("--task", default="artmanip", help="Task config name.")
    parser.add_argument("--hand", default="sharpa", help="Hand config name.")
    parser.add_argument("--object", default="knife", help="Object config name.")
    parser.add_argument(
        "--asset-dir",
        default="",
        help="Optional asset/cache directory name under assets/objects and caches/initial_grasp, e.g. knife_30.",
    )
    parser.add_argument("--robot-class", default="", help="Optional import path in module:Class format. Defaults to isaacgymenvs.deploy.<hand>.robot:<HandName>Robot.")
    parser.add_argument(
        "--observation-provider-class",
        default="",
        help="Optional import path in module:Class format. Defaults to isaacgymenvs.deploy.<hand>.observation_provider:<HandName>StaticTaskObservationProvider.",
    )
    parser.add_argument(
        "--task-state-class",
        default="isaacgymenvs.deploy.task_state_provider:GraspCacheTaskStateProvider",
        help="Import path in module:Class format for reset-time object pose / init grasp / goal logic.",
    )
    parser.add_argument("--robot-kwargs-json", default="{}", help="JSON dict passed to the robot constructor.")
    parser.add_argument("--observation-provider-kwargs-json", default="{}", help="JSON dict passed to the observation provider constructor.")
    parser.add_argument("--task-state-kwargs-json", default="{}", help="JSON dict passed to the task-state provider constructor.")
    parser.add_argument("--control-sleep", type=float, default=0.0, help="Optional sleep after each control step.")
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum control steps. Use 0 to run until the robot stops.")
    parser.add_argument(
        "--max-time-sec",
        type=float,
        default=0.0,
        help="Maximum wall-clock rollout time in seconds. Use 0 to disable the timeout.",
    )
    parser.add_argument("--print-every", type=int, default=20, help="How often to print actions.")
    parser.add_argument("--reset-joint-position", default="", help="Optional reset joint pose in policy DOF order.")
    parser.add_argument("--reset-sleep-sec", type=float, default=1.5, help="How long to wait after reset motion.")
    parser.add_argument("--zero-state-on-connect", action="store_true", help="Command all hand joints to zero right after connecting.")
    parser.add_argument("--zero-state-settle-sec", type=float, default=1.0, help="How long to wait after the zero-state command on connect.")
    parser.add_argument("--init-object-pos", default="", help="Optional init object position override as x,y,z.")
    parser.add_argument("--init-object-rot", default="", help="Optional init object rotation override as x,y,z,w.")
    parser.add_argument("--goal-offset", type=float, default=0.0, help="Static one-dimensional goal offset used by the policy.")
    parser.add_argument("--object-size", default="0.02,0.016,0.14", help="Static object size triplet used in init_obs.")
    parser.add_argument("--init-grasp-qpos", default="", help="Optional init grasp joint pose override in policy DOF order used in init_obs.")
    parser.add_argument("--init-fingertip-pos", default="", help="Optional 15-value init fingertip position override used in init_obs.")
    parser.add_argument("--grasp-instance-id", default="", help="Cache instance id, e.g. 000.")
    parser.add_argument("--use-measured-init-hand-qpos", type=parse_bool_arg, default=False, help="Use the measured real hand qpos as init_hand_qpos instead of cached grasp qpos.")
    parser.add_argument("--use-measured-init-fingertip-pos", type=parse_bool_arg, default=False, help="Use the measured real fingertip positions as init_fingertip_pos instead of cached fingertip positions.")
    return parser.parse_args()


def parse_float_list(raw_value: str, expected_len: int, name: str):
    text = raw_value.strip()
    if not text:
        return None
    try:
        if text.startswith("["):
            values = json.loads(text)
        else:
            values = [float(token) for token in text.split(",")]
    except Exception as exc:
        raise ValueError(f"Could not parse {name}: {raw_value!r}") from exc
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected_len,):
        raise ValueError(f"{name} must contain exactly {expected_len} floats, got shape {array.shape}")
    return array


def _strip_yaml_comment(value: str):
    quote_char = None
    bracket_depth = 0
    for idx, char in enumerate(value):
        if char in {"'", '"'}:
            if quote_char is None:
                quote_char = char
            elif quote_char == char:
                quote_char = None
        elif quote_char is None:
            if char in "[{(":
                bracket_depth += 1
            elif char in "]})":
                bracket_depth = max(0, bracket_depth - 1)
            elif char == "#" and bracket_depth == 0:
                return value[:idx].rstrip()
    return value.rstrip()


def _parse_scalar_yaml_value(raw_value: str):
    value = _strip_yaml_comment(raw_value.strip())
    if value == "":
        return {}
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def load_simple_yaml(path: Path):
    root = {}
    stack = [(-1, root)]
    for raw_line in path.read_text().splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = raw_line.rstrip()
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("- "):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        parsed_value = _parse_scalar_yaml_value(value)
        if value.strip() == "":
            current[key] = {}
            stack.append((indent, current[key]))
        else:
            current[key] = parsed_value
    return root


def load_urdf_joint_limits(urdf_path: Path):
    urdf_path = urdf_path.expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    lower_limits = []
    upper_limits = []
    joint_names = []
    for joint_elem in root.findall("joint"):
        if joint_elem.attrib.get("type") == "fixed":
            continue
        limit_elem = joint_elem.find("limit")
        lower_limits.append(float(limit_elem.attrib.get("lower", 0.0)) if limit_elem is not None else 0.0)
        upper_limits.append(float(limit_elem.attrib.get("upper", 0.0)) if limit_elem is not None else 0.0)
        joint_names.append(joint_elem.attrib["name"])
    return np.asarray(lower_limits, dtype=np.float32), np.asarray(upper_limits, dtype=np.float32), joint_names


def _reorder_urdf_limits_to_policy_order(joint_names, lower_limits, upper_limits, policy_joint_order):
    if not policy_joint_order:
        return lower_limits, upper_limits, joint_names
    joint_to_index = {name: idx for idx, name in enumerate(joint_names)}
    if set(joint_names) == set(policy_joint_order):
        policy_indices = [joint_to_index[name] for name in policy_joint_order]
        return lower_limits[policy_indices], upper_limits[policy_indices], list(policy_joint_order)
    return lower_limits, upper_limits, joint_names


def _set_task_state_goal_offset(task_state_provider, goal_offset: float) -> bool:
    if task_state_provider is None:
        return False
    if hasattr(task_state_provider, "goal_offset"):
        task_state_provider.goal_offset = float(goal_offset)
        return True
    set_goal_offset = getattr(task_state_provider, "set_goal_offset", None)
    if callable(set_goal_offset):
        set_goal_offset(float(goal_offset))
        return True
    return False


def _provider_fingertip_names(provider, count: int):
    names = getattr(provider, "fingertip_names", None)
    if names is None:
        return [f"tip_{idx}" for idx in range(count)]
    names = list(names)
    if len(names) != count:
        return [f"tip_{idx}" for idx in range(count)]
    return names


def print_reset_debug_summary(provider, joint_names):
    reset_debug = getattr(provider, "reset_debug_info", None) or {}
    if not reset_debug:
        return
    init_sources = reset_debug.get("init_sources", {})
    cached_vs_measured = reset_debug.get("cached_vs_measured", {})
    hand_qpos_diff = cached_vs_measured.get("hand_qpos", {})
    hand_qpos_values = cached_vs_measured.get("hand_qpos_values", {})
    fingertip_pos_diff = cached_vs_measured.get("fingertip_pos", {})
    fingertip_pos_values = cached_vs_measured.get("fingertip_pos_values", {})
    print(
        "reset_debug "
        f"goal_offset={float(reset_debug.get('goal_offset', 0.0)):.6f} "
        f"object_pos_source={init_sources.get('object_pos', '')} "
        f"object_rot_source={init_sources.get('object_rot', '')} "
        f"hand_qpos_source={init_sources.get('hand_qpos', '')} "
        f"fingertip_pos_source={init_sources.get('fingertip_pos', '')}"
    )
    print(
        "reset_debug_diff "
        f"hand_qpos_l2={float(hand_qpos_diff.get('l2_norm', 0.0)):.6f} "
        f"hand_qpos_max_abs={float(hand_qpos_diff.get('max_abs', 0.0)):.6f} "
        f"fingertip_pos_l2={float(fingertip_pos_diff.get('l2_norm', 0.0)):.6f} "
        f"fingertip_pos_max_abs={float(fingertip_pos_diff.get('max_abs', 0.0)):.6f}"
    )
    cached_hand_qpos = np.asarray(hand_qpos_values.get("cached", []), dtype=np.float32)
    measured_hand_qpos = np.asarray(hand_qpos_values.get("measured", []), dtype=np.float32)
    hand_qpos_delta = np.asarray(hand_qpos_values.get("diff", []), dtype=np.float32)
    if (
        cached_hand_qpos.shape == measured_hand_qpos.shape == hand_qpos_delta.shape
        and cached_hand_qpos.shape == (len(joint_names),)
    ):
        for joint_name, cached_value, measured_value, delta_value in zip(
            joint_names,
            cached_hand_qpos.tolist(),
            measured_hand_qpos.tolist(),
            hand_qpos_delta.tolist(),
        ):
            print(
                "reset_joint "
                f"name={joint_name} "
                f"cached={cached_value:.6f} "
                f"measured={measured_value:.6f} "
                f"diff={delta_value:.6f}"
            )
    cached_fingertip_pos = np.asarray(fingertip_pos_values.get("cached", []), dtype=np.float32)
    measured_fingertip_pos = np.asarray(fingertip_pos_values.get("measured", []), dtype=np.float32)
    fingertip_pos_delta = np.asarray(fingertip_pos_values.get("diff", []), dtype=np.float32)
    if (
        cached_fingertip_pos.shape == measured_fingertip_pos.shape == fingertip_pos_delta.shape
        and cached_fingertip_pos.size % 3 == 0
    ):
        fingertip_count = cached_fingertip_pos.size // 3
        fingertip_names = _provider_fingertip_names(provider, fingertip_count)
        cached_fingertip_pos = cached_fingertip_pos.reshape(fingertip_count, 3)
        measured_fingertip_pos = measured_fingertip_pos.reshape(fingertip_count, 3)
        fingertip_pos_delta = fingertip_pos_delta.reshape(fingertip_count, 3)
        for finger_name, cached_tip, measured_tip, delta_tip in zip(
            fingertip_names,
            cached_fingertip_pos,
            measured_fingertip_pos,
            fingertip_pos_delta,
        ):
            print(
                "reset_tip "
                f"name={finger_name} "
                f"cached={np.array2string(cached_tip, precision=4, suppress_small=True)} "
                f"measured={np.array2string(measured_tip, precision=4, suppress_small=True)} "
                f"diff={np.array2string(delta_tip, precision=4, suppress_small=True)} "
                f"dist={float(np.linalg.norm(delta_tip)):.6f}"
            )


def build_local_task_env(task: str, hand: str, object_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    task_cfg = load_simple_yaml(repo_root / "isaacgymenvs" / "cfg" / "task" / f"{task}.yaml")
    hand_cfg = load_simple_yaml(repo_root / "isaacgymenvs" / "cfg" / "hand" / f"{hand}.yaml")
    object_cfg = load_simple_yaml(repo_root / "isaacgymenvs" / "cfg" / "object" / f"{object_name}.yaml")
    task_env_cfg = task_cfg["env"]
    asset_relpath = hand_cfg["asset"]
    urdf_path = (repo_root / asset_relpath).resolve()
    lower_limits, upper_limits, joint_names = load_urdf_joint_limits(urdf_path)
    lower_limits, upper_limits, joint_names = _reorder_urdf_limits_to_policy_order(
        joint_names,
        lower_limits,
        upper_limits,
        get_optional_hand_joint_order(hand),
    )
    num_actions = int(hand_cfg["task"]["numActions"])
    if lower_limits.shape[0] != num_actions:
        raise ValueError(f"Expected {num_actions} URDF joints, got {lower_limits.shape[0]} from {urdf_path}")
    object_task_cfg = object_cfg.get("task", {})
    object_goals = object_task_cfg.get("goals")
    if object_goals is None:
        raise ValueError("object.task.goals is required.")
    return SimpleNamespace(
        task=task,
        hand=hand,
        object_name=object_name,
        action_dim=num_actions,
        num_hand_dofs=num_actions,
        dt=float(task_cfg["sim"]["dt"]),
        control_freq_inv=int(task_env_cfg["controlFrequencyInv"]),
        use_relative_control=bool(task_env_cfg["useRelativeControl"]),
        act_moving_average=float(task_env_cfg["actionsMovingAverage"]),
        hand_dof_speed_scale=float(task_env_cfg["dofSpeedScale"]),
        hand_dof_lower_limits=lower_limits,
        hand_dof_upper_limits=upper_limits,
        hand_joint_names=joint_names,
        urdf_path=str(urdf_path),
        goal_switch_timeout_sec=float(task_env_cfg.get("goalSwitchTimeoutSec", 0.0)),
        object_goals=object_goals,
    )


class RemoteStudentPolicyDeployer:
    def __init__(self, local_task_env, remote_context, session_id: str, deterministic: bool):
        self.local_task_env = local_task_env
        self.remote_context = remote_context
        self.session_id = session_id
        self.deterministic = bool(deterministic)
        self.prev_targets = None
        self.need_reset_rnn = True
        if int(remote_context["action_dim"]) != int(local_task_env.action_dim):
            raise ValueError(
                f"Server action_dim={remote_context['action_dim']} does not match local action_dim={local_task_env.action_dim}"
            )

    def build_context(self):
        return PolicyDeploymentContext(
            policy_obs_dim=int(self.remote_context["policy_obs_dim"]),
            student_obs_dim=int(self.remote_context["student_obs_dim"]),
            privileged_obs_dim=int(self.remote_context["privileged_obs_dim"]),
            expl_feature_dim=int(self.remote_context["expl_feature_dim"]),
            action_dim=int(self.remote_context["action_dim"]),
            hand_dof_dim=int(self.local_task_env.num_hand_dofs),
            control_dt=float(self.local_task_env.dt * self.local_task_env.control_freq_inv),
            use_relative_control=bool(self.local_task_env.use_relative_control),
            action_moving_average=float(self.local_task_env.act_moving_average),
            hand_dof_speed_scale=float(self.local_task_env.hand_dof_speed_scale),
            hand_dof_lower_limits=self.local_task_env.hand_dof_lower_limits.copy(),
            hand_dof_upper_limits=self.local_task_env.hand_dof_upper_limits.copy(),
            student_temporal_obs_mode=str(self.remote_context["student_temporal_obs_mode"]),
            student_history_len=int(self.remote_context["student_history_len"]),
            student_proprio_dim_per_step=int(self.remote_context["student_proprio_dim_per_step"]),
            student_contact_dim_per_step=int(self.remote_context["student_contact_dim_per_step"]),
            student_init_dim=int(self.remote_context["student_init_dim"]),
            distill_meta={},
        )

    def reset(self, hand_joint_positions):
        qpos = np.asarray(hand_joint_positions, dtype=np.float32)
        if qpos.shape != (self.local_task_env.num_hand_dofs,):
            raise ValueError(
                f"hand_joint_positions must have shape ({self.local_task_env.num_hand_dofs},), got {qpos.shape}"
            )
        self.prev_targets = qpos.copy()
        self.need_reset_rnn = True

    def _action_to_joint_targets(self, action):
        if self.prev_targets is None:
            raise RuntimeError("Call deployer.reset(...) before requesting actions.")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.local_task_env.action_dim,):
            raise ValueError(f"action must have shape ({self.local_task_env.action_dim},), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        lower = self.local_task_env.hand_dof_lower_limits
        upper = self.local_task_env.hand_dof_upper_limits

        if self.local_task_env.use_relative_control:
            cur_targets = self.prev_targets + self.local_task_env.hand_dof_speed_scale * self.local_task_env.dt * action
            cur_targets = np.clip(cur_targets, lower, upper)
        else:
            desired = lower + 0.5 * (action + 1.0) * (upper - lower)
            cur_targets = (
                self.local_task_env.act_moving_average * desired
                + (1.0 - self.local_task_env.act_moving_average) * self.prev_targets
            )
            cur_targets = np.clip(cur_targets, lower, upper)
        self.prev_targets = cur_targets.copy()
        return cur_targets

    def act(self, rpc_client: JsonLineSocketClient, observations):
        payload = {
            "type": "infer",
            "session_id": self.session_id,
            "policy_obs": np.asarray(observations.policy_obs, dtype=np.float32).tolist(),
            "student_obs": np.asarray(observations.student_obs, dtype=np.float32).tolist(),
            "deterministic": self.deterministic,
            "reset_rnn": self.need_reset_rnn,
        }
        if int(self.remote_context["expl_feature_dim"]) > 0 and observations.expl_features is not None:
            payload["expl_features"] = np.asarray(observations.expl_features, dtype=np.float32).tolist()
        response = rpc_client.request(payload)
        raw_action = np.asarray(response["action"], dtype=np.float32)
        joint_targets = self._action_to_joint_targets(raw_action)
        self.need_reset_rnn = False
        return raw_action, joint_targets, response


def main():
    args = parse_args()

    local_task_env = build_local_task_env(args.task, args.hand, args.object)
    robot_cls = load_class(resolve_hand_robot_class_spec(args.hand, args.robot_class), HandAPI)
    provider_cls = load_class(
        resolve_hand_observation_provider_class_spec(args.hand, args.observation_provider_class),
        TaskObservationProvider,
    )
    task_state_cls = load_class(args.task_state_class, TaskStateProvider)
    robot_kwargs = json.loads(args.robot_kwargs_json)
    provider_kwargs = json.loads(args.observation_provider_kwargs_json)
    task_state_kwargs = json.loads(args.task_state_kwargs_json)

    parsed_reset_qpos = parse_float_list(args.reset_joint_position, local_task_env.num_hand_dofs, "reset_joint_position")
    parsed_init_object_pos = parse_float_list(args.init_object_pos, 3, "init_object_pos")
    parsed_init_object_rot = parse_float_list(args.init_object_rot, 4, "init_object_rot")
    parsed_object_size = parse_float_list(args.object_size, 3, "object_size")
    parsed_init_grasp_qpos = parse_float_list(args.init_grasp_qpos, local_task_env.num_hand_dofs, "init_grasp_qpos")
    parsed_init_fingertip_pos = parse_float_list(args.init_fingertip_pos, 15, "init_fingertip_pos")

    robot_kwargs.setdefault("reset_joint_position", parsed_reset_qpos)
    robot_kwargs.setdefault("reset_sleep_sec", args.reset_sleep_sec)
    robot_kwargs.setdefault("zero_state_on_connect", args.zero_state_on_connect)
    robot_kwargs.setdefault("zero_state_settle_sec", args.zero_state_settle_sec)

    provider_kwargs.setdefault(
        "object_size",
        parsed_object_size if parsed_object_size is not None else np.array([0.02, 0.016, 0.14], dtype=np.float32),
    )
    provider_kwargs.setdefault("urdf_path", local_task_env.urdf_path)

    task_state_kwargs.setdefault("goal_offset", args.goal_offset)
    task_state_kwargs.setdefault("hand_type", args.hand)
    task_state_kwargs.setdefault("object_type", args.object)
    task_state_kwargs.setdefault("asset_dir", args.asset_dir)
    if args.grasp_instance_id:
        task_state_kwargs.setdefault("instance_id", args.grasp_instance_id)
    task_state_kwargs.setdefault("use_measured_init_hand_qpos", args.use_measured_init_hand_qpos)
    task_state_kwargs.setdefault("use_measured_init_fingertip_pos", args.use_measured_init_fingertip_pos)
    if parsed_init_object_pos is not None:
        task_state_kwargs.setdefault("init_object_pos", parsed_init_object_pos)
    if parsed_init_object_rot is not None:
        task_state_kwargs.setdefault("init_object_rot", parsed_init_object_rot)
    if parsed_init_grasp_qpos is not None:
        task_state_kwargs.setdefault("init_hand_qpos", parsed_init_grasp_qpos)
    if parsed_init_fingertip_pos is not None:
        task_state_kwargs.setdefault("init_fingertip_pos", parsed_init_fingertip_pos)

    robot = None
    with JsonLineSocketClient(args.host, args.port, timeout_sec=args.request_timeout_sec) as rpc_client:
        init_response = rpc_client.request({"type": "init_session", "session_id": args.session_id})
        remote_context = init_response["context"]
        deployer = RemoteStudentPolicyDeployer(
            local_task_env=local_task_env,
            remote_context=remote_context,
            session_id=args.session_id,
            deterministic=args.deterministic,
        )

        robot = robot_cls(**filter_constructor_kwargs(robot_cls, robot_kwargs))
        provider = provider_cls(**filter_constructor_kwargs(provider_cls, provider_kwargs))
        task_state_provider = task_state_cls(**filter_constructor_kwargs(task_state_cls, task_state_kwargs))
        default_goal_offset = float(getattr(task_state_provider, "goal_offset", args.goal_offset))
        context = deployer.build_context()
        robot.set_deployment_context(context)
        provider.set_deployment_context(context)
        task_state_provider.set_deployment_context(context)
        provider.set_task_state_provider(task_state_provider)

        initial_goal_override = remote_context.get("goal_override")
        server_interactive_goal_input = bool(remote_context.get("server_interactive_goal_input", False))
        if initial_goal_override is not None:
            _set_task_state_goal_offset(task_state_provider, float(initial_goal_override))
            print(f"Applied initial server goal_override={float(initial_goal_override):.6f}")

        if getattr(provider, "uses_static_object_state", False):
            print("Warning: using static object/init/goal inputs for real deployment; no live object tracking is active.")

        if server_interactive_goal_input:
            print("Server interactive goal switching is enabled.")

        step_idx = 0
        reset_idx = 0
        last_action = np.zeros(context.action_dim, dtype=np.float32)
        try:
            robot.connect()
            robot.reset()
            provider.reset(robot)
            print_reset_debug_summary(provider, local_task_env.hand_joint_names)
            deployer.reset(robot.get_hand_joint_positions())
            print(
                f"Student policy client connected to {args.host}:{args.port}; "
                f"control mode={'relative' if context.use_relative_control else 'absolute'}"
            )
            rollout_start_time = time.monotonic()

            while True:
                if robot.should_stop():
                    break
                if args.max_steps > 0 and step_idx >= args.max_steps:
                    break
                if args.max_time_sec > 0.0 and (time.monotonic() - rollout_start_time) >= args.max_time_sec:
                    print(f"Reached max rollout time: {args.max_time_sec:.3f}s. Exiting deployment loop.")
                    break

                observations = provider.build_observations(robot, last_action)
                raw_action, joint_targets, response = deployer.act(rpc_client, observations)
                if "goal_override" in response:
                    goal_override = response.get("goal_override")
                    if goal_override is None:
                        _set_task_state_goal_offset(task_state_provider, default_goal_offset)
                    else:
                        _set_task_state_goal_offset(task_state_provider, float(goal_override))
                robot.command_joint_targets(joint_targets)

                if step_idx % max(args.print_every, 1) == 0:
                    policy_goal = float(np.asarray(observations.policy_obs, dtype=np.float32)[-16])
                    provider_goal = float(getattr(provider, "goal_offset", policy_goal))
                    print(
                        f"step={step_idx} "
                        f"raw_action={np.array2string(raw_action, precision=4, suppress_small=True)} "
                        f"joint_targets={np.array2string(joint_targets, precision=4, suppress_small=True)} "
                        f"server_step={response.get('server_step', -1)} "
                        f"timing_ms={response.get('timing_ms', 0.0):.2f} "
                        f"policy_goal={policy_goal:.6f} "
                        f"provider_goal={provider_goal:.6f} "
                        f"goal_override={response.get('goal_override', None)}"
                    )
                last_action = raw_action.astype(np.float32, copy=False)
                if observations.episode_done:
                    robot.reset()
                    provider.reset(robot)
                    print_reset_debug_summary(provider, local_task_env.hand_joint_names)
                    reset_idx += 1
                    deployer.reset(robot.get_hand_joint_positions())
                    last_action = np.zeros(context.action_dim, dtype=np.float32)

                if args.control_sleep > 0.0:
                    time.sleep(args.control_sleep)
                step_idx += 1
        except KeyboardInterrupt:
            print("Termination requested. Stopping deployment loop.")
        except PolicyRPCError as exc:
            raise RuntimeError(f"Policy server RPC failed: {exc}") from exc
        finally:
            try:
                rpc_client.request({"type": "close_session", "session_id": args.session_id})
            except Exception:
                pass
            if robot is not None:
                robot.disconnect()


if __name__ == "__main__":
    main()
