from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

try:
    import isaacgym  # noqa: F401
except ModuleNotFoundError:
    isaacgym = None

import numpy as np
import torch

from isaacgymenvs.distill import (
    _infer_expl_num_blocks,
    build_student_encoder_from_spec,
    preprocess_train_config,
    resolve_student_encoder_spec,
)
from isaacgymenvs.utils.distributed_runtime import force_single_process_env
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.student_obs_utils import DEFAULT_STUDENT_INIT_OBS_DIM
from isaacgymenvs.utils.student_runtime_utils import resolve_student_runtime_layout, resolve_teacher_identity_from_meta


def ensure_gym_module():
    try:
        import gym  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    try:
        import gymnasium as gym
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Policy server requires `gym` or `gymnasium` in the igym environment."
        ) from exc

    sys.modules.setdefault("gym", gym)


def clone_states(states):
    if states is None:
        return None
    return [state.detach().clone() for state in states]


def reset_player_rnn_state(player):
    if not getattr(player, "is_rnn", False):
        player.states = None
        return

    default_states = player.model.get_default_rnn_state()
    resized_states = []
    for state in default_states:
        state = state.to(player.device)
        if state.size(1) == 1:
            resized_states.append(state.clone())
        else:
            resized_states.append(
                torch.zeros(
                    (state.size(0), 1, state.size(2)),
                    dtype=state.dtype,
                    device=player.device,
                )
            )
    player.states = resized_states


class _DeployVecEnvStub:
    def __init__(self, num_envs: int = 1):
        self.num_envs = int(num_envs)
        self._env_state = None

    def set_env_state(self, env_state):
        self._env_state = env_state

    def get_env_state(self):
        return self._env_state


def build_cfg(task: str, train: str, hand: str, object_name: str, asset_dir: str, rl_device: str, seed: int):
    import isaacgymenvs
    from hydra import compose, initialize_config_dir

    force_single_process_env()
    isaacgymenvs.register_omegaconf_resolvers()
    config_dir = str((Path(__file__).resolve().parents[1] / "cfg").resolve())
    overrides = [
        f"task={task}",
        f"train={train}",
        f"hand={hand}",
        f"object={object_name}",
        f"asset_dir={asset_dir}",
        "headless=True",
        f"sim_device={rl_device}",
        f"rl_device={rl_device}",
        "graphics_device_id=0",
        f"seed={seed}",
        "task.env.numEnvs=1",
        "multi_gpu=False",
    ]
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)
    cfg.multi_gpu = False
    return cfg


def build_student_encoder_from_artifact(player, cfg, rlg_config_dict, student_artifact, distill_meta):
    teacher_obs_dim = int(distill_meta["teacher_obs_dim"])
    runtime_layout = resolve_student_runtime_layout(
        distill_meta=distill_meta,
        fallback_history_len=int(cfg.task.env.get("proprioHistoryLen", 1)),
        fallback_proprio_dim_per_step=int(cfg.task.env.get("proprioObsDim", 2 * int(cfg.hand.task.numActions))),
        fallback_student_obs_dim=int(cfg.task.env.get("studentObsDim", 0)),
        fallback_init_dim=DEFAULT_STUDENT_INIT_OBS_DIM,
    )
    student_temporal_obs_mode = runtime_layout["student_temporal_obs_mode"]
    proprio_history_len = int(runtime_layout["history_len"])
    proprio_obs_dim = int(runtime_layout["proprio_dim_per_step"])
    student_obs_dim = int(runtime_layout["student_obs_dim"])

    sapg_priv_cfg = rlg_config_dict["params"]["network"]["sapg_priv"]
    encoder_cfg = dict(sapg_priv_cfg.get("encoder", {}))
    encoder_spec = distill_meta.get("student_encoder_spec")
    if encoder_spec is None:
        encoder_spec = resolve_student_encoder_spec(
            student_obs_dim=student_obs_dim,
            proprio_obs_dim=proprio_obs_dim,
            proprio_history_len=proprio_history_len,
            sapg_priv_cfg=sapg_priv_cfg,
            requested_type=distill_meta.get("student_encoder_type"),
        )

    latent_dim = player.model.a2c_network.priv_encoder(
        torch.zeros(1, teacher_obs_dim, device=player.device)
    ).shape[1]
    student_encoder = build_student_encoder_from_spec(
        student_obs_dim=student_obs_dim,
        latent_dim=latent_dim,
        encoder_cfg=encoder_cfg,
        fallback_activation=rlg_config_dict["params"]["network"]["mlp"]["activation"],
        encoder_spec=encoder_spec,
    ).to(player.device)
    student_encoder.load_state_dict(student_artifact["student_encoder_state_dict"])
    for param in student_encoder.parameters():
        param.requires_grad = False
    student_encoder.eval()
    return student_encoder, student_obs_dim, proprio_obs_dim, proprio_history_len


def build_policy_player(cfg, rlg_config_dict, teacher_checkpoint_path: Path, inferred_blocks, selected_block_idx: int):
    ensure_gym_module()

    import gym
    from rl_games.algos_torch import model_builder, players

    from isaacgymenvs.learning import a2c_dict_network_builder, a2c_sapg_priv_network_builder

    runtime_device = str(torch.device(cfg.rl_device))
    params = copy.deepcopy(rlg_config_dict["params"])
    config = params["config"]
    config["device_name"] = runtime_device
    config["device"] = runtime_device

    num_actions = int(cfg.hand.task.numActions)
    obs_dim = (
        int(cfg.task.env.policyObsDim)
        + int(cfg.task.env.privilegedObsDim)
        + 5
    )
    env_info = {
        "action_space": gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32),
        "observation_space": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        "agents": 1,
        "value_size": 1,
    }
    config["env_info"] = env_info
    config["vec_env"] = _DeployVecEnvStub(num_envs=1)

    player_cfg = config.setdefault("player", {})
    if inferred_blocks is not None:
        player_cfg["expl_num_blocks"] = inferred_blocks

    model_builder.register_network("actor_critic_dict", a2c_dict_network_builder.A2CBuilder)
    model_builder.register_network("actor_critic_sapg_priv", a2c_sapg_priv_network_builder.A2CSAPGPrivBuilder)

    player = players.PpoPlayerContinuous(params=params)
    player.restore(str(teacher_checkpoint_path))
    player.has_batch_dimension = True
    player.model.eval()

    if inferred_blocks is not None and player.intr_reward_coef_embd is not None:
        coef_ids = torch.linspace(50.0, 0.0, inferred_blocks, device=player.device).view(-1, 1)
        block_idx = max(0, min(int(selected_block_idx), inferred_blocks - 1))
        player.intr_reward_coef_embd[:] = coef_ids[block_idx]
        print(f"Using teacher exploration block {block_idx} with id {coef_ids[block_idx].item():.4f}")

    return player


class StudentPolicyRuntime:
    def __init__(
        self,
        player,
        policy_obs_dim: int,
        student_obs_dim: int,
        privileged_obs_dim: int,
        distill_meta: dict,
        goal_sequence=None,
    ):
        self.player = player
        self.device = player.device
        self.policy_obs_dim = int(policy_obs_dim)
        self.student_obs_dim = int(student_obs_dim)
        self.privileged_obs_dim = int(privileged_obs_dim)
        self.action_dim = int(player.actions_num)
        self.critic_policy_contact_dim = int(getattr(player.model.a2c_network, "critic_policy_contact_dim", 0))
        self.raw_extra_obs_dim = max(
            0,
            int(
                getattr(
                    player.model.a2c_network,
                    "original_input_shape",
                    self.policy_obs_dim + self.privileged_obs_dim,
                )
            )
            - (self.policy_obs_dim + self.privileged_obs_dim),
        )
        self.expl_feature_dim = max(0, self.raw_extra_obs_dim - self.critic_policy_contact_dim)

        runtime_layout = resolve_student_runtime_layout(
            distill_meta=distill_meta,
            fallback_history_len=1,
            fallback_proprio_dim_per_step=2 * self.action_dim,
            fallback_student_obs_dim=self.student_obs_dim,
            fallback_init_dim=DEFAULT_STUDENT_INIT_OBS_DIM,
        )
        self.student_temporal_obs_mode = runtime_layout["student_temporal_obs_mode"]
        self.student_history_len = int(runtime_layout["history_len"])
        self.student_proprio_dim_per_step = int(runtime_layout["proprio_dim_per_step"])
        self.student_contact_dim_per_step = int(runtime_layout["contact_dim_per_step"])
        self.student_init_dim = int(runtime_layout["init_dim"])
        self.goal_override = None
        self.goal_sequence = [float(goal) for goal in (goal_sequence or [])]
        self.goal_sequence_index = -1
        self.interactive_goal_input_enabled = False
        self.sessions = {}

    def init_session(self, session_id: str):
        self.sessions[session_id] = {"states": None, "step": 0}
        return {
            "policy_obs_dim": self.policy_obs_dim,
            "student_obs_dim": self.student_obs_dim,
            "privileged_obs_dim": self.privileged_obs_dim,
            "expl_feature_dim": self.expl_feature_dim,
            "action_dim": self.action_dim,
            "student_temporal_obs_mode": self.student_temporal_obs_mode,
            "student_history_len": self.student_history_len,
            "student_proprio_dim_per_step": self.student_proprio_dim_per_step,
            "student_contact_dim_per_step": self.student_contact_dim_per_step,
            "student_init_dim": self.student_init_dim,
            "goal_override": self.get_goal_override(),
            "server_interactive_goal_input": bool(self.interactive_goal_input_enabled),
        }

    def set_goal_override(self, goal_offset):
        self.goal_override = None if goal_offset is None else float(goal_offset)
        return self.goal_override

    def get_goal_override(self):
        return self.goal_override

    def clear_goal_override(self):
        self.goal_sequence_index = -1
        self.goal_override = None
        return self.goal_override

    def cycle_goal_override(self):
        if not self.goal_sequence:
            raise ValueError("No goal sequence is configured for server-side goal cycling.")
        self.goal_sequence_index = (self.goal_sequence_index + 1) % len(self.goal_sequence)
        self.goal_override = float(self.goal_sequence[self.goal_sequence_index])
        return self.goal_sequence_index, self.goal_override

    def close_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def _to_tensor(self, value, dim: int, name: str):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape != (1, dim):
            raise ValueError(f"{name} must have shape ({dim},) or (1, {dim}), got {tuple(tensor.shape)}")
        return tensor

    def _build_actor_obs(self, policy_obs, expl_features=None):
        dummy_privileged = torch.zeros((1, self.privileged_obs_dim), dtype=torch.float32, device=self.device)
        parts = [policy_obs, dummy_privileged]
        if self.raw_extra_obs_dim > 0:
            tail_obs = torch.zeros((1, self.raw_extra_obs_dim), dtype=torch.float32, device=self.device)
            if self.expl_feature_dim > 0:
                if self.player.intr_reward_coef_embd is not None:
                    extra_obs = self.player.intr_reward_coef_embd[:1].to(self.device)
                    if extra_obs.shape[1] != self.expl_feature_dim:
                        raise ValueError(
                            "intr_reward_coef_embd width does not match the deploy runtime exploration tail: "
                            f"embedding={extra_obs.shape[1]}, expected={self.expl_feature_dim}"
                        )
                elif expl_features is not None:
                    extra_obs = self._to_tensor(expl_features, self.expl_feature_dim, "expl_features")
                else:
                    extra_obs = torch.zeros((1, self.expl_feature_dim), dtype=torch.float32, device=self.device)
                tail_obs[:, self.critic_policy_contact_dim:] = extra_obs
            parts.append(tail_obs)
        return torch.cat(parts, dim=-1)

    def infer(
        self,
        session_id: str,
        policy_obs,
        student_obs,
        expl_features=None,
        deterministic: bool = False,
        reset_rnn: bool = False,
    ):
        if session_id not in self.sessions:
            self.init_session(session_id)
        session = self.sessions[session_id]

        if reset_rnn or session["states"] is None:
            reset_player_rnn_state(self.player)
        else:
            self.player.states = clone_states(session["states"])

        policy_obs = self._to_tensor(policy_obs, self.policy_obs_dim, "policy_obs")
        student_obs = self._to_tensor(student_obs, self.student_obs_dim, "student_obs")
        actor_obs = self._build_actor_obs(policy_obs, expl_features=expl_features)
        proc_obs = self.player._preproc_obs(actor_obs)

        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": proc_obs,
            "rnn_states": self.player.states,
        }

        self.player.model.a2c_network.actor_encoder_obs_override = student_obs
        start_time = time.perf_counter()
        try:
            with torch.no_grad():
                res_dict = self.player.model(input_dict)
        finally:
            self.player.model.a2c_network.actor_encoder_obs_override = None

        self.player.states = res_dict.get("rnn_states")
        session["states"] = clone_states(self.player.states)
        session["step"] += 1

        mus = res_dict["mus"]
        sampled_action = res_dict["actions"]
        chosen_action = mus if deterministic else sampled_action
        if getattr(self.player, "clip_actions", False):
            chosen_action = torch.clamp(chosen_action, -1.0, 1.0)

        latent = getattr(self.player.model.a2c_network, "last_privileged_latent", None)
        timing_ms = 1000.0 * (time.perf_counter() - start_time)
        return {
            "action": chosen_action.squeeze(0).detach().cpu().tolist(),
            "mu": mus.squeeze(0).detach().cpu().tolist(),
            "latent": None if latent is None else latent.squeeze(0).detach().cpu().tolist(),
            "server_step": int(session["step"]),
            "timing_ms": float(timing_ms),
            "goal_override": self.get_goal_override(),
        }


def load_student_policy_runtime(
    student_artifact_path: str,
    checkpoint_path: str,
    task: str,
    train: str,
    hand: str,
    object_name: str,
    asset_dir: str,
    rl_device: str,
    seed: int,
    expl_block_idx: int,
):
    from omegaconf import open_dict

    force_single_process_env()
    student_artifact_path = Path(student_artifact_path).expanduser().resolve()
    if not student_artifact_path.exists():
        raise FileNotFoundError(f"Student artifact not found: {student_artifact_path}")

    student_artifact = torch.load(student_artifact_path, map_location="cpu")
    distill_meta = student_artifact.get("distill_meta", {})
    resolved_identity = resolve_teacher_identity_from_meta(
        distill_meta=distill_meta,
        task=task,
        train=train,
        hand=hand,
        object_name=object_name,
    )

    teacher_checkpoint = checkpoint_path or distill_meta.get("teacher_checkpoint", "")
    if not teacher_checkpoint:
        raise ValueError("Could not determine the teacher checkpoint. Pass --checkpoint or save it in distill_meta.")
    teacher_checkpoint_path = Path(teacher_checkpoint).expanduser().resolve()
    if not teacher_checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint_path}")

    inferred_blocks = _infer_expl_num_blocks(teacher_checkpoint_path)
    if inferred_blocks is not None and expl_block_idx >= 0:
        block_idx = max(0, min(int(expl_block_idx), inferred_blocks - 1))
    else:
        block_idx = int(distill_meta.get("expl_block_idx", 0))

    cfg = build_cfg(
        task=resolved_identity["task"],
        train=resolved_identity["train"],
        hand=resolved_identity["hand"],
        object_name=resolved_identity["object"],
        asset_dir=asset_dir,
        rl_device=rl_device,
        seed=seed,
    )
    with open_dict(cfg):
        cfg.test = True
        cfg.multi_gpu = False
        cfg.checkpoint = str(teacher_checkpoint_path)

    rlg_config_dict = preprocess_train_config(cfg, omegaconf_to_dict(cfg.train))
    player = build_policy_player(
        cfg=cfg,
        rlg_config_dict=rlg_config_dict,
        teacher_checkpoint_path=teacher_checkpoint_path,
        inferred_blocks=inferred_blocks,
        selected_block_idx=block_idx,
    )
    student_encoder, student_obs_dim, _, _ = build_student_encoder_from_artifact(
        player=player,
        cfg=cfg,
        rlg_config_dict=rlg_config_dict,
        student_artifact=student_artifact,
        distill_meta=distill_meta,
    )
    player.model.a2c_network.priv_encoder = student_encoder
    player.model.eval()

    goal_sequence = cfg.object.task.get("goals")
    if goal_sequence is None:
        raise ValueError("object.task.goals is required.")

    return StudentPolicyRuntime(
        player=player,
        policy_obs_dim=int(cfg.task.env.policyObsDim),
        student_obs_dim=int(student_obs_dim),
        privileged_obs_dim=int(cfg.task.env.privilegedObsDim),
        distill_meta=distill_meta,
        goal_sequence=goal_sequence,
    )
