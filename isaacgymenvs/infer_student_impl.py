import argparse
from pathlib import Path

import isaacgym  # noqa: F401
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import open_dict

from isaacgymenvs.distill import (
    _infer_expl_num_blocks,
    build_student_encoder_from_spec,
    preprocess_train_config,
    resolve_student_encoder_spec,
)
from isaacgymenvs.utils.infer_recording import (
    build_reset_first_cur_targets,
)
from isaacgymenvs.utils.infer_runtime import (
    configure_fixed_grasp_inference_until_terminated,
    is_inference_eval_complete,
)
from isaacgymenvs.utils.student_obs_utils import DEFAULT_STUDENT_INIT_OBS_DIM
from isaacgymenvs.utils.student_runtime_utils import (
    build_student_cfg_overrides_from_meta,
    resolve_teacher_identity_from_meta,
    resolve_student_runtime_layout,
)
from isaacgymenvs.utils.distributed_runtime import force_single_process_env
from isaacgymenvs.utils.player_utils import init_player_rnn_for_batch, parse_bool_arg


def parse_args():
    parser = argparse.ArgumentParser(description="Run standalone inference with a distilled student encoder.")
    parser.add_argument("--student-artifact", required=True, help="Path to the distilled student artifact.")
    parser.add_argument("--checkpoint", default="", help="Optional teacher checkpoint override.")
    parser.add_argument("--task", default="artmanip", help="Task config name.")
    parser.add_argument("--train", default="artmanipSAPGPrivLSTMPPO", help="Train config name.")
    parser.add_argument("--hand", default="sharpa", help="Hand config name.")
    parser.add_argument("--object", default="knife", help="Object config name.")
    parser.add_argument("--asset-dir", default="", help="Optional asset directory under assets/objects, e.g. knife_multi.")
    parser.add_argument("--instance-id", required=True, help="Object instance id used for one-instance inference.")
    parser.add_argument("--max-steps", type=int, default=3000, help="Maximum number of environment steps to run.")
    parser.add_argument(
        "--goal-switch-interval-secs",
        type=float,
        default=None,
        help="Optional success-hold duration required before switching goals during inference.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions.")
    parser.add_argument("--expl-block-idx", type=int, default=-1, help="Teacher exploration block id. Use -1 to follow the student artifact metadata when applicable.")
    parser.add_argument("--sim-device", default="cuda:0", help="Simulation device.")
    parser.add_argument("--rl-device", default="cuda:0", help="RL device.")
    parser.add_argument("--graphics-device-id", type=int, default=0, help="Graphics device id.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--torch-deterministic",
        type=parse_bool_arg,
        default=None,
        help="Optional override for cfg.torch_deterministic.",
    )
    parser.add_argument("--save-latents", default="", help="Optional .pt path to save student latents.")
    parser.add_argument("--save-cur-targets", default="", help="Optional .npy path to save task_env.cur_targets for open-loop replay.")
    parser.add_argument("--save-video", default="", help="Optional video path (.mp4/.gif) recorded from artmanip.yaml camera.")
    parser.add_argument("--video-env-index", type=int, default=0, help="Which env camera to record when --save-video is used.")
    parser.add_argument("--video-fps", type=int, default=30, help="Output video fps when --save-video is used.")
    parser.add_argument(
        "--object-dof-damping-override",
        type=float,
        default=None,
        help="Optional override for object.default_props.dof_damping during inference.",
    )
    parser.add_argument(
        "--randomize",
        type=parse_bool_arg,
        default=False,
        help="Whether to enable task.randomize during inference. Defaults to disabled.",
    )
    return parser.parse_args()


def set_module_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


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
    latent_dim = player.model.a2c_network.priv_encoder(torch.zeros(1, teacher_obs_dim, device=player.device)).shape[1]
    student_encoder = build_student_encoder_from_spec(
        student_obs_dim=student_obs_dim,
        latent_dim=latent_dim,
        encoder_cfg=encoder_cfg,
        fallback_activation=rlg_config_dict["params"]["network"]["mlp"]["activation"],
        encoder_spec=encoder_spec,
    ).to(player.device)

    student_encoder.load_state_dict(student_artifact["student_encoder_state_dict"])
    set_module_requires_grad(student_encoder, False)
    student_encoder.eval()

    return student_encoder, student_obs_dim, proprio_obs_dim, proprio_history_len


def main():
    args = parse_args()
    force_single_process_env()

    import isaacgymenvs
    isaacgymenvs.register_omegaconf_resolvers()
    from isaacgymenvs.learning import a2c_dict_network_builder, a2c_sapg_priv_network_builder
    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from rl_games.algos_torch import model_builder
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    student_artifact_path = Path(args.student_artifact).resolve()
    if not student_artifact_path.exists():
        raise FileNotFoundError(f"Student artifact not found: {student_artifact_path}")

    student_artifact = torch.load(student_artifact_path, map_location="cpu")
    distill_meta = student_artifact.get("distill_meta", {})
    teacher_checkpoint = args.checkpoint or distill_meta.get("teacher_checkpoint", "")
    if not teacher_checkpoint:
        raise ValueError("Could not determine the teacher checkpoint. Please pass --checkpoint or save teacher_checkpoint in the student artifact.")
    teacher_checkpoint_path = Path(teacher_checkpoint).resolve()
    if not teacher_checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint_path}")

    resolved_identity = resolve_teacher_identity_from_meta(
        distill_meta=distill_meta,
        task=args.task,
        train=args.train,
        hand=args.hand,
        object_name=args.object,
    )

    overrides = [
        f"task={resolved_identity['task']}",
        f"train={resolved_identity['train']}",
        f"hand={resolved_identity['hand']}",
        f"object={resolved_identity['object']}",
        f"asset_dir={args.asset_dir}",
        f"headless={args.headless}",
        f"sim_device={args.sim_device}",
        f"rl_device={args.rl_device}",
        f"graphics_device_id={args.graphics_device_id}",
        f"seed={args.seed}",
        "task.env.numEnvs=1",
        "multi_gpu=False",
    ]
    overrides.extend(build_student_cfg_overrides_from_meta(distill_meta))
    with initialize(version_base="1.1", config_path="./cfg"):
        cfg = compose(config_name="config", overrides=overrides)

    inferred_blocks = _infer_expl_num_blocks(teacher_checkpoint_path)

    instance_id_list = list(cfg.object.asset.instance_id_list)
    if args.instance_id not in instance_id_list and instance_id_list != [""]:
        raise ValueError(f"instance_id '{args.instance_id}' not found in object config list {instance_id_list}")

    with open_dict(cfg):
        cfg.task.env.numEnvs = 1
        cfg.task.env.episodeLength = int(args.max_steps)
        if args.goal_switch_interval_secs is not None:
            cfg.task.env.successHoldDurationSec = float(args.goal_switch_interval_secs)
            cfg.task.env.successHoldDurationRangeSec = [
                float(args.goal_switch_interval_secs),
                float(args.goal_switch_interval_secs),
            ]
        cfg.task.task.randomize = bool(args.randomize)
        if args.torch_deterministic is not None:
            cfg.torch_deterministic = bool(args.torch_deterministic)
        cfg.multi_gpu = False
        cfg.task.env.graspSplit = "selected"
        cfg.object.asset.instance_id_list = [args.instance_id]
        if args.object_dof_damping_override is not None:
            cfg.object.default_props.dof_damping = float(args.object_dof_damping_override)
        cfg.test = True
        cfg.checkpoint = str(teacher_checkpoint_path)
        if args.save_video:
            cfg.task.env.enableCameraSensors = True

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0)

    def create_env(**kwargs):
        return isaacgymenvs.make(
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
            **kwargs,
        )

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: create_env(**kwargs),
        },
    )
    vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    env_cls = isaacgym_task_map[cfg.task_name]
    if getattr(env_cls, "dict_obs_cls", False):
            raise RuntimeError("Student inference currently supports flat observations only.")

    rlg_config_dict = preprocess_train_config(cfg, omegaconf_to_dict(cfg.train))
    player_cfg = rlg_config_dict["params"]["config"].setdefault("player", {})
    if inferred_blocks is not None:
        player_cfg["expl_num_blocks"] = inferred_blocks

    model_builder.register_network("actor_critic_dict", a2c_dict_network_builder.A2CBuilder)
    model_builder.register_network("actor_critic_sapg_priv", a2c_sapg_priv_network_builder.A2CSAPGPrivBuilder)

    runner = Runner()
    runner.load(rlg_config_dict)
    player = runner.create_player()
    player.restore(cfg.checkpoint)
    player.has_batch_dimension = True

    if inferred_blocks is not None and player.intr_reward_coef_embd is not None:
        if args.expl_block_idx >= 0:
            block_idx = max(0, min(args.expl_block_idx, inferred_blocks - 1))
        else:
            block_idx = int(distill_meta.get("expl_block_idx", 0))
        coef_ids = torch.linspace(50.0, 0.0, inferred_blocks, device=player.device)
        player.intr_reward_coef_embd[:] = coef_ids[block_idx]
        print(f"Using teacher exploration block {block_idx} with id {coef_ids[block_idx].item():.4f}")

    student_encoder, _, _, _ = build_student_encoder_from_artifact(
        player=player,
        cfg=cfg,
        rlg_config_dict=rlg_config_dict,
        student_artifact=student_artifact,
        distill_meta=distill_meta,
    )
    player.model.a2c_network.priv_encoder = student_encoder
    player.model.eval()
    task_env = player.env.env if hasattr(player.env, "env") else player.env
    if hasattr(task_env, "set_student_encoder_obs_enabled"):
        task_env.set_student_encoder_obs_enabled(True)
    if hasattr(task_env, "set_runtime_grasp_selection"):
        task_env.set_runtime_grasp_selection(args.instance_id, grasp_split="selected")
    inference_eval_mode = configure_fixed_grasp_inference_until_terminated(
        task_env,
        args,
        policy_name="Student",
    )

    obses = player.env_reset(player.env)
    batch_size = player.get_batch_size(obses, 1)
    init_player_rnn_for_batch(player, batch_size)

    latents = []
    recorded_cur_targets = []
    video_writer = None
    if args.save_video:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError("imageio is required for --save-video. Please install imageio in this environment.") from exc
        video_path = Path(args.save_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=args.video_fps)

    try:
        for step in range(args.max_steps):
            if hasattr(task_env, "get_student_encoder_observations"):
                student_obs = task_env.get_student_encoder_observations().to(player.device)
            else:
                student_obs = task_env.student_obs_buf.to(player.device)
            player.model.a2c_network.actor_encoder_obs_override = student_obs

            action = player.get_action(obses, is_deterministic=args.deterministic)
            latent = getattr(player.model.a2c_network, "last_privileged_latent", None)
            if latent is not None and args.save_latents:
                latents.append(latent.detach().cpu())

            obses, rewards, done, _ = player.env_step(player.env, action)
            if args.save_cur_targets:
                if not hasattr(task_env, "cur_targets"):
                    raise RuntimeError("Task env does not expose cur_targets, so --save-cur-targets cannot be used.")
                recorded_cur_targets.append(task_env.cur_targets.detach().cpu().clone())
            if video_writer is not None:
                if not hasattr(task_env, "get_camera_frame"):
                    raise RuntimeError("Task env does not expose get_camera_frame(), so --save-video cannot be used.")
                video_writer.append_data(task_env.get_camera_frame(env_id=args.video_env_index))

            if player.is_rnn:
                all_done_indices = done.nonzero(as_tuple=False).squeeze(-1)
                if len(all_done_indices) > 0:
                    for state in player.states:
                        state[:, all_done_indices, :] = 0.0

            if step % 100 == 0:
                reward_value = rewards.mean().item() if hasattr(rewards, "mean") else float(rewards)
                print(f"step={step} reward={reward_value:.4f}")

            if inference_eval_mode and is_inference_eval_complete(task_env):
                print(f"Inference evaluation completed after {step + 1} environment steps.")
                break

            if not args.headless:
                task_env.render(mode="human")
    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"Saved video to {Path(args.save_video)}")

    if args.save_latents:
        latent_path = Path(args.save_latents)
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        stacked = torch.cat(latents, dim=0) if latents else torch.empty(0)
        torch.save(stacked, latent_path)
        print(f"Saved student latents to {latent_path}")

    if args.save_cur_targets:
        cur_targets_path = Path(args.save_cur_targets)
        cur_targets_path.parent.mkdir(parents=True, exist_ok=True)
        cur_targets = build_reset_first_cur_targets(recorded_cur_targets, task_env)
        np.save(cur_targets_path, cur_targets.numpy())
        print(f"Saved cur_targets to {cur_targets_path}")

    if inference_eval_mode == "consecutive" and hasattr(task_env, "get_grasp_consecutive_evaluation_stats"):
        stats = task_env.get_grasp_consecutive_evaluation_stats()
        print(
            "Inference consecutive summary: "
            f"cycles={stats['consecutive_success_cycles']} "
            f"reason={stats['completion_reason']} "
            f"stage={stats['completion_stage']}"
        )


if __name__ == "__main__":
    main()
