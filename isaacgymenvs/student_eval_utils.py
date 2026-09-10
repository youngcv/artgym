import time

import torch

from isaacgymenvs.utils.player_utils import init_player_rnn_for_batch


def set_student_encoder_obs_override(player, task_env, enabled):
    if not enabled:
        player.model.a2c_network.actor_encoder_obs_override = None
        return

    if hasattr(task_env, "get_student_encoder_observations"):
        student_obs = task_env.get_student_encoder_observations().to(player.device)
    else:
        student_obs = task_env.student_obs_buf.to(player.device)
    player.model.a2c_network.actor_encoder_obs_override = student_obs


def run_grasp_evaluation_loop(
    player,
    task_env,
    deterministic=False,
    progress_interval_sec=0.0,
    use_student_encoder=False,
    render_human=False,
    record_action_sequence=False,
    video_writer=None,
    video_env_index=0,
):
    if record_action_sequence and task_env.num_envs != 1:
        raise ValueError("Action-sequence replay/record currently supports only num_envs=1.")

    obses = player.env_reset(player.env)
    batch_size = player.get_batch_size(obses, 1)
    player.has_batch_dimension = True
    init_player_rnn_for_batch(player, batch_size)

    recorded_actions = []
    progress_interval_sec = float(progress_interval_sec)
    next_progress_time = time.monotonic() + progress_interval_sec if progress_interval_sec > 0.0 else None
    total_target_episodes = int(task_env.eval_episodes_per_grasp * task_env.num_envs)

    while not task_env.is_grasp_evaluation_complete():
        set_student_encoder_obs_override(player, task_env, use_student_encoder)
        action = player.get_action(obses, is_deterministic=deterministic)
        if record_action_sequence:
            recorded_actions.append(action[0].detach().cpu().clone())
        obses, _, done, _ = player.env_step(player.env, action)

        if player.is_rnn:
            all_done_indices = done.nonzero(as_tuple=False).squeeze(-1)
            if len(all_done_indices) > 0:
                for state in player.states:
                    state[:, all_done_indices, :] = 0.0

        if video_writer is not None:
            if not hasattr(task_env, "get_camera_frame"):
                raise RuntimeError("Task env does not expose get_camera_frame(), so video recording cannot be used.")
            video_writer.append_data(task_env.get_camera_frame(env_id=video_env_index))

        if render_human:
            task_env.render(mode="human")

        if next_progress_time is not None and time.monotonic() >= next_progress_time:
            completed_episodes = int(task_env.eval_episode_counts.sum().item())
            active_envs = int(task_env.eval_active_mask.sum().item())
            max_episode_count = int(task_env.eval_episode_counts.max().item()) if task_env.eval_episode_counts.numel() > 0 else 0
            progress_pct = 100.0 * completed_episodes / max(1, total_target_episodes)
            print(
                f"[eval progress] episodes={completed_episodes}/{total_target_episodes} "
                f"({progress_pct:.1f}%) | active_envs={active_envs}/{task_env.num_envs} "
                f"| max_eps_per_grasp={max_episode_count}/{task_env.eval_episodes_per_grasp}"
            )
            next_progress_time = time.monotonic() + progress_interval_sec

    player.model.a2c_network.actor_encoder_obs_override = None
    if record_action_sequence:
        if recorded_actions:
            task_env.last_eval_action_sequence = torch.stack(recorded_actions, dim=0)
        else:
            task_env.last_eval_action_sequence = torch.zeros(
                (0, task_env.cfg["env"]["numActions"]), dtype=torch.float32
            )
    else:
        task_env.last_eval_action_sequence = None
    return task_env.get_grasp_consecutive_evaluation_stats()
