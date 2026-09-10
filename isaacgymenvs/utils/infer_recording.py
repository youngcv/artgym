import torch


def build_reset_first_cur_targets(recorded_cur_targets, task_env):
    cur_targets = torch.stack(recorded_cur_targets, dim=0) if recorded_cur_targets else torch.empty(0)
    if cur_targets.numel() == 0:
        return cur_targets
    if not hasattr(task_env, "init_targets"):
        raise RuntimeError("Task env does not expose init_targets, so reset-first cur_targets cannot be saved.")
    init_targets = task_env.init_targets.detach().cpu()
    num_hand_dofs = init_targets.shape[1]
    cur_targets[0, :, :num_hand_dofs] = init_targets
    return cur_targets

