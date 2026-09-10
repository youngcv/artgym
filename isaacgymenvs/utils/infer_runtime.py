from __future__ import annotations


def configure_fixed_grasp_inference_until_terminated(task_env, args, *, policy_name: str):
    if not hasattr(task_env, "configure_fixed_grasp_consecutive_evaluation"):
        return ""
    if not hasattr(task_env, "runtime_grasp_fixed_state") or task_env.runtime_grasp_fixed_state is None:
        raise RuntimeError(
            f"{policy_name} inference now requires a fixed selected grasp. "
            "Please ensure selected_grasps.npy contains exactly one grasp for the requested instance."
        )
    task_env.configure_fixed_grasp_consecutive_evaluation(
        instance_id=args.instance_id,
        grasp_state=task_env.runtime_grasp_fixed_state,
        episodes_per_grasp=1,
    )
    return "consecutive"


def is_inference_eval_complete(task_env):
    if not hasattr(task_env, "is_grasp_evaluation_complete"):
        return False
    return bool(task_env.is_grasp_evaluation_complete())
