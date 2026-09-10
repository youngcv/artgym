from __future__ import annotations


STUDENT_TEMPORAL_OBS_MODE = "proprio_only"
DEFAULT_STUDENT_INIT_OBS_DIM = 57


def validate_student_temporal_obs_mode(mode):
    if mode in (None, "", "auto", STUDENT_TEMPORAL_OBS_MODE):
        return STUDENT_TEMPORAL_OBS_MODE
    raise ValueError(
        f"Unsupported student observation mode metadata: {mode!r}. "
        "Student observations are proprio-only now; re-distill old tactile/contact artifacts."
    )


def get_student_temporal_obs_layout(history_len, proprio_dim_per_step, init_dim=DEFAULT_STUDENT_INIT_OBS_DIM):
    history_len = int(history_len)
    proprio_dim_per_step = int(proprio_dim_per_step)
    init_dim = int(init_dim)
    if history_len <= 0:
        raise ValueError(f"history_len must be positive, got {history_len}")
    if proprio_dim_per_step <= 0:
        raise ValueError(f"proprio_dim_per_step must be positive, got {proprio_dim_per_step}")
    if init_dim <= 0:
        raise ValueError(f"init_dim must be positive, got {init_dim}")

    student_obs_dim = history_len * proprio_dim_per_step + init_dim
    return {
        "mode": STUDENT_TEMPORAL_OBS_MODE,
        "history_len": history_len,
        "proprio_dim_per_step": proprio_dim_per_step,
        "contact_dim_per_step": 0,
        "init_dim": init_dim,
        "student_obs_dim": student_obs_dim,
    }


def infer_student_temporal_obs_layout(student_obs_dim, history_len, proprio_dim_per_step, init_dim=DEFAULT_STUDENT_INIT_OBS_DIM):
    student_obs_dim = int(student_obs_dim)
    base_layout = get_student_temporal_obs_layout(
        history_len=history_len,
        proprio_dim_per_step=proprio_dim_per_step,
        init_dim=init_dim,
    )
    expected_dim = base_layout["init_dim"] + (
        base_layout["history_len"] * base_layout["proprio_dim_per_step"]
    )
    if student_obs_dim != expected_dim:
        raise ValueError(
            "Student observations are proprio-only now. "
            f"Expected student_obs_dim={expected_dim}, got {student_obs_dim}."
        )
    return base_layout
