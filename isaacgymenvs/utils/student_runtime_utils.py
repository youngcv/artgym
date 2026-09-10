from __future__ import annotations

from isaacgymenvs.utils.student_obs_utils import (
    DEFAULT_STUDENT_INIT_OBS_DIM,
    STUDENT_TEMPORAL_OBS_MODE,
    get_student_temporal_obs_layout,
    infer_student_temporal_obs_layout,
    validate_student_temporal_obs_mode,
)


LEGACY_STUDENT_RUNTIME_DEFAULTS = {
    "task": "artmanip",
    "train": "artmanipSAPGPrivLSTMPPO",
    "hand": "sharpa",
    "object": "knife",
}


def build_student_cfg_overrides_from_meta(distill_meta):
    overrides = []
    if not distill_meta:
        return overrides
    proprio_history_len = distill_meta.get("proprio_history_len")
    if proprio_history_len is not None:
        overrides.append(f"task.env.proprioHistoryLen={int(proprio_history_len)}")
    return overrides


def resolve_teacher_identity_from_meta(
    *,
    distill_meta,
    task,
    train,
    hand,
    object_name,
):
    distill_meta = distill_meta or {}

    def _resolve(current_value, meta_key):
        meta_value = distill_meta.get(meta_key)
        if not meta_value:
            return current_value
        legacy_default = LEGACY_STUDENT_RUNTIME_DEFAULTS[meta_key]
        if not current_value or str(current_value) == legacy_default:
            return str(meta_value)
        return current_value

    return {
        "task": _resolve(task, "task"),
        "train": _resolve(train, "train"),
        "hand": _resolve(hand, "hand"),
        "object": _resolve(object_name, "object"),
    }


def resolve_student_runtime_layout(
    *,
    distill_meta,
    fallback_history_len=1,
    fallback_proprio_dim_per_step=0,
    fallback_student_obs_dim=None,
    fallback_init_dim=DEFAULT_STUDENT_INIT_OBS_DIM,
):
    distill_meta = distill_meta or {}
    student_temporal_obs_mode = validate_student_temporal_obs_mode(
        distill_meta.get("student_temporal_obs_mode")
    )
    history_len = int(distill_meta.get("proprio_history_len", fallback_history_len))
    proprio_dim_per_step = int(distill_meta.get("proprio_obs_dim", fallback_proprio_dim_per_step))
    encoder_spec = distill_meta.get("student_encoder_spec") or {}
    init_dim = int(encoder_spec.get("init_dim", fallback_init_dim))

    temporal_layout = get_student_temporal_obs_layout(
        history_len=history_len,
        proprio_dim_per_step=proprio_dim_per_step,
        init_dim=init_dim,
    )
    saved_contact_dim = encoder_spec.get("contact_dim_per_step")
    if saved_contact_dim is not None and int(saved_contact_dim) != 0:
        raise ValueError(
            "Student artifact contains tactile/contact obs dims and must be re-distilled with proprio-only observations."
        )
    student_obs_dim = int(distill_meta.get("student_obs_dim", fallback_student_obs_dim or temporal_layout["student_obs_dim"]))
    if student_obs_dim != temporal_layout["student_obs_dim"]:
        infer_student_temporal_obs_layout(
            student_obs_dim=student_obs_dim,
            history_len=history_len,
            proprio_dim_per_step=proprio_dim_per_step,
            init_dim=init_dim,
        )

    return {
        "student_temporal_obs_mode": STUDENT_TEMPORAL_OBS_MODE,
        "student_obs_dim": int(student_obs_dim),
        "history_len": int(temporal_layout["history_len"]),
        "proprio_dim_per_step": int(temporal_layout["proprio_dim_per_step"]),
        "contact_dim_per_step": int(temporal_layout["contact_dim_per_step"]),
        "init_dim": int(temporal_layout["init_dim"]),
        "encoder_spec": encoder_spec,
    }
