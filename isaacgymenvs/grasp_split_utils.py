from __future__ import annotations

from pathlib import Path


GRASP_SPLIT_SELECT = "selected"
GRASP_SPLIT_CHOICES = ("train", "test", "valid", "select", "selected", "success")
NORMALIZED_GRASP_SPLIT_CHOICES = ("train", "test", "valid", "selected", "success")


def normalize_grasp_split(grasp_split: str) -> str:
    value = str(grasp_split).strip().lower()
    if value == "select":
        return GRASP_SPLIT_SELECT
    if value in NORMALIZED_GRASP_SPLIT_CHOICES:
        return value
    raise ValueError(
        f"Unsupported grasp split '{grasp_split}'. Expected one of: "
        f"{', '.join(GRASP_SPLIT_CHOICES)}."
    )


def resolve_grasp_cache_path(instance_dir: Path, grasp_split: str, *, allow_fallback_root: bool = False) -> Path:
    split = normalize_grasp_split(grasp_split)
    if split == "selected":
        path = instance_dir / "selected_grasps.npy"
    elif split == "success":
        path = instance_dir / "success" / "valid_grasps.npy"
    elif split == "valid":
        path = instance_dir / "valid" / "valid_grasps.npy"
    elif split == "train":
        path = instance_dir / "train" / "valid_grasps.npy"
    elif split == "test":
        path = instance_dir / "test" / "valid_grasps.npy"
    else:
        path = instance_dir / "valid_grasps.npy"

    if path.exists():
        return path

    raise FileNotFoundError(
        f"Requested grasp split '{split}' but cache file does not exist: {path}"
    )
