from pathlib import Path

import torch


def preprocess_train_config(cfg, config_dict):
    train_cfg = config_dict["params"]["config"]
    train_cfg["device"] = cfg.rl_device
    train_cfg["full_experiment_name"] = cfg.get("experiment") or train_cfg["name"]
    return config_dict


def _infer_expl_num_blocks(checkpoint_path: Path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{checkpoint_path}'. "
            "This usually means the file is corrupted, truncated, or not actually a PyTorch checkpoint."
        ) from exc
    if 0 in checkpoint:
        checkpoint = checkpoint[0]
    model_state = checkpoint.get("model", checkpoint)

    candidate_keys = [
        "a2c_network.sigma",
        "sigma",
        "a2c_network.extra_params",
        "extra_params",
    ]
    for key in candidate_keys:
        tensor = model_state.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
            return int(tensor.shape[0])
    return None
