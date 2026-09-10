import argparse
import copy
import os
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

try:
    import isaacgym  # noqa: F401
except ModuleNotFoundError:
    isaacgym = None
import numpy as np
from hydra import compose, initialize
from omegaconf import open_dict

from isaacgymenvs.grasp_split_utils import GRASP_SPLIT_CHOICES, normalize_grasp_split, resolve_grasp_cache_path
from isaacgymenvs.consecutive_eval_utils import aggregate_consecutive_stats, compute_grasp_consecutive_metric_means
from isaacgymenvs.student_eval_utils import run_grasp_evaluation_loop
from isaacgymenvs.utils.player_utils import init_player_rnn_for_batch, parse_bool_arg
from isaacgymenvs.utils.student_obs_utils import (
    DEFAULT_STUDENT_INIT_OBS_DIM,
    STUDENT_TEMPORAL_OBS_MODE,
    get_student_temporal_obs_layout,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

def preprocess_train_config(cfg, config_dict):
    train_cfg = config_dict["params"]["config"]
    train_cfg["device"] = cfg.rl_device
    train_cfg["device_name"] = cfg.rl_device
    train_cfg["full_experiment_name"] = cfg.get("experiment") or train_cfg["name"]
    return config_dict


def infer_teacher_algo_family(rlg_config_dict):
    expl_type = str(rlg_config_dict["params"]["config"].get("expl_type", "none") or "none")
    return "ppo" if expl_type == "none" else "sapg"


def _infer_expl_num_blocks(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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


def reset_done_rnn_states(player, done):
    if not player.is_rnn:
        return
    all_done_indices = done.nonzero(as_tuple=False).squeeze(-1)
    if len(all_done_indices) > 0:
        for state in player.states:
            state[:, all_done_indices, :] = 0.0


def reset_module_parameters(module):
    for child in module.modules():
        if child is module:
            continue
        reset_fn = getattr(child, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()


def normalize_obs_slice(model, obs_slice, start_idx):
    if not model.normalize_input:
        return obs_slice

    running_mean_std = model.running_mean_std
    end_idx = start_idx + obs_slice.shape[1]
    if end_idx > running_mean_std.running_mean.shape[0]:
        return obs_slice
    mean = running_mean_std.running_mean[start_idx:end_idx].to(obs_slice.device).float()
    var = running_mean_std.running_var[start_idx:end_idx].to(obs_slice.device).float()
    normalized = (obs_slice - mean) / torch.sqrt(var + running_mean_std.epsilon)
    return torch.clamp(normalized, min=-5.0, max=5.0)


def sync_torch_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _require_torchrun_env(flag_name):
    missing = [name for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE") if os.getenv(name) is None]
    if missing:
        raise RuntimeError(
            f"{flag_name} requires launching with torchrun so {', '.join(missing)} are defined. "
            "Example: torchrun --standalone --nnodes=1 --nproc_per_node=4 -m isaacgymenvs.distill --multi-gpu ..."
        )


def _init_multi_gpu_distillation(args):
    _require_torchrun_env("--multi-gpu")
    if args.pipeline != "gpu":
        raise ValueError("--multi-gpu distillation currently supports only --pipeline gpu.")

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if world_size <= 1:
        raise ValueError("--multi-gpu was set but WORLD_SIZE <= 1. Launch with torchrun using more than one GPU.")

    if not torch.cuda.is_available():
        raise RuntimeError("--multi-gpu distillation requires CUDA, but torch.cuda.is_available() is False.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo", rank=global_rank, world_size=world_size)
    return {
        "enabled": True,
        "local_rank": local_rank,
        "global_rank": global_rank,
        "world_size": world_size,
        "is_rank0": global_rank == 0,
        "sim_device": f"cuda:{local_rank}",
        "rl_device": f"cuda:{local_rank}",
    }


def _single_process_runtime(args):
    return {
        "enabled": False,
        "local_rank": 0,
        "global_rank": 0,
        "world_size": 1,
        "is_rank0": True,
        "sim_device": args.sim_device,
        "rl_device": args.rl_device,
    }


def _dist_reduce_tensor(tensor, op):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


def _dist_reduce_scalar(value, *, reduce_op="mean", device="cpu"):
    backend = dist.get_backend() if dist.is_available() and dist.is_initialized() else None
    reduce_device = device if backend == "nccl" else "cpu"
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=reduce_device)
    if reduce_op == "sum":
        return _dist_reduce_tensor(tensor, dist.ReduceOp.SUM).item()
    if reduce_op == "max":
        return _dist_reduce_tensor(tensor, dist.ReduceOp.MAX).item()
    if reduce_op == "mean":
        tensor = _dist_reduce_tensor(tensor, dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
        return tensor.item()
    raise ValueError(f"Unsupported distributed reduction op: {reduce_op}")


def _unwrap_student_encoder(student_encoder):
    return getattr(student_encoder, "module", student_encoder)


def _average_module_gradients(module):
    if not dist.is_available() or not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    params = list(module.parameters())
    if not params:
        return

    flat_grads = []
    grad_specs = []
    for param in params:
        if param.grad is None:
            flat_grads.append(torch.zeros(param.numel(), dtype=param.dtype, device="cpu"))
            grad_specs.append((param, False))
        else:
            flat_grads.append(param.grad.detach().reshape(-1).to(device="cpu"))
            grad_specs.append((param, True))

    packed = torch.cat(flat_grads, dim=0)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed /= world_size

    offset = 0
    for param, had_grad in grad_specs:
        numel = param.numel()
        if had_grad:
            param.grad.copy_(packed[offset:offset + numel].view_as(param).to(device=param.grad.device, dtype=param.grad.dtype))
        offset += numel


@contextmanager
def _temporarily_clear_torchrun_env():
    keys = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _initialize_linear_modules(module, initializer_cfg):
    from rl_games.algos_torch.network_builder import NetworkBuilder

    base_network = NetworkBuilder.BaseNetwork()
    initializer = base_network.init_factory.create(**dict(initializer_cfg))
    for child in module.modules():
        if isinstance(child, nn.Linear):
            initializer(child.weight)
            if child.bias is not None:
                torch.nn.init.zeros_(child.bias)


def _build_initialized_mlp(input_dim, encoder_cfg, fallback_activation, default_units=None):
    from rl_games.algos_torch.network_builder import NetworkBuilder

    units = list(encoder_cfg.get("units", default_units or []))
    if not units:
        return nn.Identity(), input_dim

    base_network = NetworkBuilder.BaseNetwork()
    student_encoder = base_network._build_mlp(
        input_size=input_dim,
        units=units,
        activation=encoder_cfg.get("activation", fallback_activation),
        dense_func=torch.nn.Linear,
        norm_func_name=encoder_cfg.get("normalization", None),
        d2rl=encoder_cfg.get("d2rl", False),
        norm_only_first_layer=encoder_cfg.get("norm_only_first_layer", False),
    )

    _initialize_linear_modules(student_encoder, encoder_cfg.get("initializer", {"name": "default"}))
    return student_encoder, units[-1]


def _build_activation_module(name, fallback_activation):
    activation_name = (name or fallback_activation or "relu").lower()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "elu":
        return nn.ELU()
    if activation_name == "tanh":
        return nn.Tanh()
    if activation_name == "sigmoid":
        return nn.Sigmoid()
    if activation_name in {"silu", "swish"}:
        return nn.SiLU()
    if activation_name == "gelu":
        return nn.GELU()
    if activation_name in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"Unsupported temporal activation: {activation_name}")


class CausalTemporalConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dilation, activation_name, fallback_activation, dropout):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.activation = _build_activation_module(activation_name, fallback_activation)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.use_residual = input_channels == output_channels

        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        residual = x
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + residual
        return x


class TemporalConvBackbone(nn.Module):
    def __init__(self, input_dim, temporal_cfg, fallback_activation):
        super().__init__()
        hidden_size = int(temporal_cfg.get("hidden_size", 128))
        num_layers = int(temporal_cfg.get("num_layers", 2))
        kernel_size = int(temporal_cfg.get("kernel_size", 3))
        dilation_growth = int(temporal_cfg.get("dilation_growth", 1))
        activation_name = temporal_cfg.get("activation", fallback_activation)
        dropout = float(temporal_cfg.get("dropout", 0.0))

        if hidden_size <= 0:
            raise ValueError(f"Temporal conv hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"Temporal conv num_layers must be positive, got {num_layers}")
        if kernel_size <= 0:
            raise ValueError(f"Temporal conv kernel_size must be positive, got {kernel_size}")
        if dilation_growth <= 0:
            raise ValueError(f"Temporal conv dilation_growth must be positive, got {dilation_growth}")

        layers = []
        current_dim = int(input_dim)
        dilation = 1
        for _ in range(num_layers):
            layers.append(
                CausalTemporalConvBlock(
                    current_dim,
                    hidden_size,
                    kernel_size,
                    dilation,
                    activation_name,
                    fallback_activation,
                    dropout,
                )
            )
            current_dim = hidden_size
            dilation *= dilation_growth
        self.layers = nn.ModuleList(layers)
        self.output_dim = hidden_size

    def forward(self, seq_obs):
        x = seq_obs.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x[:, :, -1]


class CustomCausalTemporalConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dilation, activation_name, fallback_activation, dropout):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_padding = self.dilation * (self.kernel_size - 1)
        self.weight = nn.Parameter(torch.empty(self.output_channels, self.input_channels, self.kernel_size))
        self.bias = nn.Parameter(torch.empty(self.output_channels))
        self.activation = _build_activation_module(activation_name, fallback_activation)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.use_residual = self.input_channels == self.output_channels

        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        residual = x
        padded = F.pad(x, (self.left_padding, 0))
        seq_len = x.shape[-1]
        taps = []
        for tap_idx in range(self.kernel_size):
            start = tap_idx * self.dilation
            taps.append(padded[:, :, start:start + seq_len])
        kernel_view = torch.stack(taps, dim=2)
        x = torch.einsum("bckt,ock->bot", kernel_view, self.weight)
        x = x + self.bias.view(1, -1, 1)
        x = self.activation(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + residual
        return x


class CustomTemporalConvBackbone(nn.Module):
    def __init__(self, input_dim, temporal_cfg, fallback_activation):
        super().__init__()
        hidden_size = int(temporal_cfg.get("hidden_size", 128))
        num_layers = int(temporal_cfg.get("num_layers", 2))
        kernel_size = int(temporal_cfg.get("kernel_size", 3))
        dilation_growth = int(temporal_cfg.get("dilation_growth", 1))
        activation_name = temporal_cfg.get("activation", fallback_activation)
        dropout = float(temporal_cfg.get("dropout", 0.0))

        if hidden_size <= 0:
            raise ValueError(f"Temporal conv hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"Temporal conv num_layers must be positive, got {num_layers}")
        if kernel_size <= 0:
            raise ValueError(f"Temporal conv kernel_size must be positive, got {kernel_size}")
        if dilation_growth <= 0:
            raise ValueError(f"Temporal conv dilation_growth must be positive, got {dilation_growth}")

        layers = []
        current_dim = int(input_dim)
        dilation = 1
        for _ in range(num_layers):
            layers.append(
                CustomCausalTemporalConvBlock(
                    current_dim,
                    hidden_size,
                    kernel_size,
                    dilation,
                    activation_name,
                    fallback_activation,
                    dropout,
                )
            )
            current_dim = hidden_size
            dilation *= dilation_growth
        self.layers = nn.ModuleList(layers)
        self.output_dim = hidden_size

    def forward(self, seq_obs):
        x = seq_obs.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x[:, :, -1]


def _build_temporal_model(input_dim, temporal_cfg, fallback_activation, default_hidden_size):
    temporal_type = str(temporal_cfg.get("type", "tconv")).lower()
    if temporal_type == "tconv":
        temporal_impl = str(temporal_cfg.get("impl", "torch_conv1d")).lower()
        if temporal_impl == "custom_tcn":
            model = CustomTemporalConvBackbone(
                input_dim=input_dim,
                temporal_cfg=temporal_cfg,
                fallback_activation=fallback_activation,
            )
        elif temporal_impl == "torch_conv1d":
            model = TemporalConvBackbone(
                input_dim=input_dim,
                temporal_cfg=temporal_cfg,
                fallback_activation=fallback_activation,
            )
        else:
            raise ValueError(f"Unsupported temporal backbone implementation: {temporal_impl}")
        return model, model.output_dim
    raise ValueError(f"Unsupported temporal backbone type: {temporal_type}")


def _forward_temporal_model(temporal_model, temporal_input):
    return temporal_model(temporal_input)


def build_student_encoder(input_dim, encoder_cfg, fallback_activation):
    student_encoder, _ = _build_initialized_mlp(input_dim, encoder_cfg, fallback_activation)
    return student_encoder


class ProprioInitTemporalStudentEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_spec, fallback_activation):
        super().__init__()

        self.input_dim = int(input_dim)
        self.history_len = int(encoder_spec["history_len"])
        self.proprio_dim_per_step = int(encoder_spec["proprio_dim_per_step"])
        self.init_dim = int(encoder_spec["init_dim"])

        if self.history_len <= 0:
            raise ValueError("proprio_init_temporal student encoder requires history_len > 0")
        if self.proprio_dim_per_step <= 0:
            raise ValueError("proprio_init_temporal student encoder requires proprio_dim_per_step > 0")
        if self.init_dim <= 0:
            raise ValueError("proprio_init_temporal student encoder requires init_dim > 0")

        expected_input_dim = self.history_len * self.proprio_dim_per_step + self.init_dim
        if expected_input_dim != self.input_dim:
            raise ValueError(
                "proprio_init_temporal student encoder expected "
                f"history_len * proprio_dim_per_step + init_dim = {expected_input_dim}, "
                f"but got input_dim={self.input_dim}"
            )

        self.proprio_flat_dim = self.history_len * self.proprio_dim_per_step
        self.proprio_encoder, self.proprio_feature_dim = _build_initialized_mlp(
            self.proprio_dim_per_step,
            encoder_spec["proprio_encoder"],
            fallback_activation,
            default_units=[128, 64],
        )
        self.init_encoder, self.init_feature_dim = _build_initialized_mlp(
            self.init_dim,
            encoder_spec["init_encoder"],
            fallback_activation,
            default_units=[128, 64],
        )

        temporal_input_dim = self.proprio_feature_dim + self.init_feature_dim
        temporal_cfg = dict(encoder_spec.get("temporal", {}))
        self.temporal_model, temporal_output_dim = _build_temporal_model(
            input_dim=temporal_input_dim,
            temporal_cfg=temporal_cfg,
            fallback_activation=fallback_activation,
            default_hidden_size=128,
        )

        latent_head_cfg = dict(encoder_spec.get("latent_head", {}))
        latent_head_units = list(latent_head_cfg.get("units", [128])) + [latent_dim]
        self.latent_head, _ = _build_initialized_mlp(
            temporal_output_dim,
            {**latent_head_cfg, "units": latent_head_units},
            fallback_activation,
        )

    def forward(self, flat_obs):
        batch_size = flat_obs.shape[0]

        proprio_flat = flat_obs[:, :self.proprio_flat_dim]
        init_flat = flat_obs[:, self.proprio_flat_dim:self.proprio_flat_dim + self.init_dim]
        proprio_seq = proprio_flat.view(batch_size, self.history_len, self.proprio_dim_per_step)

        proprio_feat = self.proprio_encoder(
            proprio_seq.reshape(batch_size * self.history_len, self.proprio_dim_per_step)
        )
        proprio_feat = proprio_feat.view(batch_size, self.history_len, self.proprio_feature_dim)
        init_feat = self.init_encoder(init_flat).unsqueeze(1).expand(-1, self.history_len, -1)
        temporal_input = torch.cat([proprio_feat, init_feat], dim=-1)

        temporal_feat = _forward_temporal_model(self.temporal_model, temporal_input)
        return self.latent_head(temporal_feat)


def _clone_encoder_cfg(encoder_cfg):
    return copy.deepcopy(dict(encoder_cfg or {}))


def _resolve_temporal_cfg(temporal_cfg, *, custom_tcn=False):
    resolved = _clone_encoder_cfg(temporal_cfg)
    resolved.setdefault("type", "tconv")
    if str(resolved.get("type", "tconv")).lower() == "tconv":
        resolved["impl"] = "custom_tcn" if custom_tcn else str(resolved.get("impl", "torch_conv1d")).lower()
    return resolved


def resolve_student_encoder_spec(
    student_obs_dim,
    proprio_obs_dim,
    proprio_history_len,
    sapg_priv_cfg,
    requested_type=None,
    custom_tcn=False,
):
    student_encoder_cfg = _clone_encoder_cfg(sapg_priv_cfg.get("student_encoder", {}))
    requested_type = requested_type or student_encoder_cfg.get("type")
    temporal_layout = get_student_temporal_obs_layout(
        history_len=proprio_history_len,
        proprio_dim_per_step=proprio_obs_dim,
        init_dim=int(student_encoder_cfg.get("init_dim", DEFAULT_STUDENT_INIT_OBS_DIM)),
    )
    if temporal_layout["student_obs_dim"] != student_obs_dim:
        raise ValueError(
            "studentObsDim must match the proprio-only temporal observation layout: "
            f"expected {temporal_layout['student_obs_dim']}, got {student_obs_dim}."
        )
    if requested_type in (None, "", "auto"):
        encoder_type = "proprio_init_temporal"
    else:
        encoder_type = requested_type

    if encoder_type == "proprio_init_temporal":
        return {
            "type": "proprio_init_temporal",
            "history_len": int(temporal_layout["history_len"]),
            "proprio_dim_per_step": int(temporal_layout["proprio_dim_per_step"]),
            "init_dim": int(temporal_layout["init_dim"]),
            "proprio_encoder": _clone_encoder_cfg(student_encoder_cfg.get("proprio_encoder", {})),
            "init_encoder": _clone_encoder_cfg(student_encoder_cfg.get("init_encoder", {})),
            "temporal": _resolve_temporal_cfg(student_encoder_cfg.get("temporal", {}), custom_tcn=custom_tcn),
            "latent_head": _clone_encoder_cfg(student_encoder_cfg.get("latent_head", {})),
        }

    if encoder_type != "mlp":
        raise ValueError(
            f"Unsupported student encoder type: {encoder_type}. "
            "Student distillation now supports mlp and proprio_init_temporal only."
        )

    return {"type": "mlp"}


def build_student_encoder_from_spec(student_obs_dim, latent_dim, encoder_cfg, fallback_activation, encoder_spec):
    encoder_type = encoder_spec["type"]
    if encoder_type == "proprio_init_temporal":
        return ProprioInitTemporalStudentEncoder(
            input_dim=student_obs_dim,
            latent_dim=latent_dim,
            encoder_spec=encoder_spec,
            fallback_activation=fallback_activation,
        )
    return build_student_encoder(
        input_dim=student_obs_dim,
        encoder_cfg=encoder_cfg,
        fallback_activation=fallback_activation,
    )


def transplant_encoder_weights(student_encoder, teacher_encoder):
    student_linears = [m for m in student_encoder.modules() if isinstance(m, nn.Linear)]
    teacher_linears = [m for m in teacher_encoder.modules() if isinstance(m, nn.Linear)]

    for student_layer, teacher_layer in zip(student_linears, teacher_linears):
        with torch.no_grad():
            out_dim = min(student_layer.weight.shape[0], teacher_layer.weight.shape[0])
            in_dim = min(student_layer.weight.shape[1], teacher_layer.weight.shape[1])
            student_layer.weight[:out_dim, :in_dim].copy_(teacher_layer.weight[:out_dim, :in_dim])
            if student_layer.bias is not None and teacher_layer.bias is not None:
                student_layer.bias[:out_dim].copy_(teacher_layer.bias[:out_dim])


def load_student_encoder_state(student_encoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("student_encoder_state_dict", checkpoint)
    student_encoder.load_state_dict(state_dict)


def set_module_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def build_output_payload(teacher_checkpoint_path, player_model, student_encoder, distill_meta, include_full_model):
    if include_full_model:
        checkpoint = torch.load(teacher_checkpoint_path, map_location="cpu")
        if 0 in checkpoint:
            checkpoint = checkpoint[0]
        model_state_dict = player_model.state_dict()
        if getattr(student_encoder, "module", None) is not None and _unwrap_student_encoder(student_encoder) is not student_encoder:
            model_state_dict = {
                key.replace("a2c_network.priv_encoder.module.", "a2c_network.priv_encoder."): value
                for key, value in model_state_dict.items()
            }
        checkpoint["model"] = model_state_dict
        checkpoint["student_encoder_state_dict"] = _unwrap_student_encoder(student_encoder).state_dict()
        checkpoint["distill_meta"] = distill_meta
        return checkpoint

    return {
        "student_encoder_state_dict": _unwrap_student_encoder(student_encoder).state_dict(),
        "distill_meta": distill_meta,
    }


def build_distill_meta(
    checkpoint_path,
    args,
    teacher_algo_family,
    policy_obs_dim,
    teacher_obs_dim,
    student_obs_dim,
    student_encoder_type,
    student_encoder_spec,
    proprio_obs_dim,
    proprio_history_len,
    selected_block_idx,
    best_loss,
    best_reward,
):
    return {
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_algo_family": str(teacher_algo_family),
        "task": args.task,
        "train": args.train,
        "hand": args.hand,
        "object": args.object,
        "policy_obs_dim": policy_obs_dim,
        "teacher_obs_dim": teacher_obs_dim,
        "student_obs_dim": student_obs_dim,
        "student_temporal_obs_mode": STUDENT_TEMPORAL_OBS_MODE,
        "student_encoder_type": student_encoder_type,
        "student_encoder_spec": copy.deepcopy(student_encoder_spec),
        "proprio_obs_dim": proprio_obs_dim,
        "proprio_history_len": proprio_history_len,
        "expl_block_idx": selected_block_idx,
        "updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "lr": args.lr,
        "cosine_coef": args.cosine_coef,
        "custom_tcn": bool(args.custom_tcn),
        "best_loss": best_loss,
        "best_reward": best_reward,
        "selected_only": bool(args.grasp_split == "selected"),
        "object_randomize": bool(getattr(args, "object_randomize", True)),
        "grasp_split": str(args.grasp_split),
        "deterministic": bool(args.deterministic),
        "goal_logic": "training_success_immediate_switch_timeout_reset",
        "success_hold_duration": float(getattr(args, "success_hold_duration", 0.0)),
        "goal_switch_timeout_sec": float(getattr(args, "goal_switch_timeout_sec", 0.0)),
    }


def save_distilled_checkpoint(
    output_path,
    checkpoint_path,
    player_model,
    student_encoder,
    distill_meta,
    include_full_model,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = build_output_payload(
        teacher_checkpoint_path=checkpoint_path,
        player_model=player_model,
        student_encoder=student_encoder,
        distill_meta=distill_meta,
        include_full_model=include_full_model,
    )
    torch.save(checkpoint, output_path)
    print(f"Saved distilled checkpoint to {output_path}")


def _resolve_teacher_run_dir(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parent
    if checkpoint_path.stem == "model" and run_dir.name in {"best", "last", "nn", "checkpoint", "checkpoints"}:
        run_dir = run_dir.parent
    return run_dir


def resolve_output_checkpoint_path(output_checkpoint, checkpoint_path):
    if output_checkpoint:
        return Path(output_checkpoint)
    return _resolve_teacher_run_dir(checkpoint_path) / "student" / f"{STUDENT_TEMPORAL_OBS_MODE}.pth"


def resolve_summary_dir(output_checkpoint, checkpoint_path):
    output_path = resolve_output_checkpoint_path(output_checkpoint, checkpoint_path)
    return output_path.parent / f"{output_path.stem}_summaries"


def resolve_best_reward_checkpoint_path(output_checkpoint, checkpoint_path):
    output_path = resolve_output_checkpoint_path(output_checkpoint, checkpoint_path)
    return output_path.with_name(f"{output_path.stem}_best_reward{output_path.suffix}")


def _flatten_dict(d, prefix='', separator='/'):
    result = {}
    for key, value in d.items():
        flat_key = prefix + key
        if isinstance(value, dict):
            result.update(_flatten_dict(value, flat_key + separator, separator))
        else:
            result[flat_key] = value
    return result


class DistillTensorboardLogger:
    def __init__(self, summary_dir, games_to_track=3000, enabled=True):
        self.enabled = bool(enabled)
        self.summary_dir = Path(summary_dir)
        if self.enabled:
            self.summary_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.summary_dir))
        else:
            self.writer = None
        self.games_to_track = int(games_to_track)
        self.current_rewards = None
        self.current_lengths = None
        self.episode_rewards = deque(maxlen=self.games_to_track)
        self.episode_lengths = deque(maxlen=self.games_to_track)
        self.episode_cumulative = {}
        self.episode_cumulative_avg = {}
        self.ep_infos = []
        self.last_direct_info = {}
        self.new_finished_episodes = False

    def _ensure_batch(self, batch_size):
        if not self.enabled:
            return
        if self.current_rewards is None or self.current_rewards.shape[0] != batch_size:
            self.current_rewards = torch.zeros(batch_size, dtype=torch.float32)
            self.current_lengths = torch.zeros(batch_size, dtype=torch.float32)

    def process_step(self, rewards, dones, infos):
        if not self.enabled:
            return
        rewards_cpu = rewards.detach().cpu().view(-1).float()
        dones_cpu = dones.detach().cpu().view(-1).bool()
        self._ensure_batch(rewards_cpu.shape[0])
        self.current_rewards += rewards_cpu
        self.current_lengths += 1.0
        done_indices = dones_cpu.nonzero(as_tuple=False).view(-1)

        if isinstance(infos, dict):
            if 'episode' in infos:
                self.ep_infos.append(infos['episode'])
            if 'episode_cumulative' in infos:
                for key, value in infos['episode_cumulative'].items():
                    value_cpu = value.detach().cpu().view(-1).float() if isinstance(value, torch.Tensor) else torch.as_tensor(value).view(-1).float()
                    if key not in self.episode_cumulative:
                        self.episode_cumulative[key] = torch.zeros_like(value_cpu)
                    self.episode_cumulative[key] += value_cpu
                for done_idx in done_indices.tolist():
                    self.new_finished_episodes = True
                    for key in infos['episode_cumulative'].keys():
                        if key not in self.episode_cumulative_avg:
                            self.episode_cumulative_avg[key] = deque([], maxlen=self.games_to_track)
                        self.episode_cumulative_avg[key].append(self.episode_cumulative[key][done_idx].item())
                        self.episode_cumulative[key][done_idx] = 0.0
            infos_flat = _flatten_dict(infos, prefix='', separator='/') if len(infos) > 0 else {}
            self.last_direct_info = {}
            for key, value in infos_flat.items():
                if key.startswith('episode/') or key.startswith('episode_cumulative/'):
                    continue
                if isinstance(value, (float, int)):
                    self.last_direct_info[key] = float(value)
                elif isinstance(value, torch.Tensor):
                    value_cpu = value.detach().cpu().float()
                    if value_cpu.numel() > 0:
                        self.last_direct_info[key] = value_cpu.mean().item()
                elif isinstance(value, np.ndarray):
                    value_np = np.asarray(value, dtype=np.float32)
                    if value_np.size > 0:
                        self.last_direct_info[key] = float(value_np.mean())

        for done_idx in done_indices.tolist():
            self.episode_rewards.append(self.current_rewards[done_idx].item())
            self.episode_lengths.append(self.current_lengths[done_idx].item())
            self.current_rewards[done_idx] = 0.0
            self.current_lengths[done_idx] = 0.0

    def log_update(self, frame, epoch_num, total_time, metrics):
        if not self.enabled:
            return
        for key, value in metrics.items():
            self.writer.add_scalar(f'{key}/frame', value, frame)
            self.writer.add_scalar(f'{key}/iter', value, epoch_num)
            self.writer.add_scalar(f'{key}/time', value, total_time)

        if self.episode_rewards:
            mean_reward = float(np.mean(self.episode_rewards))
            self.writer.add_scalar('rewards/step', mean_reward, frame)
            self.writer.add_scalar('rewards/iter', mean_reward, epoch_num)
            self.writer.add_scalar('rewards/time', mean_reward, total_time)
            self.writer.add_scalar('shaped_rewards/step', mean_reward, frame)
            self.writer.add_scalar('shaped_rewards/iter', mean_reward, epoch_num)
            self.writer.add_scalar('shaped_rewards/time', mean_reward, total_time)
        if self.episode_lengths:
            mean_length = float(np.mean(self.episode_lengths))
            self.writer.add_scalar('episode_lengths/step', mean_length, frame)
            self.writer.add_scalar('episode_lengths/iter', mean_length, epoch_num)
            self.writer.add_scalar('episode_lengths/time', mean_length, total_time)

        if self.ep_infos:
            episode_scalar_map = {}
            for ep_info in self.ep_infos:
                if not isinstance(ep_info, dict):
                    continue
                for key, value in ep_info.items():
                    if isinstance(value, torch.Tensor):
                        if value.ndim == 0:
                            scalar = value.item()
                        elif value.numel() == 1:
                            scalar = value.view(-1)[0].item()
                        else:
                            continue
                    elif isinstance(value, (float, int)):
                        scalar = float(value)
                    else:
                        continue
                    episode_scalar_map.setdefault(key, []).append(scalar)
            for key, values in episode_scalar_map.items():
                self.writer.add_scalar(f'Episode/{key}', float(np.mean(values)), epoch_num)
            self.ep_infos.clear()

        if self.new_finished_episodes:
            for key, values in self.episode_cumulative_avg.items():
                if len(values) == 0:
                    continue
                self.writer.add_scalar(f'episode_cumulative/{key}', float(np.mean(values)), frame)
                self.writer.add_scalar(f'episode_cumulative_min/{key}_min', float(np.min(values)), frame)
                self.writer.add_scalar(f'episode_cumulative_max/{key}_max', float(np.max(values)), frame)
            self.new_finished_episodes = False

        for key, value in self.last_direct_info.items():
            self.writer.add_scalar(f'{key}/frame', value, frame)
            self.writer.add_scalar(f'{key}/iter', value, epoch_num)
            self.writer.add_scalar(f'{key}/time', value, total_time)
        self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()


def get_encoder_observations(task_env):
    if hasattr(task_env, "get_student_encoder_observations"):
        student_obs = task_env.get_student_encoder_observations()
    else:
        student_obs = task_env.student_obs_buf

    if hasattr(task_env, "get_teacher_encoder_observations"):
        teacher_obs = task_env.get_teacher_encoder_observations()
    else:
        teacher_obs = task_env.teacher_privileged_obs_buf

    return student_obs, teacher_obs


def parse_args():
    parser = argparse.ArgumentParser(description="Online latent distillation for a frozen teacher policy and a trainable student encoder.")
    parser.add_argument("--checkpoint", required=True, help="Path to the teacher checkpoint.")
    parser.add_argument("--task", default="artmanip", help="Task config name.")
    parser.add_argument("--train", default="artmanipSAPGPrivLSTMPPO", help="Train config name.")
    parser.add_argument("--hand", default="sharpa", help="Hand config name.")
    parser.add_argument("--object", default="knife", help="Object config name.")
    parser.add_argument("--asset-dir", default="", help="Optional asset directory under assets/objects, e.g. knife_multi.")
    parser.add_argument("--num-envs", type=int, default=32, help="How many envs to use for rollout collection.")
    parser.add_argument("--grasp-split", choices=GRASP_SPLIT_CHOICES, default="train", help="Which grasp split to use for distillation rollout sampling.")
    parser.add_argument("--updates", type=int, default=200, help="How many distillation updates to run.")
    parser.add_argument("--rollout-steps", type=int, default=16, help="How many rollout steps to collect per update.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Student encoder learning rate.")
    parser.add_argument("--cosine-coef", type=float, default=1.0, help="Weight on cosine latent loss.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions during rollout.")
    parser.add_argument("--expl-block-idx", type=int, default=0, help="Which teacher exploration block id to use when applicable.")
    parser.add_argument("--sim-device", default="cuda:0", help="Simulation device.")
    parser.add_argument("--rl-device", default="cuda:0", help="RL/model device.")
    parser.add_argument("--pipeline", choices=["gpu", "cpu"], default="gpu", help="Isaac Gym pipeline mode.")
    parser.add_argument("--graphics-device-id", type=int, default=0, help="Graphics device id.")
    parser.add_argument("--multi-gpu", action="store_true", help="Use one distillation process per GPU via torchrun on a single machine.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--student-init", choices=["teacher", "random"], default="teacher", help="How to initialize the student encoder.")
    parser.add_argument("--student-checkpoint", default="", help="Optional path to an existing student distillation artifact.")
    parser.add_argument(
        "--save-every-updates",
        type=int,
        default=0,
        help="If > 0, save an intermediate checkpoint every N updates using the resolved output path as the base name.",
    )
    parser.add_argument(
        "--save-best-after-updates",
        type=int,
        default=0,
        help="Only start updating the best-reward distilled checkpoint after this many updates.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping for the student encoder.")
    parser.add_argument(
        "--custom_tcn",
        action="store_true",
        help="Use the custom causal TCN backend for student temporal encoders instead of nn.Conv1d.",
    )
    parser.add_argument(
        "--output-checkpoint",
        default="",
        help="Optional path to save the distilled checkpoint. Defaults to <teacher_run>/student/proprio_only.pth.",
    )
    parser.add_argument("--eval-every-updates", type=int, default=0, help="If > 0, run eval-style student validation every N updates.")
    parser.add_argument("--eval-episodes-per-grasp", type=int, default=10, help="Episodes per grasp for periodic eval-style validation.")
    parser.add_argument(
        "--eval-randomize",
        type=parse_bool_arg,
        default=True,
        help="Whether to enable task.randomize during periodic eval-style validation.",
    )
    parser.add_argument("--eval-instance-id", default="", help="Object instance id to use for periodic eval-style validation.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.grasp_split = normalize_grasp_split(args.grasp_split)
    dist_runtime = _single_process_runtime(args)
    if args.multi_gpu:
        dist_runtime = _init_multi_gpu_distillation(args)
        args.sim_device = dist_runtime["sim_device"]
        args.rl_device = dist_runtime["rl_device"]
        args.graphics_device_id = dist_runtime["local_rank"]

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

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    overrides = [
        f"task={args.task}",
        f"train={args.train}",
        f"hand={args.hand}",
        f"object={args.object}",
        f"asset_dir={args.asset_dir}",
        f"task.env.graspSplit={args.grasp_split}",
        f"headless={args.headless}",
        f"sim_device={args.sim_device}",
        f"rl_device={args.rl_device}",
        f"pipeline={args.pipeline}",
        f"graphics_device_id={args.graphics_device_id}",
        f"seed={args.seed}",
        f"task.env.numEnvs={args.num_envs}",
        f"multi_gpu={str(bool(args.multi_gpu))}",
    ]
    with initialize(version_base="1.1", config_path="./cfg"):
        cfg = compose(config_name="config", overrides=overrides)

    inferred_blocks = _infer_expl_num_blocks(checkpoint_path)

    with open_dict(cfg):
        cfg.test = True
        cfg.checkpoint = str(checkpoint_path)

    set_np_formatting()
    cfg.seed = set_seed(
        cfg.seed,
        torch_deterministic=cfg.torch_deterministic,
        rank=dist_runtime["global_rank"],
    )

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
        raise RuntimeError("Teacher distillation currently supports flat observations only.")

    rlg_config_dict = preprocess_train_config(cfg, omegaconf_to_dict(cfg.train))
    if rlg_config_dict["params"]["network"]["name"] != "actor_critic_sapg_priv":
        raise ValueError(
            "Teacher distillation expects train.params.network.name=actor_critic_sapg_priv "
            "so the rollout policy can swap the encoder input while keeping the teacher policy frozen. "
            "Older PPO checkpoints built with plain actor_critic are not supported and must be retrained."
        )
    player_cfg = rlg_config_dict["params"]["config"].setdefault("player", {})
    teacher_algo_family = infer_teacher_algo_family(rlg_config_dict)
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
        block_idx = max(0, min(args.expl_block_idx, inferred_blocks - 1))
        coef_ids = torch.linspace(50.0, 0.0, inferred_blocks, device=player.device)
        player.intr_reward_coef_embd[:] = coef_ids[block_idx]
        if dist_runtime["is_rank0"]:
            print(f"Using teacher exploration block {block_idx} with id {coef_ids[block_idx].item():.4f}")

    task_env = player.env.env
    if hasattr(task_env, "set_student_encoder_obs_enabled"):
        task_env.set_student_encoder_obs_enabled(True)
    if hasattr(task_env, "set_runtime_grasp_split"):
        task_env.set_runtime_grasp_split(args.grasp_split)
    obses = player.env_reset(player.env)
    batch_size = player.get_batch_size(obses, 1)
    init_player_rnn_for_batch(player, batch_size)

    policy_obs_dim = int(cfg.task.env.policyObsDim)
    teacher_obs_dim = int(cfg.task.env.privilegedObsDim)
    proprio_history_len = int(cfg.task.env.get("proprioHistoryLen", 1))
    proprio_obs_dim = int(cfg.task.env.get("proprioObsDim", 2 * int(cfg.hand.task.numActions)))
    temporal_layout = get_student_temporal_obs_layout(
        history_len=proprio_history_len,
        proprio_dim_per_step=proprio_obs_dim,
        init_dim=DEFAULT_STUDENT_INIT_OBS_DIM,
    )
    student_obs_dim = int(temporal_layout["student_obs_dim"])
    sapg_priv_cfg = rlg_config_dict["params"]["network"]["sapg_priv"]
    encoder_cfg = dict(sapg_priv_cfg.get("encoder", {}))
    student_encoder_spec = resolve_student_encoder_spec(
        student_obs_dim=student_obs_dim,
        proprio_obs_dim=proprio_obs_dim,
        proprio_history_len=proprio_history_len,
        sapg_priv_cfg=sapg_priv_cfg,
        custom_tcn=args.custom_tcn,
    )
    student_encoder_type = student_encoder_spec["type"]
    resolved_output_checkpoint = resolve_output_checkpoint_path(
        output_checkpoint=args.output_checkpoint,
        checkpoint_path=checkpoint_path,
    )
    best_reward_output_checkpoint = resolve_best_reward_checkpoint_path(
        output_checkpoint=args.output_checkpoint,
        checkpoint_path=checkpoint_path,
    )
    summary_dir = resolve_summary_dir(
        output_checkpoint=args.output_checkpoint,
        checkpoint_path=checkpoint_path,
    )
    if dist_runtime["is_rank0"]:
        print(f"Best-loss distilled checkpoint path: {resolved_output_checkpoint}")
        print(f"Best-reward distilled checkpoint path: {best_reward_output_checkpoint}")
        print(f"TensorBoard summary path: {summary_dir}")

    teacher_encoder = copy.deepcopy(player.model.a2c_network.priv_encoder).to(player.device)
    teacher_encoder.eval()
    for param in teacher_encoder.parameters():
        param.requires_grad = False

    teacher_latent_dim = teacher_encoder(torch.zeros(1, teacher_obs_dim, device=player.device)).shape[1]

    if student_encoder_type == "mlp" and student_obs_dim == teacher_obs_dim:
        student_encoder = copy.deepcopy(teacher_encoder)
        if args.student_init == "random":
            reset_module_parameters(student_encoder)
    else:
        student_encoder = build_student_encoder_from_spec(
            student_obs_dim=student_obs_dim,
            latent_dim=teacher_latent_dim,
            encoder_cfg=encoder_cfg,
            fallback_activation=rlg_config_dict["params"]["network"]["mlp"]["activation"],
            encoder_spec=student_encoder_spec,
        ).to(player.device)
        if args.student_init == "teacher":
            transplant_encoder_weights(student_encoder, teacher_encoder)

    if isinstance(student_encoder, nn.Identity) and student_obs_dim != teacher_obs_dim:
        raise ValueError(
            "studentObsDim differs from privilegedObsDim, but the encoder config has no hidden units. "
            "Please provide network.sapg_priv.encoder.units so the student latent can match the teacher latent size."
        )

    if args.student_checkpoint:
        load_student_encoder_state(student_encoder, Path(args.student_checkpoint).resolve())
    set_module_requires_grad(student_encoder, True)
    student_encoder.train()

    with torch.no_grad():
        student_latent_dim = student_encoder(torch.zeros(1, student_obs_dim, device=player.device)).shape[1]
    if student_latent_dim != teacher_latent_dim:
        raise ValueError(
            f"Student latent dim ({student_latent_dim}) must match teacher latent dim ({teacher_latent_dim}). "
            "Adjust network.sapg_priv.encoder.units so both encoders end at the same width."
        )

    trainable_params = [param for param in student_encoder.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("Student encoder has no trainable parameters. Please check the student initialization path.")

    for param in player.model.parameters():
        param.requires_grad = False
    player.model.a2c_network.priv_encoder = student_encoder

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    player.model.eval()
    student_encoder.train()

    total_steps = 0
    best_loss = None
    best_reward = None
    selected_block_idx = 0 if inferred_blocks is None else max(0, min(args.expl_block_idx, inferred_blocks - 1))
    summary_logger = DistillTensorboardLogger(
        summary_dir=summary_dir,
        games_to_track=int(rlg_config_dict["params"]["config"].get("games_to_track", 3000)),
        enabled=dist_runtime["is_rank0"],
    )
    distill_start_time = time.perf_counter()

    if args.eval_every_updates > 0 and not args.eval_instance_id:
        raise ValueError("--eval-instance-id is required when --eval-every-updates > 0")

    args.object_randomize = bool(cfg.task.task.get("randomize", True))
    args.success_hold_duration = float(task_env.success_hold_duration)
    args.goal_switch_timeout_sec = float(task_env.goal_switch_timeout_sec)

    def run_periodic_student_evaluation():
        with _temporarily_clear_torchrun_env():
            eval_overrides = [
                f"task={args.task}",
                f"train={args.train}",
                f"hand={args.hand}",
                f"object={args.object}",
                f"asset_dir={args.asset_dir}",
                "headless=True",
                f"sim_device={args.sim_device}",
                f"rl_device={args.rl_device}",
                f"graphics_device_id={args.graphics_device_id}",
                f"seed={args.seed}",
                "multi_gpu=False",
            ]
            with initialize(version_base="1.1", config_path="./cfg"):
                eval_cfg = compose(config_name="config", overrides=eval_overrides)

            repo_root = Path(__file__).resolve().parent.parent
            grasp_dir = repo_root / "caches" / "initial_grasp" / args.hand / Path(str(eval_cfg.object.asset.asset_root)).name / args.eval_instance_id
            grasp_cache = resolve_grasp_cache_path(grasp_dir, args.grasp_split)
            effective_grasp_split = args.grasp_split
            if not grasp_cache.exists():
                raise FileNotFoundError(f"Could not find grasp file for periodic eval: {grasp_cache}")

            grasp_count = int(torch.from_numpy(np.load(grasp_cache)).shape[0])
            instance_id_list = list(eval_cfg.object.asset.instance_id_list)
            if args.eval_instance_id not in instance_id_list and instance_id_list != [""]:
                raise ValueError(f"instance_id '{args.eval_instance_id}' not found in object config list {instance_id_list}")

            with open_dict(eval_cfg):
                eval_cfg.task.env.numEnvs = grasp_count
                eval_cfg.task.task.randomize = bool(args.eval_randomize)
                eval_cfg.object.asset.instance_id_list = [args.eval_instance_id]
                eval_cfg.test = True
                eval_cfg.checkpoint = str(checkpoint_path)

            eval_cfg.seed = set_seed(eval_cfg.seed, torch_deterministic=eval_cfg.torch_deterministic, rank=0)

            def create_eval_env(**kwargs):
                return isaacgymenvs.make(
                    eval_cfg.seed,
                    eval_cfg.task_name,
                    eval_cfg.task.env.numEnvs,
                    eval_cfg.sim_device,
                    eval_cfg.rl_device,
                    eval_cfg.graphics_device_id,
                    eval_cfg.headless,
                    eval_cfg.multi_gpu,
                    False,
                    eval_cfg.force_render,
                    eval_cfg,
                    **kwargs,
                )

            env_configurations.register(
                "rlgpu",
                {
                    "vecenv_type": "RLGPU",
                    "env_creator": lambda **kwargs: create_eval_env(**kwargs),
                },
            )
            vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

            eval_env_cls = isaacgym_task_map[eval_cfg.task_name]
            if getattr(eval_env_cls, "dict_obs_cls", False):
                raise RuntimeError("periodic student eval currently supports flat observations only.")

            eval_rlg_config_dict = preprocess_train_config(eval_cfg, omegaconf_to_dict(eval_cfg.train))
            eval_player_cfg = eval_rlg_config_dict["params"]["config"].setdefault("player", {})
            if inferred_blocks is not None:
                eval_player_cfg["expl_num_blocks"] = inferred_blocks

            model_builder.register_network("actor_critic_dict", a2c_dict_network_builder.A2CBuilder)
            model_builder.register_network("actor_critic_sapg_priv", a2c_sapg_priv_network_builder.A2CSAPGPrivBuilder)
            eval_runner = Runner()
            eval_runner.load(eval_rlg_config_dict)
            eval_player = eval_runner.create_player()
            eval_player.restore(eval_cfg.checkpoint)

            if inferred_blocks is not None and eval_player.intr_reward_coef_embd is not None:
                coef_ids = torch.linspace(50.0, 0.0, inferred_blocks, device=eval_player.device)
                eval_player.intr_reward_coef_embd[:] = coef_ids[selected_block_idx]

            eval_teacher_latent_dim = eval_player.model.a2c_network.priv_encoder(
                torch.zeros(1, teacher_obs_dim, device=eval_player.device)
            ).shape[1]
            eval_student_encoder = build_student_encoder_from_spec(
                student_obs_dim=student_obs_dim,
                latent_dim=eval_teacher_latent_dim,
                encoder_cfg=encoder_cfg,
                fallback_activation=eval_rlg_config_dict["params"]["network"]["mlp"]["activation"],
                encoder_spec=student_encoder_spec,
            ).to(eval_player.device)
            eval_student_encoder.load_state_dict(copy.deepcopy(_unwrap_student_encoder(student_encoder).state_dict()))
            eval_student_encoder.eval()
            set_module_requires_grad(eval_student_encoder, False)
            eval_player.model.a2c_network.priv_encoder = eval_student_encoder
            eval_player.model.eval()

            eval_task_env = eval_player.env.env
            if hasattr(eval_task_env, "set_student_encoder_obs_enabled"):
                eval_task_env.set_student_encoder_obs_enabled(True)

            object_goals = eval_cfg.object.task.get("goals")
            if object_goals is None:
                raise ValueError("object.task.goals is required.")
            goal_sequence = list(object_goals)
            if len(goal_sequence) != 2:
                raise ValueError(f"object.task.goals must contain exactly two values, got {goal_sequence}")

            trial_stats = []
            for trial_idx in range(args.eval_episodes_per_grasp):
                eval_task_env.configure_grasp_consecutive_evaluation(
                    instance_id=args.eval_instance_id,
                    goal_sequence=tuple(goal_sequence),
                    stage_duration=None,
                    grasp_split=effective_grasp_split,
                )
                run_grasp_evaluation_loop(
                    player=eval_player,
                    task_env=eval_task_env,
                    deterministic=args.deterministic,
                    progress_interval_sec=0.0,
                    use_student_encoder=True,
                    render_human=False,
                )
                trial_stats.append(eval_task_env.get_grasp_consecutive_evaluation_stats())
            stats = aggregate_consecutive_stats(trial_stats)
            del eval_player, eval_runner, eval_task_env
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return stats

    try:
        for update_idx in range(args.updates):
            sync_torch_device(player.device)
            update_start = time.perf_counter()
            rollout_collect_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_mse = 0.0
            total_cos = 0.0
            total_reward = 0.0
            total_loss_steps = 0

            for step_idx in range(args.rollout_steps):
                student_obs, teacher_obs = get_encoder_observations(task_env)
                student_encoder_obs = student_obs.to(player.device)
                teacher_obs = teacher_obs.to(player.device)
                teacher_encoder_obs = normalize_obs_slice(player.model, teacher_obs, policy_obs_dim)
                player.model.a2c_network.actor_encoder_obs_override = student_encoder_obs

                with torch.no_grad():
                    action = player.get_action(obses, is_deterministic=args.deterministic)

                with torch.no_grad():
                    teacher_latent = teacher_encoder(teacher_encoder_obs)
                student_latent = student_encoder(student_encoder_obs)

                mse_loss = F.mse_loss(student_latent, teacher_latent)
                cosine_loss = 1.0 - F.cosine_similarity(student_latent, teacher_latent, dim=-1).mean()
                loss = mse_loss + args.cosine_coef * cosine_loss

                loss.backward()

                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_cos += cosine_loss.item()
                total_loss_steps += 1

                obses, rewards, done, infos = player.env_step(player.env, action)
                total_reward += rewards.mean().item()
                summary_logger.process_step(rewards, done, infos)
                reset_done_rnn_states(player, done)

            rollout_collect_time = time.perf_counter() - rollout_collect_start
            optimize_start = time.perf_counter()
            if dist_runtime["enabled"]:
                _average_module_gradients(student_encoder)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(student_encoder.parameters(), args.max_grad_norm)
            sync_torch_device(player.device)
            optimizer.step()
            sync_torch_device(player.device)
            player.model.a2c_network.actor_encoder_obs_override = None
            optimize_time = time.perf_counter() - optimize_start
            update_total_time = time.perf_counter() - update_start

            device_for_reduce = player.device if torch.device(player.device).type == "cuda" else "cpu"
            reduced_loss_sum = _dist_reduce_scalar(total_loss, reduce_op="sum", device=device_for_reduce)
            reduced_mse_sum = _dist_reduce_scalar(total_mse, reduce_op="sum", device=device_for_reduce)
            reduced_cos_sum = _dist_reduce_scalar(total_cos, reduce_op="sum", device=device_for_reduce)
            reduced_reward_sum = _dist_reduce_scalar(total_reward, reduce_op="sum", device=device_for_reduce)
            reduced_loss_steps = _dist_reduce_scalar(total_loss_steps, reduce_op="sum", device=device_for_reduce)
            rollout_collect_time_max = _dist_reduce_scalar(rollout_collect_time, reduce_op="max", device=device_for_reduce)
            optimize_time_max = _dist_reduce_scalar(optimize_time, reduce_op="max", device=device_for_reduce)
            update_total_time_max = _dist_reduce_scalar(update_total_time, reduce_op="max", device=device_for_reduce)

            loss_denominator = max(int(round(reduced_loss_steps)), 1)
            mean_loss = reduced_loss_sum / loss_denominator
            mean_mse = reduced_mse_sum / loss_denominator
            mean_cos = reduced_cos_sum / loss_denominator
            mean_reward = reduced_reward_sum / max(args.rollout_steps * dist_runtime["world_size"], 1)
            current_frames = batch_size * args.rollout_steps * dist_runtime["world_size"]
            total_steps += current_frames
            collect_fps = current_frames / max(rollout_collect_time_max, 1e-8)
            train_steps_per_sec = current_frames / max(update_total_time_max, 1e-8)
            total_elapsed_time = time.perf_counter() - distill_start_time

            if dist_runtime["is_rank0"]:
                print(
                    f"update={update_idx:04d} "
                    f"loss={mean_loss:.6f} mse={mean_mse:.6f} cos={mean_cos:.6f} "
                    f"reward={mean_reward:.4f} "
                    f"collect_fps={collect_fps:.2f} total_steps={total_steps} "
                    f"train_steps_per_sec={train_steps_per_sec:.2f}"
                )
            loss_improved = (
                (update_idx + 1) >= int(args.save_best_after_updates)
                and (best_loss is None or mean_loss < best_loss)
            )
            if loss_improved:
                best_loss = mean_loss
            reward_improved = (
                (update_idx + 1) >= int(args.save_best_after_updates)
                and (best_reward is None or mean_reward > best_reward)
            )
            if reward_improved:
                best_reward = mean_reward

            if dist_runtime["is_rank0"]:
                update_metrics = {
                    'performance/step_inference_rl_update_fps': train_steps_per_sec,
                    'performance/step_inference_fps': collect_fps,
                    'performance/step_fps': current_frames / max(update_total_time_max, 1e-8),
                    'performance/rl_update_time': optimize_time_max,
                    'performance/step_inference_time': rollout_collect_time_max,
                    'performance/step_time': update_total_time_max,
                    'losses/distill_loss': mean_loss,
                    'losses/distill_mse': mean_mse,
                    'losses/distill_cosine': mean_cos,
                    'info/last_lr': optimizer.param_groups[0]['lr'],
                    'info/epochs': update_idx,
                }
                if best_loss is not None:
                    update_metrics['student/best_loss'] = best_loss
            else:
                update_metrics = None

            if args.eval_every_updates > 0 and (update_idx + 1) % args.eval_every_updates == 0:
                if dist_runtime["enabled"]:
                    dist.barrier()
                if dist_runtime["is_rank0"]:
                    eval_stats = run_periodic_student_evaluation()
                    update_metrics.update(compute_grasp_consecutive_metric_means(eval_stats))
                    print(
                        f"average consecutive success cycles: {update_metrics['eval/average_consecutive_success_cycles']:.4f}\n"
                        f"best consecutive success cycles: {update_metrics['eval/best_consecutive_success_cycles']:.4f}\n"
                        f"average completion goal distance: {update_metrics['eval/mean_completion_goal_distance']:.6f}"
                    )
                if dist_runtime["enabled"]:
                    dist.barrier()

            summary_logger.log_update(
                frame=total_steps,
                epoch_num=update_idx,
                total_time=total_elapsed_time,
                metrics=update_metrics or {},
            )

            if (
                dist_runtime["is_rank0"]
                and args.save_every_updates > 0
                and (update_idx + 1) % args.save_every_updates == 0
            ):
                periodic_path = resolved_output_checkpoint
                periodic_output_path = periodic_path.with_name(
                    f"{periodic_path.stem}_update{update_idx + 1:04d}{periodic_path.suffix}"
                )
                distill_meta = build_distill_meta(
                    checkpoint_path=checkpoint_path,
                    args=args,
                    teacher_algo_family=teacher_algo_family,
                    policy_obs_dim=policy_obs_dim,
                    teacher_obs_dim=teacher_obs_dim,
                    student_obs_dim=student_obs_dim,
                    student_encoder_type=student_encoder_type,
                    student_encoder_spec=student_encoder_spec,
                    proprio_obs_dim=proprio_obs_dim,
                    proprio_history_len=proprio_history_len,
                    selected_block_idx=selected_block_idx,
                    best_loss=best_loss,
                    best_reward=best_reward,
                )
                save_distilled_checkpoint(
                    output_path=periodic_output_path,
                    checkpoint_path=checkpoint_path,
                    player_model=player.model,
                    student_encoder=student_encoder,
                    distill_meta=distill_meta,
                    include_full_model=(student_obs_dim == teacher_obs_dim and student_encoder_type == "mlp"),
                )
            if dist_runtime["is_rank0"] and loss_improved:
                distill_meta = build_distill_meta(
                    checkpoint_path=checkpoint_path,
                    args=args,
                    teacher_algo_family=teacher_algo_family,
                    policy_obs_dim=policy_obs_dim,
                    teacher_obs_dim=teacher_obs_dim,
                    student_obs_dim=student_obs_dim,
                    student_encoder_type=student_encoder_type,
                    student_encoder_spec=student_encoder_spec,
                    proprio_obs_dim=proprio_obs_dim,
                    proprio_history_len=proprio_history_len,
                    selected_block_idx=selected_block_idx,
                    best_loss=best_loss,
                    best_reward=best_reward,
                )
                save_distilled_checkpoint(
                    output_path=resolved_output_checkpoint,
                    checkpoint_path=checkpoint_path,
                    player_model=player.model,
                    student_encoder=student_encoder,
                    distill_meta=distill_meta,
                    include_full_model=(student_obs_dim == teacher_obs_dim and student_encoder_type == "mlp"),
                )
            if dist_runtime["is_rank0"] and reward_improved:
                distill_meta = build_distill_meta(
                    checkpoint_path=checkpoint_path,
                    args=args,
                    teacher_algo_family=teacher_algo_family,
                    policy_obs_dim=policy_obs_dim,
                    teacher_obs_dim=teacher_obs_dim,
                    student_obs_dim=student_obs_dim,
                    student_encoder_type=student_encoder_type,
                    student_encoder_spec=student_encoder_spec,
                    proprio_obs_dim=proprio_obs_dim,
                    proprio_history_len=proprio_history_len,
                    selected_block_idx=selected_block_idx,
                    best_loss=best_loss,
                    best_reward=best_reward,
                )
                save_distilled_checkpoint(
                    output_path=best_reward_output_checkpoint,
                    checkpoint_path=checkpoint_path,
                    player_model=player.model,
                    student_encoder=student_encoder,
                    distill_meta=distill_meta,
                    include_full_model=(student_obs_dim == teacher_obs_dim and student_encoder_type == "mlp"),
                )

    finally:
        summary_logger.close()
        if dist_runtime["enabled"] and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
