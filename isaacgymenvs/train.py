import os

import isaacgym  # noqa: F401
import hydra
import torch
from omegaconf import DictConfig

import isaacgymenvs

isaacgymenvs.register_omegaconf_resolvers()


class _RankAwareTrainConfig(dict):
    def __init__(self, *args, rank_device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rank_device = rank_device

    def _rewrite_device_value(self, key, value):
        if (
            key == "device"
            and value == "cuda:0"
            and self._rank_device is not None
            and bool(self.get("multi_gpu", False))
        ):
            return self._rank_device
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, self._rewrite_device_value(key, value))

    def update(self, *args, **kwargs):
        items = dict(*args, **kwargs)
        for key, value in items.items():
            self[key] = value

    def setdefault(self, key, default=None):
        default = self._rewrite_device_value(key, default)
        return super().setdefault(key, default)


def preprocess_train_config(cfg, config_dict):
    train_cfg = config_dict["params"]["config"]
    train_cfg["device"] = cfg.rl_device
    train_cfg["full_experiment_name"] = cfg.get("experiment") or train_cfg["name"]
    return config_dict


def _get_network_inputs(net_cfg):
    input_cfg = net_cfg.get("inputs") or net_cfg.get("input_preprocessors")
    if input_cfg is None:
        raise KeyError(
            f"network `{net_cfg.name}` requires `inputs` (or legacy `input_preprocessors`) when using dict observations"
        )
    return list(input_cfg.keys())


def _network_uses_concat(net_cfg):
    return net_cfg.name not in {"complex_net", "actor_critic_dict"}


def _validate_multi_gpu_launch(cfg):
    if not cfg.multi_gpu:
        return

    missing = [name for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE") if os.getenv(name) is None]
    if missing:
        raise RuntimeError(
            "multi_gpu=True requires launching with torchrun so "
            f"{', '.join(missing)} are defined. "
            "Example: torchrun --standalone --nnodes=1 --nproc_per_node=4 "
            "-m isaacgymenvs.train task=artmanip ... multi_gpu=True"
        )


def _init_multi_gpu_runtime(cfg):
    if not cfg.multi_gpu:
        return {"enabled": False, "local_rank": 0, "global_rank": 0, "world_size": 1}

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if not torch.cuda.is_available():
        raise RuntimeError("multi_gpu=True requires CUDA, but torch.cuda.is_available() is False.")

    torch.cuda.set_device(local_rank)
    cfg.sim_device = f"cuda:{local_rank}"
    cfg.rl_device = f"cuda:{local_rank}"
    cfg.graphics_device_id = local_rank

    return {
        "enabled": True,
        "local_rank": local_rank,
        "global_rank": global_rank,
        "world_size": world_size,
    }


def _is_rank0(cfg):
    if not cfg.multi_gpu:
        return True
    return int(os.getenv("RANK", "0")) == 0


def _wrap_rlgames_train_config_for_rank(cfg, rlg_config_dict):
    params = rlg_config_dict.get("params", {})
    train_cfg = params.get("config")
    if not isinstance(train_cfg, dict):
        return rlg_config_dict
    if isinstance(train_cfg, _RankAwareTrainConfig):
        train_cfg._rank_device = cfg.rl_device
        return rlg_config_dict

    params["config"] = _RankAwareTrainConfig(train_cfg, rank_device=cfg.rl_device)
    return rlg_config_dict


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    import gym
    from datetime import datetime

    import isaacgymenvs

    from isaacgymenvs.learning import a2c_dict_network_builder, a2c_sapg_priv_network_builder
    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.rlgames_utils import (
        ComplexObsRLGPUEnv,
        RLGPUAlgoObserver,
        RLGPUEnv,
    )
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from rl_games.algos_torch import model_builder
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    _validate_multi_gpu_launch(cfg)
    dist_runtime = _init_multi_gpu_runtime(cfg)
    cfg_dict = omegaconf_to_dict(cfg)
    if _is_rank0(cfg):
        print_dict(cfg_dict)
    set_np_formatting()
    cfg.seed = set_seed(
        cfg.seed,
        torch_deterministic=cfg.torch_deterministic,
        rank=dist_runtime["global_rank"],
    )
    run_name = f"{cfg.get('experiment') or cfg.task_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    def create_env(**kwargs):
        env = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            env.is_vector_env = True
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return env

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: create_env(**kwargs),
        },
    )

    env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = (
        (hasattr(env_cls, "dict_obs_cls") and env_cls.dict_obs_cls)
        or cfg.task.env.get("use_dict_obs", False)
    )

    if dict_cls:
        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec["obs"] = {
            "names": _get_network_inputs(actor_net_cfg),
            "concat": _network_uses_concat(actor_net_cfg),
            "space_name": "observation_space",
        }
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec["states"] = {
                "names": _get_network_inputs(critic_net_cfg),
                "concat": _network_uses_concat(critic_net_cfg),
                "space_name": "state_space",
            }
        vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
    else:
        vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = preprocess_train_config(cfg, omegaconf_to_dict(cfg.train))
    rlg_config_dict = _wrap_rlgames_train_config_for_rank(cfg, rlg_config_dict)

    model_builder.register_network("actor_critic_dict", a2c_dict_network_builder.A2CBuilder)
    model_builder.register_network("actor_critic_sapg_priv", a2c_sapg_priv_network_builder.A2CSAPGPrivBuilder)
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()
    runner.run(
        {
            "train": not cfg.test,
            "play": cfg.test,
            "checkpoint": cfg.checkpoint,
            "sigma": cfg.sigma if cfg.sigma != "" else None,
        }
    )


if __name__ == "__main__":
    launch_rlg_hydra()
