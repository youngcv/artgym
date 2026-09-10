import os


def register_omegaconf_resolvers():
    from omegaconf import OmegaConf

    def _register_resolver(name, fn):
        try:
            OmegaConf.register_new_resolver(name, fn)
        except ValueError:
            pass

    _register_resolver("eq", lambda x, y: x.lower() == y.lower())
    _register_resolver("contains", lambda x, y: x.lower() in y.lower())
    _register_resolver("if", lambda pred, a, b: a if pred else b)
    _register_resolver("resolve_default", lambda default, arg: default if arg == "" else arg)


def make(
    seed: int,
    task: str,
    num_envs: int,
    sim_device: str,
    rl_device: str,
    graphics_device_id: int = -1,
    headless: bool = False,
    multi_gpu: bool = False,
    virtual_screen_capture: bool = False,
    force_render: bool = True,
    cfg=None,
):
    try:
        import isaacgym  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "isaacgym is required to create simulation environments via isaacgymenvs.make(...)."
        ) from exc
    import hydra
    from hydra import compose, initialize
    from hydra.core.hydra_config import HydraConfig

    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    register_omegaconf_resolvers()

    from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator

    if cfg is None:
        if HydraConfig.initialized():
            task = HydraConfig.get().runtime.choices["task"]
            hydra.core.global_hydra.GlobalHydra.instance().clear()

        with initialize(config_path="./cfg"):
            cfg = compose(config_name="config", overrides=[f"task={task}"])
            full_cfg = omegaconf_to_dict(cfg)
            cfg_dict = full_cfg["task"]
            cfg_dict["env"]["numEnvs"] = num_envs
    else:
        full_cfg = omegaconf_to_dict(cfg)
        asset_dir = full_cfg.get("asset_dir", "")
        if asset_dir:
            full_cfg["object"]["asset"]["asset_root"] = f"assets/objects/{asset_dir}"
        cfg_dict = full_cfg["task"]

    if "hand" in full_cfg:
        cfg_dict["hand"] = full_cfg["hand"]

    if "object" in full_cfg:
        cfg_dict["object"] = full_cfg["object"]

    if "train" in full_cfg:
        cfg_dict["train"] = full_cfg["train"]

    if "experiment" in full_cfg:
        cfg_dict["experiment"] = full_cfg["experiment"]

    if multi_gpu:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        cfg_dict["rank"] = local_rank

    create_rlgpu_env = get_rlgames_env_creator(
        seed=seed,
        task_config=cfg_dict,
        task_name=cfg_dict["name"],
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=graphics_device_id,
        headless=headless,
        multi_gpu=multi_gpu,
        virtual_screen_capture=virtual_screen_capture,
        force_render=force_render,
    )

    return create_rlgpu_env()
