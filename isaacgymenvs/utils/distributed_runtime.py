import os


TORCHRUN_ENV_KEYS = (
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def force_single_process_env():
    for key in TORCHRUN_ENV_KEYS:
        os.environ.pop(key, None)
