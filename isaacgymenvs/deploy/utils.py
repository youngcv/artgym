from __future__ import annotations

import importlib
import inspect
import argparse


def load_class(spec: str, expected_base):
    if ":" not in spec:
        raise ValueError(f"class spec must be module:ClassName, got '{spec}'")
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not issubclass(cls, expected_base):
        raise TypeError(f"{spec} must inherit from {expected_base.__name__}")
    return cls


def resolve_hand_robot_class_spec(hand: str, explicit_spec: str = "") -> str:
    if explicit_spec:
        return explicit_spec
    class_name = "".join(part.capitalize() for part in str(hand).replace("-", "_").split("_")) + "Robot"
    return f"isaacgymenvs.deploy.{hand}.robot:{class_name}"


def resolve_hand_observation_provider_class_spec(hand: str, explicit_spec: str = "") -> str:
    if explicit_spec:
        return explicit_spec
    class_name = (
        "".join(part.capitalize() for part in str(hand).replace("-", "_").split("_"))
        + "StaticTaskObservationProvider"
    )
    return f"isaacgymenvs.deploy.{hand}.observation_provider:{class_name}"


def get_optional_hand_joint_order(hand: str):
    try:
        module = importlib.import_module(f"isaacgymenvs.deploy.{hand}.robot")
    except ModuleNotFoundError:
        return None
    joint_order = getattr(module, "SIM_JOINT_NAMES", None)
    if joint_order is None:
        joint_order = getattr(module, "POLICY_JOINT_NAMES", None)
    return None if joint_order is None else list(joint_order)


def filter_constructor_kwargs(cls, kwargs):
    signature = inspect.signature(cls.__init__)
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs
    valid_names = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in valid_names}


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")
