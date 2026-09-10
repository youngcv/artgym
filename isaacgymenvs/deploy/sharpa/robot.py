from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from isaacgymenvs.deploy.real_robot_policy_api import HandAPI


DEFAULT_SDK_ROOT = Path("~/Downloads/sharpa/SDK/SharpaWaveSDK_4.6.6").expanduser()
SYSTEM_SDK_PREFIX = Path("/usr")
SYSTEM_SDK_LIB_ROOT = Path("/usr/lib/sharpa-wave-sdk")
SYSTEM_SDK_PYTHON_ROOT = Path("/usr/lib/sharpa-wave-sdk/python")
SDK_JOINT_COUNT = 22
THUMB_SIM_INDICES = np.asarray([17, 18, 19, 20, 21], dtype=np.int64)
DEFAULT_THUMB_FIRST_SETTLE_SEC = 0.5

REAL_JOINT_NAMES = [
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
]

SIM_JOINT_NAMES = [
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
]

REAL_TO_SDK = {joint_name: joint_idx for joint_idx, joint_name in enumerate(REAL_JOINT_NAMES)}
REAL_TO_SIM_INDEX = {joint_name: SIM_JOINT_NAMES.index(joint_name) for joint_name in REAL_JOINT_NAMES}
SIM_TO_REAL_INDEX = {joint_name: REAL_JOINT_NAMES.index(joint_name) for joint_name in SIM_JOINT_NAMES}


class SharpaRobot(HandAPI):
    num_hand_dofs = len(SIM_JOINT_NAMES)

    def __init__(
        self,
        sdk_root: str | os.PathLike[str] = DEFAULT_SDK_ROOT,
        hand_side: str = "left",
        device_sn: str = "",
        speed_coeff: float = 0.3,
        current_coeff: float = 0.6,
        startup_delay_sec: float = 1.0,
        connect_timeout_sec: float = 15.0,
        enable_collision_protection: bool = False,
        command_interpolation: bool = True,
        reset_joint_position: list[float] | tuple[float, ...] | np.ndarray | None = None,
        reset_sleep_sec: float = 1.5,
        zero_state_on_connect: bool = False,
        zero_state_settle_sec: float = 1.0,
    ) -> None:
        self.sdk_root = None if sdk_root in {None, ""} else Path(sdk_root).expanduser().resolve()
        self.hand_side = hand_side.lower()
        if self.hand_side not in {"left", "right"}:
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")
        self.device_sn = device_sn
        self.speed_coeff = float(speed_coeff)
        self.current_coeff = float(current_coeff)
        self.startup_delay_sec = float(startup_delay_sec)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.enable_collision_protection = bool(enable_collision_protection)
        self.command_interpolation = bool(command_interpolation)
        self.reset_sleep_sec = float(reset_sleep_sec)
        self.zero_state_on_connect = bool(zero_state_on_connect)
        self.zero_state_settle_sec = float(zero_state_settle_sec)
        self.reset_joint_position = None if reset_joint_position is None else np.asarray(reset_joint_position, dtype=np.float32)
        if self.reset_joint_position is not None and self.reset_joint_position.shape != (22,):
            raise ValueError("reset_joint_position must contain 22 joint values for Sharpa.")

        self._sharpa = None
        self.manager = None
        self.wave = None
        self.connected_device_info = None
        self.channel_offset = 5 if self.hand_side == "left" else 0
        self._sdk_command_size = SDK_JOINT_COUNT
        self._resolved_sdk_python_root = None
        self._resolved_sdk_lib_root = None

    @staticmethod
    def _real_to_sim_joint_positions(real_joint_positions: np.ndarray) -> np.ndarray:
        real_joint_positions = np.asarray(real_joint_positions, dtype=np.float32)
        sim_joint_positions = np.zeros(len(SIM_JOINT_NAMES), dtype=np.float32)
        for sim_name, real_idx in SIM_TO_REAL_INDEX.items():
            sim_joint_positions[SIM_JOINT_NAMES.index(sim_name)] = float(real_joint_positions[real_idx])
        return sim_joint_positions

    @staticmethod
    def _sim_to_sdk_joint_targets(sim_joint_targets: np.ndarray) -> np.ndarray:
        sim_joint_targets = np.asarray(sim_joint_targets, dtype=np.float32)
        sdk_joint_targets = np.zeros(SDK_JOINT_COUNT, dtype=np.float32)
        for real_name, sdk_idx in REAL_TO_SDK.items():
            sim_idx = REAL_TO_SIM_INDEX[real_name]
            sdk_joint_targets[sdk_idx] = float(sim_joint_targets[sim_idx])
        return sdk_joint_targets

    def _load_sdk(self):
        if self._sharpa is not None:
            return self._sharpa

        python_root, lib_root = self._resolve_sdk_layout()
        self._resolved_sdk_python_root = python_root
        self._resolved_sdk_lib_root = lib_root

        lib_path = str(lib_root)
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_path not in current_ld_path.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld_path}" if current_ld_path else lib_path

        python_path = str(python_root)
        if python_path not in sys.path:
            sys.path.insert(0, python_path)

        import sharpa

        self._sharpa = sharpa
        return self._sharpa

    @staticmethod
    def _is_valid_python_root(path: Path) -> bool:
        return path.exists() and (path / "sharpa").exists()

    @staticmethod
    def _is_valid_lib_root(path: Path) -> bool:
        return path.exists() and path.is_dir()

    def _layout_from_root(self, root: Path):
        root = root.expanduser().resolve()
        candidates = []

        legacy_python_root = root / "python"
        legacy_lib_root = root / "lib"
        if self._is_valid_python_root(legacy_python_root) and self._is_valid_lib_root(legacy_lib_root):
            candidates.append((legacy_python_root, legacy_lib_root))

        fhs_python_root = root / "lib" / "sharpa-wave-sdk" / "python"
        fhs_lib_root = root / "lib" / "sharpa-wave-sdk"
        if self._is_valid_python_root(fhs_python_root) and self._is_valid_lib_root(fhs_lib_root):
            candidates.append((fhs_python_root, fhs_lib_root))

        if root.name == "python":
            direct_python_root = root
            direct_lib_root = root.parent
            if self._is_valid_python_root(direct_python_root) and self._is_valid_lib_root(direct_lib_root):
                candidates.append((direct_python_root, direct_lib_root))

        if root.name == "sharpa-wave-sdk":
            direct_python_root = root / "python"
            direct_lib_root = root
            if self._is_valid_python_root(direct_python_root) and self._is_valid_lib_root(direct_lib_root):
                candidates.append((direct_python_root, direct_lib_root))

        deduped = []
        seen = set()
        for python_root, lib_root in candidates:
            key = (str(python_root), str(lib_root))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((python_root, lib_root))
        return deduped

    def _resolve_sdk_layout(self):
        search_roots = []
        if self.sdk_root is not None:
            search_roots.append(self.sdk_root)
        search_roots.extend(
            [
                DEFAULT_SDK_ROOT,
                SYSTEM_SDK_PREFIX,
                SYSTEM_SDK_LIB_ROOT,
                SYSTEM_SDK_PYTHON_ROOT,
            ]
        )

        seen_roots = set()
        for root in search_roots:
            root_key = str(root)
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)
            for python_root, lib_root in self._layout_from_root(root):
                return python_root, lib_root

        searched = ", ".join(str(root) for root in search_roots)
        raise FileNotFoundError(
            "Could not locate a usable Sharpa SDK installation. "
            "Supported layouts are: legacy '<sdk_root>/python' + '<sdk_root>/lib', "
            "or the new installed layout '/usr/lib/sharpa-wave-sdk/python' + '/usr/lib/sharpa-wave-sdk'. "
            f"Searched roots: {searched}"
        )

    @staticmethod
    def _check_error(result, action_name: str) -> None:
        code = getattr(result, "code", 0)
        if code != 0:
            message = getattr(result, "message", f"error code {code}")
            raise RuntimeError(f"{action_name} failed: {message}")

    def _match_device(self, info) -> bool:
        sharpa = self._load_sdk()
        if getattr(info, "device_type", None) != sharpa.DeviceType.HAND:
            return False
        if self.device_sn and getattr(info, "sn", "") != self.device_sn:
            return False
        hand_side = getattr(info, "hand_side", None)
        if hand_side is None:
            return not self.device_sn or getattr(info, "sn", "") == self.device_sn
        desired = sharpa.HandSide.LEFT if self.hand_side == "left" else sharpa.HandSide.RIGHT
        return hand_side == desired

    def connect(self) -> None:
        sharpa = self._load_sdk()
        self.manager = sharpa.SharpaWaveManager.get_instance()

        deadline = None if self.connect_timeout_sec <= 0.0 else time.time() + self.connect_timeout_sec
        device_infos = []
        while True:
            device_infos = [info for info in self.manager.get_all_devices() if self._match_device(info)]
            if device_infos:
                break
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for Sharpa {self.hand_side} hand"
                    + (f" with serial {self.device_sn}" if self.device_sn else "")
                )
            time.sleep(0.5)

        self.connected_device_info = device_infos[0]
        self.wave = self.manager.connect(self.connected_device_info.sn)
        self._check_error(self.wave.set_control_source(sharpa.ControlSource.SDK), "set_control_source")
        self._check_error(self.wave.set_control_mode(sharpa.ControlMode.ADMITTANCE), "set_control_mode")
        self._check_error(self.wave.set_speed_coeff(self.speed_coeff), "set_speed_coeff")
        self._check_error(self.wave.set_current_coeff(self.current_coeff), "set_current_coeff")
        self._check_error(self.wave.set_enable_state(True), "set_enable_state")
        if self.enable_collision_protection:
            self.wave.enable_collision_protection(True)
        if not self.wave.start():
            raise RuntimeError("Sharpa hand failed to start.")
        if self.startup_delay_sec > 0.0:
            time.sleep(self.startup_delay_sec)
        if self.zero_state_on_connect:
            self.reset_state_to_zero()

    def reset_state_to_zero(self) -> None:
        if self.wave is None:
            raise RuntimeError("Sharpa hand is not connected.")
        zero_joint_targets = np.zeros((22,), dtype=np.float32)
        self.command_joint_targets(zero_joint_targets)
        if self.zero_state_settle_sec > 0.0:
            time.sleep(self.zero_state_settle_sec)

    def disconnect(self) -> None:
        try:
            if self.wave is not None:
                try:
                    self.wave.set_enable_state(False)
                except Exception:
                    pass
                try:
                    self.wave.stop()
                except Exception:
                    pass
                try:
                    self.wave.destroy()
                except Exception:
                    pass
        finally:
            self.wave = None
            if self.manager is not None:
                try:
                    self.manager.disconnect_all()
                except Exception:
                    pass
            self.manager = None
            self.connected_device_info = None

    def reset(self) -> None:
        if self.reset_joint_position is None:
            print("SharpaRobot.reset(): no reset_joint_position configured, skipping startup motion.")
            return
        self.command_joint_targets(self.reset_joint_position)
        if self.reset_sleep_sec > 0.0:
            time.sleep(self.reset_sleep_sec)

    def get_hand_joint_positions(self):
        if self.wave is None:
            raise RuntimeError("Sharpa hand is not connected.")
        error, angles = self.wave.get_joint_position_rad()
        self._check_error(error, "get_joint_position_rad")
        sdk_joint_positions = np.asarray(angles, dtype=np.float32).reshape(-1)
        if sdk_joint_positions.shape[0] < len(REAL_JOINT_NAMES):
            raise ValueError(f"SDK joint array is too small: expected at least {len(REAL_JOINT_NAMES)}, got {sdk_joint_positions.shape}")
        real_joint_positions = sdk_joint_positions[: len(REAL_JOINT_NAMES)]
        sim_joint_positions = self._real_to_sim_joint_positions(real_joint_positions)
        if sim_joint_positions.shape != (22,):
            raise ValueError(f"Expected sim joint_targets shape (22,), got {sim_joint_positions.shape}")
        return sim_joint_positions

    def command_init_grasp(
        self,
        init_grasp,
        *,
        thumb_first_settle_sec: float = DEFAULT_THUMB_FIRST_SETTLE_SEC,
        final_settle_sec: float = 1.0,
    ) -> None:
        init_grasp = np.asarray(init_grasp, dtype=np.float32)
        expected_shape = (self.get_num_hand_dofs(),)
        if init_grasp.shape != expected_shape:
            raise ValueError(f"Expected init grasp shape {expected_shape}, got {init_grasp.shape}")

        try:
            current_qpos = np.asarray(self.get_hand_joint_positions(), dtype=np.float32)
            if current_qpos.shape != init_grasp.shape:
                raise ValueError(f"Expected current hand qpos shape {init_grasp.shape}, got {current_qpos.shape}")
            thumb_first_targets = current_qpos.copy()
            thumb_first_targets[THUMB_SIM_INDICES] = init_grasp[THUMB_SIM_INDICES]
            self.command_joint_targets(thumb_first_targets)
            if thumb_first_settle_sec > 0.0:
                time.sleep(float(thumb_first_settle_sec))
        except Exception as exc:
            print(f"Warning: thumb-first init grasp pre-step failed ({exc}). Falling back to direct full-hand init grasp.")

        self.command_joint_targets(init_grasp)
        if final_settle_sec > 0.0:
            time.sleep(float(final_settle_sec))

    def command_joint_targets(self, joint_targets) -> None:
        if self.wave is None:
            raise RuntimeError("Sharpa hand is not connected.")
        joint_targets = np.asarray(joint_targets, dtype=np.float32)
        if joint_targets.shape != (22,):
            raise ValueError(f"Expected joint_targets shape (22,), got {joint_targets.shape}")
        sdk_joint_targets = self._sim_to_sdk_joint_targets(joint_targets)
        if sdk_joint_targets.shape != (self._sdk_command_size,):
            raise ValueError(f"Expected SDK joint_targets shape ({self._sdk_command_size},), got {sdk_joint_targets.shape}")
        result = self.wave.set_joint_position(sdk_joint_targets.tolist(), self.command_interpolation)
        self._check_error(result, "set_joint_position")
