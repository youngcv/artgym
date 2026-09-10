# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import numpy as np
import os
from copy import deepcopy
from pathlib import Path

from isaacgym import gymtorch
from isaacgym import gymapi
import torch

from isaacgymenvs.grasp_split_utils import normalize_grasp_split, resolve_grasp_cache_path
from isaacgymenvs.utils.dr_utils import apply_random_samples
from isaacgymenvs.utils.torch_jit_utils import (
    scale,
    unscale,
    quat_mul,
    quat_conjugate,
    quat_apply,
    to_torch,
    tensor_clamp,
)
from isaacgymenvs.utils.student_obs_utils import (
    DEFAULT_STUDENT_INIT_OBS_DIM,
    STUDENT_TEMPORAL_OBS_MODE,
    get_student_temporal_obs_layout,
)
from isaacgymenvs.tasks.base.vec_task import VecTask


class ArtManip(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.hand_cfg = cfg["hand"]
        self.object_cfg = cfg["object"]
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self._build_randomization_params()
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        if "reward" not in self.object_cfg:
            raise ValueError(
                f"Object config for '{self.object_cfg['asset']['type']}' must define a reward block."
            )
        self.reward_scales = {k: float(v) for k, v in self.object_cfg["reward"].items()}
        reward_curriculum_cfg = self.cfg["env"].get("rewardWeightCurriculum", {})
        self.reward_weight_curriculum = {
            str(k): float(v) for k, v in reward_curriculum_cfg.items()
        }
        reward_curriculum_schedule_cfg = self.cfg["env"].get("rewardWeightCurriculumSchedule", {})
        self.reward_curriculum_warmup_steps = int(reward_curriculum_schedule_cfg.get("warmupSteps", 0))
        self.reward_curriculum_total_steps = int(reward_curriculum_schedule_cfg.get("totalSteps", 100000))
        invalid_reward_curriculum_terms = sorted(
            term for term in self.reward_weight_curriculum if term not in self.reward_scales
        )
        if invalid_reward_curriculum_terms:
            raise ValueError(
                "rewardWeightCurriculum contains unknown reward term(s): "
                f"{invalid_reward_curriculum_terms}. Available reward terms: "
                f"{sorted(self.reward_scales.keys())}"
            )
        self.reward_scales_current = dict(self.reward_scales)
        self._object_rb_original_props = {}
        self._object_body_dr_first = True

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.joint_noise = float(self.cfg["env"].get("jointNoise", 0.0))
        if not self.randomize:
            self.force_scale = 0.0
            self.joint_noise = 0.0

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.policy_obs_dim = int(self.cfg["env"].get("policyObsDim", 117))
        self.privileged_obs_dim = int(self.cfg["env"].get("privilegedObsDim", 21))
        self.critic_policy_binary_tactile_dim = 5
        self.proprio_history_len = int(self.cfg["env"].get("proprioHistoryLen", 1))
        self.student_temporal_obs_mode = STUDENT_TEMPORAL_OBS_MODE
        self.student_proprio_obs_dim = int(self.hand_cfg["task"]["numActions"]) + int(self.hand_cfg["task"]["numActions"])
        self.student_temporal_obs_layout = get_student_temporal_obs_layout(
            history_len=self.proprio_history_len,
            proprio_dim_per_step=self.student_proprio_obs_dim,
            init_dim=DEFAULT_STUDENT_INIT_OBS_DIM,
        )
        self.student_obs_dim = int(self.student_temporal_obs_layout["student_obs_dim"])
        self.cfg["env"]["studentObsDim"] = self.student_obs_dim
        self.cfg["env"]["proprioObsDim"] = int(self.student_temporal_obs_layout["proprio_dim_per_step"])
        self.cfg["env"]["studentInitObsDim"] = int(self.student_temporal_obs_layout["init_dim"])
        self.student_encoder_obs_enabled = bool(self.cfg["env"].get("enableStudentEncoderObs", False))

        self.cfg["env"]["numObservations"] = (
            self.policy_obs_dim
            + self.privileged_obs_dim
            + self.critic_policy_binary_tactile_dim
        )
        self.cfg["env"]["numStates"] = 0
        self.cfg["env"]["numActions"] = self.hand_cfg['task']['numActions']
        self.hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]


        self.enable_camera = self.cfg['env']['enableCameraSensors']
        self.image_height = self.cfg['env']['camera']['height']
        self.image_width = self.cfg['env']['camera']['width']
        self.use_collision_geometry = self.cfg['env']['camera']['use_collision_geometry']
        self.horizontal_fov = self.cfg['env']['camera']['horizontal_fov']
        self.cam_pos = self.cfg['env']['camera']['cam_pos']
        self.cam_target = self.cfg['env']['camera']['cam_target']
        self.object_is_lighter = str(self.object_cfg['asset'].get('type', '')).strip().lower() == "lighter"
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.teacher_privileged_obs_buf = torch.zeros((self.num_envs, self.privileged_obs_dim), dtype=torch.float, device=self.device)
        self.student_obs_buf = torch.zeros((self.num_envs, self.student_obs_dim), dtype=torch.float, device=self.device)

        self.dt = self.sim_params.dt
        self.control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        control_dt = self.dt * self.control_freq_inv
        if control_dt <= 0.0:
            raise ValueError("dt * controlFrequencyInv must be positive.")


        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self._refresh_gym()

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_hand_dofs]
        self.hand_dof_pos = self.hand_dof_state[..., 0]
        self.hand_dof_vel = self.hand_dof_state[..., 1]
        self.obj_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_hand_dofs:]
        self.obj_dof_pos = self.obj_dof_state[..., 0]
        self.obj_dof_state_vel = self.obj_dof_state[..., 1]
        self.obj_dof_vel = torch.zeros_like(self.obj_dof_pos)
        self.prev_obj_dof_pos = self.obj_dof_pos.clone()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.init_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.dof_actuation_forces = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg["env"]["numActions"]), dtype=torch.float, device=self.device)
        self.prev_reward_targets = torch.zeros((self.num_envs, self.cfg["env"]["numActions"]), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self._load_instance_link_bbx()
        self._create_task_buf()
        self._load_init_states()
        self._init_grasp_eval_state()
        self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(0.0, 0.2, 1.5), gymapi.Vec3(0.0, 0.0, 0.5))
    
    def _build_randomization_params(self):
        dr_params = deepcopy(self.cfg["task"]["randomization_params"])
        dr_params.setdefault("actor_params", {})

        hand_randomization = self.hand_cfg.get("randomization")
        if hand_randomization and hand_randomization.get("randomize", True):
            hand_actor_dr = deepcopy(hand_randomization)
            hand_actor_dr.pop("randomize", None)
            if hand_actor_dr:
                dr_params["actor_params"]["hand"] = hand_actor_dr

        object_randomization = self.object_cfg.get("randomization")
        if object_randomization and object_randomization.get("randomize", True):
            object_actor_dr = deepcopy(object_randomization)
            object_actor_dr.pop("randomize", None)
            if object_actor_dr:
                dr_params["actor_params"]["object"] = object_actor_dr

        return dr_params

    def _resolve_object_asset_dir(self) -> Path:
        asset_root = self.object_cfg["asset"]["asset_root"]
        object_dir = Path(self.asset_root) / asset_root
        object_dir = object_dir.resolve()
        if not object_dir.exists():
            raise FileNotFoundError(f"Object asset_root does not exist: {object_dir}")
        if not object_dir.is_dir():
            raise NotADirectoryError(f"Object asset_root must be a directory: {object_dir}")
        return object_dir

    def _get_object_cache_dir_name(self) -> str:
        return self._resolve_object_asset_dir().name

    def _get_configured_instance_id_list(self):
        raw_instance_ids = self.object_cfg["asset"].get("instance_id_list", [""])
        if raw_instance_ids is None:
            raw_instance_ids = [""]
        instance_ids = [str(instance_id) for instance_id in raw_instance_ids]
        if len(instance_ids) == 0 or all(instance_id.strip() == "" for instance_id in instance_ids):
            object_dir = self._resolve_object_asset_dir()
            discovered_ids = sorted(path.name for path in object_dir.iterdir() if path.is_dir())
            if not discovered_ids:
                raise ValueError(f"No instance directories found under object asset_root: {object_dir}")
            instance_ids = discovered_ids
        return self._filter_instance_ids_for_runtime_grasp_split(instance_ids)

    def _filter_instance_ids_for_runtime_grasp_split(self, instance_ids):
        runtime_grasp_split = normalize_grasp_split(self.cfg["env"].get("graspSplit", "train"))
        repo_root = Path(__file__).resolve().parents[2]
        cache_root = repo_root / "caches" / "initial_grasp" / self.hand_cfg["type"] / self._get_object_cache_dir_name()
        missing_instance_ids = []

        for instance_id in instance_ids:
            try:
                split_path = resolve_grasp_cache_path(cache_root / instance_id, runtime_grasp_split)
            except FileNotFoundError:
                missing_instance_ids.append(instance_id)
                continue

            grasp_states = np.load(split_path, allow_pickle=False)
            if grasp_states.ndim == 1:
                num_grasps = 1
            else:
                num_grasps = int(grasp_states.shape[0])
            if num_grasps <= 0:
                missing_instance_ids.append(instance_id)

        if missing_instance_ids:
            raise ValueError(
                f"graspSplit={runtime_grasp_split} requested, but the corresponding grasp cache is missing "
                f"or empty for instances: {missing_instance_ids}"
            )
        return instance_ids

    def _get_instance_sampling_probs(self, instance_id_list):
        if "sampleProb" in self.object_cfg["asset"]:
            raise ValueError(
                "object.asset.sampleProb is no longer supported. Remove it from the object config; "
                "instance sampling is always uniform over the resolved instance_id_list."
            )
        return np.full(len(instance_id_list), 1.0 / float(len(instance_id_list)), dtype=np.float64)

    def _load_instance_link_bbx(self):
        object_dir = self._resolve_object_asset_dir()
        lbx_path = object_dir / "lbx.json"
        if not lbx_path.exists():
            raise FileNotFoundError(f"Missing link bounding-box file: {lbx_path}")
        with open(lbx_path, "r", encoding="utf-8") as f:
            lbx_data = json.load(f)
        instance_link0_bbx = []
        instance_link1_bbx = []

        for instance_id in self.instance_id_list:
            if instance_id not in lbx_data:
                raise KeyError(f"{lbx_path} is missing instance_id '{instance_id}'")
            link_bbx = np.asarray(lbx_data[instance_id], dtype=np.float32)
            if link_bbx.shape != (6,):
                raise ValueError(
                    f"{lbx_path} entry for instance '{instance_id}' must have 6 values "
                    f"([link0_xyz, link1_xyz]), got shape {link_bbx.shape}"
                )
            instance_link0_bbx.append(link_bbx[:3].tolist())
            instance_link1_bbx.append(link_bbx[3:].tolist())

        self.instance_link0_bbx = to_torch(instance_link0_bbx, dtype=torch.float, device=self.device)
        self.instance_link1_bbx = to_torch(instance_link1_bbx, dtype=torch.float, device=self.device)

    def get_camera_frame(self, env_id=0):
        if not self.enable_camera:
            raise RuntimeError("Camera sensors are disabled. Set task.env.enableCameraSensors=True before creating the env.")
        env_index = int(env_id)
        if env_index < 0 or env_index >= len(self.envs):
            raise IndexError(f"Camera env index {env_index} out of range for {len(self.envs)} envs.")

        cam_handle = self.cam_handle_list[env_index]
        if torch.is_tensor(cam_handle):
            cam_handle = int(cam_handle.item())

        # In headless GPU runs, viewer.render() is never called, so we need to
        # explicitly fetch the latest simulation results before stepping graphics.
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        color_image = self.gym.get_camera_image(
            self.sim,
            self.envs[env_index],
            cam_handle,
            gymapi.IMAGE_COLOR,
        )
        frame = np.asarray(color_image, dtype=np.uint8).reshape(self.image_height, self.image_width, 4)[..., :3]
        return np.ascontiguousarray(frame)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        self._get_object_props()
        if self.randomize:
            self._apply_task_randomizations()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_asset(self):
        self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        hand_asset_file = self.hand_cfg['asset']
        # load hand asset
        self.hand_type = self.hand_cfg['type']
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = False
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            hand_asset_options.use_physx_armature = True
        hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.hand_asset = self.gym.load_asset(self.sim, self.asset_root, hand_asset_file, hand_asset_options)
        # load object asset
        raw_instance_ids = self._get_configured_instance_id_list()
        raw_prob = self._get_instance_sampling_probs(raw_instance_ids)
        self.object_type = self.object_cfg['asset']['type']
        instance_pairs = sorted(zip(raw_instance_ids, raw_prob.tolist()), key=lambda item: item[0])
        self.instance_id_list = [instance_id for instance_id, _ in instance_pairs]
        self.instance_id_prob = np.asarray([prob for _, prob in instance_pairs], dtype=np.float64)
        object_dir = self._resolve_object_asset_dir()
        self.object_cache_dir_name = object_dir.name
        self.asset_files_list = [
            os.path.relpath(object_dir / instance_id / self.object_cfg['asset']['asset_file'], self.asset_root)
            for instance_id in self.instance_id_list
        ]
        self.instance_index_list = range(len(self.instance_id_list))
        self.object_semantic_body_names = ["link_0", "link_1"]
        self.object_semantic_asset_body_indices = None
        object_shape_counts=[]
        object_body_counts=[]
        self.instance_asset_list = []
        for instance_index in self.instance_index_list:
            instance_asset_file = self.asset_files_list[instance_index]
            instance_asset_options = gymapi.AssetOptions()
            instance_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
            instance_asset_options.override_com = True
            instance_asset_options.override_inertia = True
            instance_asset_options.fix_base_link = False
            instance_asset_options.disable_gravity = False
            instance_asset_options.thickness = 0.01
            instance_asset_options.collapse_fixed_joints = False
            instance_asset_options.density=1000
            instance_asset = self.gym.load_asset(self.sim, self.asset_root, instance_asset_file, instance_asset_options)
            self.instance_asset_list.append(instance_asset)
            semantic_body_indices = [
                self.gym.find_asset_rigid_body_index(instance_asset, body_name)
                for body_name in self.object_semantic_body_names
            ]
            if any(body_idx < 0 for body_idx in semantic_body_indices):
                raise ValueError(
                    f"Object asset '{instance_asset_file}' must expose semantic rigid bodies "
                    f"{self.object_semantic_body_names}, got indices {semantic_body_indices}"
                )
            if self.object_semantic_asset_body_indices is None:
                self.object_semantic_asset_body_indices = semantic_body_indices
            elif self.object_semantic_asset_body_indices != semantic_body_indices:
                raise ValueError(
                    "All object assets must share the same rigid-body ordering for "
                    f"{self.object_semantic_body_names}. First asset indices "
                    f"{self.object_semantic_asset_body_indices}, current asset indices "
                    f"{semantic_body_indices} for '{instance_asset_file}'."
                )
            object_shape_counts.append(self.gym.get_asset_rigid_shape_count(instance_asset))
            object_body_counts.append(self.gym.get_asset_rigid_body_count(instance_asset))
        # get max agg
        object_shape_count = max(object_shape_counts)
        object_body_count = max(object_body_counts)
        hand_shape_count = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        hand_body_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.max_shape_count = object_shape_count + hand_shape_count +10
        self.max_body_count = object_body_count + hand_body_count +10

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self._create_asset()
        self.num_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        self.num_hand_actuators = self.num_hand_dofs

        self.actuated_dof_indices = [i for i in range(self.num_hand_dofs)]

        # set hand dof properties
        hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.hand_dof_lower_limits = []
        self.hand_dof_upper_limits = []
        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []

        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_hand_dofs):
            self.hand_dof_lower_limits.append(hand_dof_props['lower'][i])
            self.hand_dof_upper_limits.append(hand_dof_props['upper'][i])

            hand_dof_props['stiffness'][i] = self.hand_cfg['dof_props']['stiffness'][i]
            hand_dof_props['damping'][i] = self.hand_cfg['dof_props']['damping'][i]
            hand_dof_props['friction'][i] = self.hand_cfg['dof_props']['friction'][i]
            hand_dof_props['armature'][i] = self.hand_cfg['dof_props']['armature'][i]

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.hand_dof_lower_limits = to_torch(self.hand_dof_lower_limits, device=self.device)
        self.hand_dof_upper_limits = to_torch(self.hand_dof_upper_limits, device=self.device)

        self.hand_start_pose = gymapi.Transform()
        self.hand_start_pose.p = gymapi.Vec3(0, 0.0, 0.5)
        self.hand_start_pose.r = gymapi.Quat.from_axis_angle(
        gymapi.Vec3(0, 1, 0), -np.pi / 2) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi / 2)

        self.envs = []
        self.hand_indices = []
        self.object_indices = []
        self.object_handles = []
        self.object_body_names = []
        self.env2instance = []
        self.cam_handle_list = []

        hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.object_rb_handles = [
            hand_rb_count + body_idx for body_idx in self.object_semantic_asset_body_indices
        ]
        self.object_link0_rb_handle = self.object_rb_handles[0]
        self.object_link1_rb_handle = self.object_rb_handles[1]
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in self.hand_cfg['track_links']]
        self.force_handles = [self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in self.hand_cfg['force_links']]
        self.tactile_sensor_handles = self.force_handles[-5:]
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, self.max_body_count, self.max_shape_count, True)

            if self.enable_camera:
                # add camera
                camera_props = gymapi.CameraProperties()
                camera_props.width = self.image_width
                camera_props.height = self.image_height
                camera_props.horizontal_fov = self.horizontal_fov
                # camera_props.use_collision_geometry = self.use_collision_geometry
                cam_pos = gymapi.Vec3(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2])
                cam_target = gymapi.Vec3(self.cam_target[0], self.cam_target[1], self.cam_target[2])
                cam_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
                self.gym.set_camera_location(cam_handle, env_ptr, cam_pos, cam_target)
                self.cam_handle_list.append(cam_handle)
            # add hand
            hand_handle = self.gym.create_actor(env_ptr, self.hand_asset, self.hand_start_pose, "hand", i, 0)
            self.gym.set_actor_dof_properties(env_ptr, hand_handle, hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_handle, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # add object
            instance_index = np.random.choice(len(self.instance_index_list), p=self.instance_id_prob)
            instance_asset = self.instance_asset_list[instance_index]
            self.env2instance.append(instance_index)
            object_handle = self.gym.create_actor(env_ptr, instance_asset, gymapi.Transform(), 'object', i, 0)
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.object_handles.append(object_handle)
            object_body_names = self.gym.get_actor_rigid_body_names(env_ptr, object_handle)
            self.object_body_names.append(object_body_names)

            body_shape_counts = self.gym.get_actor_rigid_body_shape_indices(env_ptr, object_handle)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            for id, shape in enumerate(body_shape_counts):
                for t in range(shape.count):
                    if object_body_names[id] == "link_0":
                        props[shape.start + t].filter = (1 << 0) | (1 << 1)  # 对应 link_0
                    elif object_body_names[id] == "link_1":
                        props[shape.start + t].filter = (1 << 0) | (1 << 2)  # 对应 link_1
            self.gym.set_actor_rigid_shape_properties(env_ptr,object_handle,props)
            body_names = self.gym.get_actor_rigid_body_names(env_ptr, hand_handle)
            body_shape_counts = self.gym.get_actor_rigid_body_shape_indices(env_ptr, hand_handle)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_handle)
            special_names = [
                'left_thumb_elastomer', 
                'left_index_elastomer', 
                'left_middle_elastomer', 
                'left_ring_elastomer', 
                'left_pinky_elastomer'
            ]
            actual_n = 0
            for id, shape in enumerate(body_shape_counts):
                if shape.count > 0:
                    is_special = any(name in body_names[id] for name in special_names)
                    if not is_special:
                        actual_n += 1
            N = sum(1 << (i + 8) for i in range(1, actual_n + 1)) if actual_n > 0 else 0
            n_counter = 1 
            for id, shape in enumerate(body_shape_counts):
                if shape.count == 0:
                    continue
                name = body_names[id]
                if 'left_thumb_elastomer' in name:
                    current_mask = (1 << 3) | (1 << 4) | N             # 连兄弟，加身份，连 n
                elif 'left_index_elastomer' in name:
                    current_mask = (1 << 3) | (1 << 5) | N
                elif 'left_middle_elastomer' in name:
                    current_mask = (1 << 3) | (1 << 6) | N
                elif 'left_ring_elastomer' in name:
                    current_mask = (1 << 3) | (1 << 7) | N
                elif 'left_pinky_elastomer' in name:
                    current_mask = (1 << 3) | (1 << 8) | N
                else:
                    current_mask = 1 << (n_counter + 8)
                    n_counter += 1  
                for t in range(shape.count):
                    props[shape.start + t].filter = current_mask
            self.gym.set_actor_rigid_shape_properties(env_ptr,hand_handle,props)
            #setup default value
            rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            rs_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            dof_props = self.gym.get_actor_dof_properties(env_ptr, object_handle)
            object_body_name_to_idx = {name: idx for idx, name in enumerate(object_body_names)}
            rb_props[object_body_name_to_idx["link_0"]].mass = self.object_cfg['default_props']['mass'][0]
            rb_props[object_body_name_to_idx["link_1"]].mass = self.object_cfg['default_props']['mass'][1]
            for body_name in self.object_cfg['asset']['disgravity_part']:
                disg_id = self.gym.find_actor_rigid_body_index(env_ptr, object_handle, body_name, gymapi.IndexDomain.DOMAIN_ACTOR)
                rb_props[disg_id].flags = 1
            body_shape_counts = self.gym.get_actor_rigid_body_shape_indices(env_ptr, object_handle)
            for id,shape in enumerate(body_shape_counts):
                for t in range(shape.count):
                    rs_props[shape.start + t].friction = self.object_cfg['default_props']['friction']
            dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
            if self.object_is_lighter:
                dof_props['driveMode'][:] = gymapi.DOF_MODE_EFFORT
            dof_props['friction'] = 0.001
            dof_props['stiffness'][:] = self.object_cfg['default_props'].get('dof_stiffness', 0.0)
            dof_props['damping'][:] = self.object_cfg['default_props']['dof_damping']
            dof_props['armature'] = 0.001
            self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, rb_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, rs_props)
            self.gym.set_actor_dof_properties(env_ptr, object_handle, dof_props)

            self.object_dof_lower_limits.append(dof_props['lower'][0])
            self.object_dof_upper_limits.append(dof_props['upper'][0])
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [
            object_rb_props[body_idx].mass for body_idx in self.object_semantic_asset_body_indices
        ]

        self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
        self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.force_handles = to_torch(self.force_handles, dtype=torch.long, device=self.device)
        self.tactile_sensor_handles = to_torch(self.tactile_sensor_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.env2instance = to_torch(self.env2instance, dtype=torch.long, device=self.device)
        self.cam_handle_list = to_torch(self.cam_handle_list, dtype=torch.long, device=self.device)


    def _get_object_body_dr_cfg(self):
        actor_params = self.randomization_params.get("actor_params", {})
        object_params = actor_params.get("object", {})
        return object_params.get("rigid_body_properties")

    def _get_generic_randomization_params(self):
        dr_params = deepcopy(self.randomization_params)
        actor_params = dr_params.get("actor_params", {})
        object_params = actor_params.get("object", {})
        object_params.pop("rigid_body_properties", None)
        if not object_params and "object" in actor_params:
            actor_params.pop("object")
        return dr_params

    def _has_generic_randomizations(self, dr_params):
        return any(
            key in dr_params and dr_params[key]
            for key in ("observations", "actions", "sim_params", "actor_params")
        )

    def _select_object_body_dr_env_ids(self, requested_env_ids=None):
        if self._object_body_dr_first:
            return list(range(self.num_envs))

        if requested_env_ids is None:
            return []

        if not torch.is_tensor(requested_env_ids):
            requested_env_ids = torch.as_tensor(requested_env_ids, device=self.device, dtype=torch.long)
        else:
            requested_env_ids = requested_env_ids.to(device=self.device, dtype=torch.long)

        rand_freq = self.randomization_params.get("frequency", 1)
        mask = self.randomize_buf[requested_env_ids] >= rand_freq
        return requested_env_ids[mask].tolist()

    def _apply_object_rigid_body_randomizations(self, env_ids):
        body_dr_cfg = self._get_object_body_dr_cfg()
        if not body_dr_cfg or not env_ids:
            return

        global_cfg = body_dr_cfg.get("__all__", {})
        by_body_cfg = body_dr_cfg.get("by_body", {})
        if not global_cfg and not by_body_cfg:
            # Backward-compatible path: apply the same rigid-body randomization to every body.
            global_cfg = body_dr_cfg

        curr_step = max(self.gym.get_frame_count(self.sim), 0)

        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            object_handle = self.object_handles[env_id]
            object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)

            if env_id not in self._object_rb_original_props:
                self._object_rb_original_props[env_id] = [
                    {attr: getattr(prop, attr) for attr in dir(prop)}
                    for prop in object_rb_props
                ]

            set_random_properties = False
            for body_idx, (prop, og_prop, body_name) in enumerate(
                zip(object_rb_props, self._object_rb_original_props[env_id], self.object_body_names[env_id])
            ):
                body_key = str(body_idx)
                merged_cfg = dict(global_cfg)
                if body_name in by_body_cfg:
                    merged_cfg.update(by_body_cfg[body_name])
                if body_key in by_body_cfg:
                    merged_cfg.update(by_body_cfg[body_key])

                for attr, attr_randomization_params in merged_cfg.items():
                    setup_only = attr_randomization_params.get("setup_only", False)
                    if (setup_only and not self.sim_initialized) or not setup_only:
                        apply_random_samples(prop, og_prop, attr, attr_randomization_params, curr_step)
                        set_random_properties = True

            if set_random_properties:
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, object_rb_props, True)

        self._object_body_dr_first = False

    def _apply_task_randomizations(self, env_ids=None):
        object_body_env_ids = self._select_object_body_dr_env_ids(env_ids)
        generic_dr_params = self._get_generic_randomization_params()

        if self._has_generic_randomizations(generic_dr_params):
            self.apply_randomizations(generic_dr_params)

        self._apply_object_rigid_body_randomizations(object_body_env_ids)
        self._get_object_props()


    def _get_object_props(self):
        num_object_bodies = len(self.object_semantic_body_names)
        self.object_mass = torch.zeros(
            (self.num_envs, num_object_bodies),
            dtype=torch.float,
            device=self.device,
        )
        self.object_friction = torch.zeros(
            (self.num_envs, 1),
            dtype=torch.float,
            device=self.device,
        )
        self.object_dof_damping = torch.zeros(
            (self.num_envs, 1),
            dtype=torch.float,
            device=self.device,
        )
        self.object_dof_stiffness = torch.zeros(
            (self.num_envs, 1),
            dtype=torch.float,
            device=self.device,
        )
        default_mass = torch.as_tensor(
            self.object_cfg['default_props']['mass'],
            dtype=torch.float,
            device=self.device,
        )
        default_friction = float(self.object_cfg['default_props']['friction'])
        default_damping = float(self.object_cfg['default_props']['dof_damping'])
        default_stiffness = float(self.object_cfg['default_props'].get('dof_stiffness', 0.0))
        for env_id in range(self.num_envs):
            if not self.randomize:
                masses = default_mass
                friction = torch.tensor(default_friction, dtype=torch.float, device=self.device)
                dof_stiffness = torch.tensor(default_stiffness, dtype=torch.float, device=self.device)
                dof_damping = torch.tensor(default_damping, dtype=torch.float, device=self.device)
            else:
                rb_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.object_handles[env_id])
                rs_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.object_handles[env_id])
                dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], self.object_handles[env_id])
                body_name_to_idx = {
                    name: idx for idx, name in enumerate(self.object_body_names[env_id])
                }
                masses = torch.tensor(
                    [rb_props[body_name_to_idx[body_name]].mass for body_name in self.object_semantic_body_names],
                    dtype=torch.float,
                    device=self.device,
                )

                semantic_shape_frictions = []
                body_shape_counts = self.gym.get_actor_rigid_body_shape_indices(self.envs[env_id], self.object_handles[env_id])
                for body_idx, shape in enumerate(body_shape_counts):
                    if self.object_body_names[env_id][body_idx] not in self.object_semantic_body_names:
                        continue
                    for shape_offset in range(shape.count):
                        semantic_shape_frictions.append(rs_props[shape.start + shape_offset].friction)
                if semantic_shape_frictions:
                    friction = torch.tensor(
                        float(sum(semantic_shape_frictions) / len(semantic_shape_frictions)),
                        dtype=torch.float,
                        device=self.device,
                    )
                else:
                    friction = torch.tensor(default_friction, dtype=torch.float, device=self.device)

                dof_stiffness = torch.as_tensor(dof_props['stiffness'], dtype=torch.float, device=self.device).reshape(-1)[0]
                dof_damping = torch.as_tensor(dof_props['damping'], dtype=torch.float, device=self.device).reshape(-1)[0]
            self.object_dof_stiffness[env_id, 0] = dof_stiffness
            self.object_dof_damping[env_id, 0] = dof_damping
            self.object_friction[env_id, 0] = friction
            self.object_mass[env_id, :masses.numel()] = masses


    def _create_task_buf(self):
        self.goal_obj_dof_pos = self.obj_dof_pos.clone()
        self.init_obj_dof_pos = self.obj_dof_pos.clone()
        self.init_hand_dof_pos = self.hand_dof_pos.clone()
        self.init_fingertip_pos = torch.zeros((self.num_envs,15), dtype=torch.float, device=self.device)
        self.init_contact_info = torch.zeros((self.num_envs,len(self.force_handles)), dtype=torch.float, device=self.device)
        self.init_object_pos = torch.zeros((self.num_envs,3), dtype=torch.float, device=self.device)
        self.init_object_rot = torch.zeros((self.num_envs,4), dtype=torch.float, device=self.device)
        self.prev_object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.prev_object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.link1_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.init_link1_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.init_link0_bbx = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.init_link1_bbx = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.binary_tactile_threshold = float(self.cfg["env"].get("binaryTactileThreshold", 0.001))
        self.proprio_obs_dim = self.num_hand_dofs + self.cfg["env"]["numActions"]
        if self.proprio_obs_dim != self.student_temporal_obs_layout["proprio_dim_per_step"]:
            raise ValueError(
                "Computed proprio observation dim does not match the configured student temporal layout: "
                f"runtime={self.proprio_obs_dim}, expected={self.student_temporal_obs_layout['proprio_dim_per_step']}"
            )
        self.proprioception_buf = torch.zeros(
            (self.num_envs, self.proprio_history_len, self.proprio_obs_dim),
            dtype=torch.float,
            device=self.device,
        )
        self.mini_goal_distance =  torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        raw_goals = self.object_cfg['task'].get('goals')
        if raw_goals is None:
            raise ValueError("object.task.goals is required.")
        self.goal_offsets = to_torch(raw_goals, dtype=torch.float, device=self.device).flatten()
        if self.goal_offsets.numel() < 2:
            raise ValueError("object.task.goals must contain at least two goals.")
        self.init_dof = self.object_cfg['asset']['initDof']
        self.success_hold_duration = float(self.cfg["env"].get("successHoldDurationSec", 1.0))
        success_hold_duration_range = self.cfg["env"].get("successHoldDurationRangeSec", None)
        if success_hold_duration_range is None:
            self.success_hold_duration_min = self.success_hold_duration
            self.success_hold_duration_max = self.success_hold_duration
        else:
            if len(success_hold_duration_range) != 2:
                raise ValueError("env.successHoldDurationRangeSec must have exactly two values: [min, max].")
            self.success_hold_duration_min = float(success_hold_duration_range[0])
            self.success_hold_duration_max = float(success_hold_duration_range[1])
            if self.success_hold_duration_min < 0.0 or self.success_hold_duration_max < 0.0:
                raise ValueError("env.successHoldDurationRangeSec values must be non-negative.")
            if self.success_hold_duration_max < self.success_hold_duration_min:
                raise ValueError("env.successHoldDurationRangeSec must satisfy max >= min.")
        self.max_consecutive_successes = int(self.cfg["env"].get("maxConsecutiveSuccesses", 2))
        self.goal_switch_timeout_sec = float(self.cfg["env"].get("goalSwitchTimeoutSec", 5.0))
        self.success_hold_steps_dt = self.dt * self.control_freq_inv
        self.goal_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.goal_refresh_interval = self._sample_goal_refresh_intervals(self.num_envs)
        self.goal_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.goal_achieved_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.goal_achieved_latch = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.training_success_hold_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.training_goal_success_counted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_goal_distance_reward1_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # self.at_reset_ids = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.runtime_grasp_split = normalize_grasp_split(self.cfg["env"].get("graspSplit", "train"))
        self.runtime_grasp_fixed_state = None
        self.policy_update_step = 0
        self._update_reward_scales_current()
        self.debug_reset_cause_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.debug_reset_cause_success_budget = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.debug_reset_cause_fall = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.debug_reset_cause_invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _normalize_env_ids(self, env_ids=None):
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if not torch.is_tensor(env_ids):
            return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        return env_ids.to(device=self.device, dtype=torch.long)

    def _get_schedule_progress(self, global_step, *, warmup_steps, total_steps, device):
        if total_steps <= warmup_steps:
            progress_value = 0.0 if global_step < warmup_steps else 1.0
        else:
            progress_value = (float(global_step) - float(warmup_steps)) / float(
                total_steps - warmup_steps
            )
        return torch.clamp(torch.tensor(progress_value, dtype=torch.float, device=device), 0.0, 1.0)

    def _get_reward_curriculum_progress(self, global_step, *, device):
        return self._get_schedule_progress(
            global_step,
            warmup_steps=self.reward_curriculum_warmup_steps,
            total_steps=self.reward_curriculum_total_steps,
            device=device,
        )

    def _update_reward_scales_current(self):
        self.reward_scales_current = dict(self.reward_scales)
        if not self.reward_weight_curriculum:
            return

        progress = float(
            self._get_reward_curriculum_progress(
                self.policy_update_step,
                device=self.device,
            ).item()
        )
        for reward_name, start_weight in self.reward_weight_curriculum.items():
            final_weight = self.reward_scales[reward_name]
            self.reward_scales_current[reward_name] = (
                (1.0 - progress) * start_weight + progress * final_weight
            )

    def _get_reward_curriculum_metrics(self):
        if not self.reward_weight_curriculum:
            return {}

        metrics = {
            "reward_curriculum/progress": self._get_reward_curriculum_progress(
                self.policy_update_step,
                device=self.device,
            )
        }
        for reward_name in self.reward_weight_curriculum:
            metrics[f"reward_curriculum/{reward_name}_weight"] = float(
                self.reward_scales_current[reward_name]
            )
        return metrics

    def set_train_info(self, env_frames, *args, **kwargs):
        super().set_train_info(env_frames, *args, **kwargs)
        update_step = None
        if args:
            algo = args[0]
            if hasattr(algo, "epoch_num"):
                update_step = int(algo.epoch_num)
        if "epoch_num" in kwargs:
            update_step = int(kwargs["epoch_num"])
        elif "update_num" in kwargs:
            update_step = int(kwargs["update_num"])
        if update_step is None:
            update_step = int(env_frames)
        self.policy_update_step = update_step
        self._update_reward_scales_current()

    def get_env_state(self):
        return {}

    def set_env_state(self, env_state):
        self._update_reward_scales_current()

    def _init_grasp_eval_state(self):
        self.eval_mode = False
        self.eval_consecutive_mode = False
        self.eval_instance_index = None
        self.eval_grasp_states = None
        self.eval_grasp_split = "test"
        self.eval_episodes_per_grasp = 0
        self.eval_goal_sequence = None
        self.eval_active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.eval_episode_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.eval_episode_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.eval_goal_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.eval_goal_timeout = 0.0
        self.eval_open_stage_finalized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.eval_close_stage_finalized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.eval_open_final_goal_distance_episode = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.eval_close_final_goal_distance_episode = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.eval_open_success_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.eval_close_success_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.eval_consecutive_success_cycles = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.eval_completion_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.eval_completion_stage = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.eval_completion_goal_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.eval_parallel_trials_per_grasp = 1
        self.eval_base_num_grasps = self.num_envs

    def _reset_eval_episode_state(self, env_ids):
        if len(env_ids) == 0:
            return
        self.eval_open_stage_finalized[env_ids] = False
        self.eval_close_stage_finalized[env_ids] = False
        self.eval_open_final_goal_distance_episode[env_ids] = 0.0
        self.eval_close_final_goal_distance_episode[env_ids] = 0.0
        self.eval_open_success_episode[env_ids] = False
        self.eval_close_success_episode[env_ids] = False

    def _resolve_eval_goal_sequence(self, goal_sequence, mode_name):
        if goal_sequence is None:
            goal_sequence = self.object_cfg['task'].get('goals')
        if goal_sequence is None:
            raise ValueError("object.task.goals is required.")
        if len(goal_sequence) != 2:
            raise ValueError(f"{mode_name} expects exactly two goal values: open then close.")
        return goal_sequence

    def _get_eval_grasp_states_for_split(self, instance_index, grasp_split):
        self._ensure_grasp_split_loaded(instance_index, grasp_split)
        if grasp_split == "selected":
            grasp_states = self.selected_valid_states[instance_index]
        elif grasp_split == "success":
            grasp_states = self.success_valid_states[instance_index]
        elif grasp_split == "train":
            grasp_states = self.train_valid_states[instance_index]
        elif grasp_split == "test":
            grasp_states = self.test_valid_states[instance_index]
        else:
            grasp_states = self.valid_states[instance_index]
        return grasp_states

    def _reset_eval_session_buffers(self):
        self.eval_active_mask[:] = True
        self.eval_episode_started.zero_()
        self.eval_episode_counts.zero_()
        self.eval_open_stage_finalized.zero_()
        self.eval_close_stage_finalized.zero_()
        self.eval_open_final_goal_distance_episode.zero_()
        self.eval_close_final_goal_distance_episode.zero_()
        self.eval_open_success_episode.zero_()
        self.eval_close_success_episode.zero_()
        self.eval_consecutive_success_cycles.zero_()
        self.eval_completion_reason.zero_()
        self.eval_completion_stage.fill_(-1)
        self.eval_completion_goal_distance.zero_()
        self.eval_goal_stage.zero_()
        self.goal_achieved_latch.zero_()
        self.goal_timer.zero_()
        self.goal_refresh_interval.fill_(float("inf"))

    def _activate_eval_session(
        self,
        *,
        instance_index,
        grasp_states,
        grasp_split,
        episodes_per_grasp,
        goal_sequence,
        stage_duration,
        parallel_trials_per_grasp=1,
        base_num_grasps=None,
    ):
        self.eval_mode = True
        self.eval_consecutive_mode = True
        self.eval_instance_index = instance_index
        self.eval_grasp_states = grasp_states
        self.eval_grasp_split = str(grasp_split)
        self.eval_episodes_per_grasp = int(episodes_per_grasp)
        self.eval_parallel_trials_per_grasp = int(parallel_trials_per_grasp)
        self.eval_base_num_grasps = int(grasp_states.shape[0] if base_num_grasps is None else base_num_grasps)
        self.eval_goal_sequence = to_torch(goal_sequence, dtype=torch.float, device=self.device).view(-1, 1)
        self.eval_goal_timeout = float(self.goal_switch_timeout_sec if stage_duration is None else stage_duration)
        if self.eval_goal_timeout <= 0.0:
            raise ValueError(f"Eval goal timeout must be positive, got {self.eval_goal_timeout}")
        self._reset_eval_session_buffers()

    def _record_eval_completion(self, env_ids, reason_code, stage=None, goal_distance=None):
        if len(env_ids) == 0:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        pending_mask = self.eval_completion_reason[env_ids] == 0
        if not torch.any(pending_mask):
            return
        pending_env_ids = env_ids[pending_mask]

        self.eval_completion_reason[pending_env_ids] = int(reason_code)

        if stage is None:
            stage_values = self.eval_goal_stage[pending_env_ids]
        elif isinstance(stage, int):
            stage_values = torch.full((len(pending_env_ids),), stage, device=self.device, dtype=torch.long)
        elif torch.is_tensor(stage):
            stage_values = stage.to(device=self.device, dtype=torch.long)
            if stage_values.numel() != len(env_ids):
                raise ValueError("stage tensor must match env_ids length when recording eval completion.")
            stage_values = stage_values[pending_mask]
        else:
            stage_values = torch.as_tensor(stage, device=self.device, dtype=torch.long)
            if stage_values.numel() != len(env_ids):
                raise ValueError("stage values must match env_ids length when recording eval completion.")
            stage_values = stage_values[pending_mask]
        self.eval_completion_stage[pending_env_ids] = stage_values

        if goal_distance is None:
            goal_distance_values = self._get_eval_stage_goal_distance(pending_env_ids, stage_values)
        elif torch.is_tensor(goal_distance):
            goal_distance_values = goal_distance.to(device=self.device, dtype=torch.float)
            if goal_distance_values.numel() != len(env_ids):
                raise ValueError("goal_distance tensor must match env_ids length when recording eval completion.")
            goal_distance_values = goal_distance_values[pending_mask]
        else:
            goal_distance_values = torch.as_tensor(goal_distance, device=self.device, dtype=torch.float)
            if goal_distance_values.numel() != len(env_ids):
                raise ValueError("goal_distance values must match env_ids length when recording eval completion.")
            goal_distance_values = goal_distance_values[pending_mask]
        self.eval_completion_goal_distance[pending_env_ids] = goal_distance_values

    def configure_grasp_consecutive_evaluation(
        self,
        instance_id,
        goal_sequence=None,
        stage_duration=None,
        grasp_split="test",
        episodes_per_grasp=1,
    ):
        if instance_id not in self.instance_id_list:
            raise ValueError(f"Unknown instance_id '{instance_id}'. Available: {self.instance_id_list}")
        goal_sequence = self._resolve_eval_goal_sequence(goal_sequence, "This consecutive evaluation mode")
        grasp_split = normalize_grasp_split(grasp_split)
        instance_index = self.instance_id_list.index(instance_id)
        if not torch.all(self.env2instance == instance_index):
            raise ValueError(
                "All envs must use the same object instance for consecutive grasp evaluation. "
                "Set object.asset.instance_id_list to only the target instance before creating the env."
            )

        grasp_states = self._get_eval_grasp_states_for_split(instance_index, grasp_split)
        base_num_grasps = int(grasp_states.shape[0])
        episodes_per_grasp = int(episodes_per_grasp)
        if episodes_per_grasp <= 0:
            raise ValueError(f"episodes_per_grasp must be positive, got {episodes_per_grasp}")

        if self.num_envs == base_num_grasps:
            expanded_grasp_states = grasp_states
            eval_episodes_per_env = episodes_per_grasp
            parallel_trials_per_grasp = 1
        elif self.num_envs == base_num_grasps * episodes_per_grasp:
            expanded_grasp_states = grasp_states.repeat(episodes_per_grasp, 1)
            eval_episodes_per_env = 1
            parallel_trials_per_grasp = episodes_per_grasp
        else:
            raise ValueError(
                f"num_envs ({self.num_envs}) must equal either the number of {grasp_split} grasps "
                f"for instance '{instance_id}' ({base_num_grasps}) or "
                f"episodes_per_grasp * num_grasps ({episodes_per_grasp} * {base_num_grasps} = "
                f"{episodes_per_grasp * base_num_grasps})."
            )

        self._activate_eval_session(
            instance_index=instance_index,
            grasp_states=expanded_grasp_states,
            grasp_split=grasp_split,
            episodes_per_grasp=eval_episodes_per_env,
            goal_sequence=goal_sequence,
            stage_duration=stage_duration,
            parallel_trials_per_grasp=parallel_trials_per_grasp,
            base_num_grasps=base_num_grasps,
        )

    def configure_fixed_grasp_consecutive_evaluation(
        self,
        instance_id,
        grasp_state,
        goal_sequence=None,
        stage_duration=None,
        episodes_per_grasp=1,
    ):
        if instance_id not in self.instance_id_list:
            raise ValueError(f"Unknown instance_id '{instance_id}'. Available: {self.instance_id_list}")
        goal_sequence = self._resolve_eval_goal_sequence(goal_sequence, "This consecutive evaluation mode")
        episodes_per_grasp = int(episodes_per_grasp)
        if episodes_per_grasp <= 0:
            raise ValueError(f"episodes_per_grasp must be positive, got {episodes_per_grasp}")

        instance_index = self.instance_id_list.index(instance_id)
        if not torch.all(self.env2instance == instance_index):
            raise ValueError(
                "All envs must use the same object instance for fixed grasp consecutive evaluation. "
                "Set object.asset.instance_id_list to only the target instance before creating the env."
            )

        fixed_grasp_state = to_torch(grasp_state, dtype=torch.float, device=self.device).view(1, -1)
        if fixed_grasp_state.shape[1] != self.grasp_state_dim:
            raise ValueError(
                f"Fixed grasp state width {fixed_grasp_state.shape[1]} does not match runtime grasp_state_dim "
                f"{self.grasp_state_dim}."
            )

        self._activate_eval_session(
            instance_index=instance_index,
            grasp_states=fixed_grasp_state.expand(self.num_envs, -1).clone(),
            grasp_split="fixed",
            episodes_per_grasp=1 if self.num_envs == episodes_per_grasp else episodes_per_grasp,
            goal_sequence=goal_sequence,
            stage_duration=stage_duration,
            parallel_trials_per_grasp=episodes_per_grasp if self.num_envs == episodes_per_grasp else 1,
            base_num_grasps=1,
        )

    def set_runtime_grasp_split(self, grasp_split="train"):
        self.runtime_grasp_split = normalize_grasp_split(grasp_split)
        self.runtime_grasp_fixed_state = None
        for instance_index in range(len(self.instance_id_list)):
            self._ensure_grasp_split_loaded(instance_index, self.runtime_grasp_split)
        self._refresh_grasp_split_tensors()

    def set_runtime_grasp_selection(self, instance_id, grasp_split="selected"):
        grasp_split = normalize_grasp_split(grasp_split)
        if grasp_split != "selected":
            raise ValueError(
                f"Inference runtime grasp selection only supports the selected pool, got '{grasp_split}'."
            )
        if instance_id not in self.instance_id_list:
            raise ValueError(f"Unknown instance_id '{instance_id}'. Available: {self.instance_id_list}")
        instance_index = self.instance_id_list.index(instance_id)
        if not torch.all(self.env2instance == instance_index):
            raise ValueError(
                "All inference envs must use the same object instance for fixed selected-grasp inference. "
                "Set object.asset.instance_id_list to only the target instance before creating the env."
            )
        self._ensure_grasp_split_loaded(instance_index, "selected")
        grasp_states = self.selected_valid_states[instance_index]
        if grasp_states.shape[0] == 0:
            raise ValueError(f"No selected grasp states available for instance '{instance_id}'.")
        if grasp_states.shape[0] != 1:
            raise ValueError(
                f"selected grasps for instance '{instance_id}' must contain exactly one row for inference, "
                f"got {grasp_states.shape[0]}."
            )
        self.runtime_grasp_split = "selected"
        self.runtime_grasp_fixed_state = grasp_states[0].clone()

    def _set_eval_goal_targets(self, env_ids, stage):
        if len(env_ids) == 0:
            return

        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        if isinstance(stage, int):
            stage = torch.full((len(env_ids),), stage, device=self.device, dtype=torch.long)
        elif not torch.is_tensor(stage):
            stage = torch.as_tensor(stage, device=self.device, dtype=torch.long)
        else:
            stage = stage.to(device=self.device, dtype=torch.long)

        self.eval_goal_stage[env_ids] = stage
        self.goal_achieved_latch[env_ids] = False
        self.training_success_hold_time[env_ids] = 0.0
        goal_targets = self.eval_goal_sequence[stage].expand(-1, self.goal_obj_dof_pos.shape[1])
        self.goal_obj_dof_pos[env_ids] = self.init_obj_dof_pos[env_ids] + goal_targets
        self.mini_goal_distance[env_ids] = torch.norm(
            self.obj_dof_pos[env_ids] - self.goal_obj_dof_pos[env_ids], p=1, dim=-1
        )

    def _get_eval_stage_goal_distance(self, env_ids, stage):
        if len(env_ids) == 0:
            return torch.zeros(0, dtype=torch.float, device=self.device)
        goal_offset = self.eval_goal_sequence[stage].expand(len(env_ids), self.goal_obj_dof_pos.shape[1])
        stage_goal_obj_dof_pos = self.init_obj_dof_pos[env_ids] + goal_offset
        goal_distance = torch.norm(self.obj_dof_pos[env_ids] - stage_goal_obj_dof_pos, p=1, dim=-1)
        return torch.nan_to_num(goal_distance, nan=1.0).clamp(0.0, 1.0)

    def _finalize_eval_stage(self, env_ids, stage, goal_distance=None, success_override=None):
        if len(env_ids) == 0:
            return
        if goal_distance is None:
            goal_distance = self._get_eval_stage_goal_distance(env_ids, stage)
        if success_override is None:
            success = goal_distance < self.object_cfg['task']['success_threshold']
        else:
            success = success_override.to(dtype=torch.bool, device=self.device)
        if stage == 0:
            pending_env_ids = env_ids[~self.eval_open_stage_finalized[env_ids]]
            if len(pending_env_ids) == 0:
                return
            pending_goal_distance = goal_distance[~self.eval_open_stage_finalized[env_ids]]
            pending_success = success[~self.eval_open_stage_finalized[env_ids]]
            self.eval_open_stage_finalized[pending_env_ids] = True
            self.eval_open_final_goal_distance_episode[pending_env_ids] = pending_goal_distance
            self.eval_open_success_episode[pending_env_ids] = pending_success
            return

        pending_env_ids = env_ids[~self.eval_close_stage_finalized[env_ids]]
        if len(pending_env_ids) == 0:
            return
        pending_goal_distance = goal_distance[~self.eval_close_stage_finalized[env_ids]]
        pending_success = success[~self.eval_close_stage_finalized[env_ids]]
        self.eval_close_stage_finalized[pending_env_ids] = True
        self.eval_close_final_goal_distance_episode[pending_env_ids] = pending_goal_distance
        self.eval_close_success_episode[pending_env_ids] = pending_success

    def is_grasp_evaluation_complete(self):
        return self.eval_mode and bool(torch.all(self.eval_episode_counts >= self.eval_episodes_per_grasp).item())

    def _get_eval_instance_id(self):
        if self.eval_instance_index is not None:
            return self.instance_id_list[self.eval_instance_index]
        if self.env2instance.numel() == 0:
            raise RuntimeError("Could not resolve eval instance id because env2instance is empty.")
        unique_instance_indices = torch.unique(self.env2instance).detach().cpu().tolist()
        if len(unique_instance_indices) != 1:
            raise RuntimeError(
                "Could not resolve eval instance id because multiple instance indices are active "
                f"during inference/evaluation: {unique_instance_indices}"
            )
        return self.instance_id_list[int(unique_instance_indices[0])]

    def get_grasp_consecutive_evaluation_stats(self):
        if not self.eval_mode or not self.eval_consecutive_mode:
            raise RuntimeError("Consecutive grasp evaluation has not been configured.")

        reason_map = {
            0: "unfinished",
            1: "goal_timeout",
            2: "fall",
            3: "invalid",
            5: "reset_without_reason",
            6: "episode_timeout",
        }
        stage_map = {
            -1: "none",
            0: "open",
            1: "close",
        }

        cycles = self.eval_consecutive_success_cycles.detach().cpu()
        reason_codes = self.eval_completion_reason.detach().cpu()
        stage_codes = self.eval_completion_stage.detach().cpu()
        goal_distance = self.eval_completion_goal_distance.detach().cpu()
        sorted_indices = torch.argsort(cycles, descending=True).tolist()
        best_grasp_index = int(sorted_indices[0]) if len(sorted_indices) > 0 else -1
        best_cycles = int(cycles.max().item()) if cycles.numel() > 0 else 0

        return {
            "instance_id": self._get_eval_instance_id(),
            "grasp_split": self.eval_grasp_split,
            "goal_sequence": self.eval_goal_sequence.view(-1).detach().cpu().tolist(),
            "stage_duration": float(self.eval_goal_timeout),
            "episode_length_steps": int(self.max_episode_length),
            "num_grasps": int(self.num_envs),
            "base_num_grasps": int(self.eval_base_num_grasps),
            "parallel_trials_per_grasp": int(self.eval_parallel_trials_per_grasp),
            "consecutive_success_cycles": cycles.tolist(),
            "completion_reason_code": reason_codes.tolist(),
            "completion_reason": [reason_map.get(int(code), "unknown") for code in reason_codes.tolist()],
            "completion_stage_code": stage_codes.tolist(),
            "completion_stage": [stage_map.get(int(code), "unknown") for code in stage_codes.tolist()],
            "completion_goal_distance": goal_distance.tolist(),
            "best_grasp_index": best_grasp_index,
            "best_consecutive_success_cycles": best_cycles,
            "sorted_grasp_indices_by_cycles": [int(idx) for idx in sorted_indices],
        }

    def _get_dones(self):
        # fall = ((self.object_pos[:,2] - self.init_object_pos[:,2]).abs().ravel() > 0.1) | \
        pos_diff = torch.norm(self.object_pos-self.init_object_pos,p=2, dim=-1)
        quat_diff = quat_mul(self.object_rot, quat_conjugate(self.init_object_rot))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
        fall = (pos_diff > self.object_cfg['task']['pos_devia_threshold']) | (rot_dist > self.object_cfg['task']['rot_devia_threshold'])
        invalid = self._get_nonfinite_env_mask()
        self.debug_reset_cause_fall[:] = fall
        self.debug_reset_cause_invalid[:] = invalid
        self.debug_reset_cause_timeout.zero_()
        self.debug_reset_cause_success_budget.zero_()
        if self.eval_mode:
            if self.eval_consecutive_mode:
                timeout = self.progress_buf >= self.max_episode_length
                self.debug_reset_cause_timeout[:] = timeout
                self.truncated_envs = fall | invalid | timeout
                self.reset_buf[:] = fall | invalid | timeout
            else:
                self.truncated_envs = fall | invalid
                self.reset_buf[:] = fall | invalid
            return
        timeout = self.progress_buf >= self.max_episode_length
        self.debug_reset_cause_timeout[:] = timeout
        self.truncated_envs = fall | invalid
        self.reset_buf[:] = fall | invalid | timeout

    def _update_obj_dof_velocity(self):
        control_dt = self.dt * self.control_freq_inv
        if control_dt <= 0.0:
            self.obj_dof_vel.zero_()
            self.prev_obj_dof_pos.copy_(self.obj_dof_pos)
            return
        self.obj_dof_vel[:] = (self.obj_dof_pos - self.prev_obj_dof_pos) / control_dt
        self.prev_obj_dof_pos.copy_(self.obj_dof_pos)

    def compute_reward(self, actions):
        if self.eval_mode:
            self.reset_buf[~self.eval_active_mask] = 0
        self.goal_achieved_step.zero_()
        goal_distance = torch.norm(self.obj_dof_pos - self.goal_obj_dof_pos, p=1, dim=-1)
        goal_distance = torch.nan_to_num(goal_distance, nan=1.0).clamp(0.0, 1.0)
        success_envs = (goal_distance < self.object_cfg['task']['success_threshold'])
        counted_success_envs = torch.zeros_like(success_envs)
        if self.eval_mode:
            success_envs = torch.logical_and(success_envs, self.eval_active_mask)
        entered_success_envs = torch.logical_and(success_envs, ~self.goal_achieved_latch)
        self.goal_achieved_latch[:] = success_envs
        self.training_success_hold_time[~success_envs] = 0.0
        self.training_success_hold_time[success_envs] += self.success_hold_steps_dt
        if not self.eval_mode:
            self.goal_timer += self.success_hold_steps_dt
            hold_active_envs = torch.logical_and(success_envs, ~self.training_goal_success_counted)
            counted_success_envs = torch.logical_and(
                hold_active_envs,
                self.training_success_hold_time >= self.goal_refresh_interval,
            )
            if torch.any(counted_success_envs):
                self.goal_achieved_step[counted_success_envs] = 1.0
                self.training_goal_success_counted[counted_success_envs] = True
                self.successes[counted_success_envs] += 1.0
                reached_budget = torch.zeros_like(counted_success_envs)
                if self.max_consecutive_successes > 0:
                    reached_budget = self.successes >= float(self.max_consecutive_successes)
                    self.debug_reset_cause_success_budget[:] = reached_budget
                    self.reset_buf[reached_budget] = 1
                switch_goal_envs = torch.logical_and(counted_success_envs, ~reached_budget)
                if torch.any(switch_goal_envs):
                    self.update_goal(switch_goal_envs.nonzero(as_tuple=False).squeeze(-1))
            if self.goal_switch_timeout_sec > 0.0:
                timeout_goal_envs = self.goal_timer >= self.goal_switch_timeout_sec
                timeout_goal_envs = torch.logical_and(timeout_goal_envs, self.reset_buf == 0)
                if torch.any(timeout_goal_envs):
                    self.reset_buf[timeout_goal_envs] = 1
        else:
            self.goal_timer[self.eval_active_mask] += self.success_hold_steps_dt
        if self.eval_mode:
            counted_eval_success_envs = torch.logical_and(
                success_envs,
                ~self.training_goal_success_counted,
            )
            counted_eval_success_envs = torch.logical_and(
                counted_eval_success_envs,
                self.training_success_hold_time >= self.success_hold_duration,
            )
            if torch.any(counted_eval_success_envs):
                self.goal_achieved_step[counted_eval_success_envs] = 1.0
                self.training_goal_success_counted[counted_eval_success_envs] = True
                self.successes[counted_eval_success_envs] += 1.0

                stage_snapshot = self.eval_goal_stage.clone()
                open_success_ids = torch.logical_and(
                    counted_eval_success_envs,
                    stage_snapshot == 0,
                ).nonzero(as_tuple=False).squeeze(-1)
                close_success_ids = torch.logical_and(
                    counted_eval_success_envs,
                    stage_snapshot == 1,
                ).nonzero(as_tuple=False).squeeze(-1)

                if len(open_success_ids) > 0:
                    self._finalize_eval_stage(
                        open_success_ids,
                        0,
                        goal_distance=goal_distance[open_success_ids],
                        success_override=torch.ones(len(open_success_ids), dtype=torch.bool, device=self.device),
                    )
                    self._set_eval_goal_targets(open_success_ids, 1)
                    self.goal_timer[open_success_ids] = 0.0
                    self.training_goal_success_counted[open_success_ids] = False

                if len(close_success_ids) > 0:
                    self._finalize_eval_stage(
                        close_success_ids,
                        1,
                        goal_distance=goal_distance[close_success_ids],
                        success_override=torch.ones(len(close_success_ids), dtype=torch.bool, device=self.device),
                    )
                    if self.eval_consecutive_mode:
                        self.eval_consecutive_success_cycles[close_success_ids] += 1
                        self._reset_eval_episode_state(close_success_ids)
                        self._set_eval_goal_targets(close_success_ids, 0)
                        self.goal_timer[close_success_ids] = 0.0
                        self.training_goal_success_counted[close_success_ids] = False
                    else:
                        self.reset_buf[close_success_ids] = 1

            if self.eval_goal_timeout > 0.0:
                timed_out_eval_envs = torch.logical_and(
                    self.eval_active_mask,
                    torch.logical_and(
                        self.goal_timer >= self.eval_goal_timeout,
                        ~self.training_goal_success_counted,
                    ),
                )
                timed_out_eval_envs = torch.logical_and(timed_out_eval_envs, self.reset_buf == 0)
                if torch.any(timed_out_eval_envs):
                    open_timeout_ids = torch.logical_and(
                        timed_out_eval_envs,
                        self.eval_goal_stage == 0,
                    ).nonzero(as_tuple=False).squeeze(-1)
                    if len(open_timeout_ids) > 0:
                        self._finalize_eval_stage(
                            open_timeout_ids,
                            0,
                            goal_distance=goal_distance[open_timeout_ids],
                            success_override=torch.zeros(len(open_timeout_ids), dtype=torch.bool, device=self.device),
                        )
                        if self.eval_consecutive_mode:
                            self._record_eval_completion(
                                open_timeout_ids,
                                reason_code=1,
                                stage=0,
                                goal_distance=goal_distance[open_timeout_ids],
                            )
                            self.reset_buf[open_timeout_ids] = 1
                        else:
                            self._set_eval_goal_targets(open_timeout_ids, 1)
                            self.goal_timer[open_timeout_ids] = 0.0
                            self.training_goal_success_counted[open_timeout_ids] = False

                    close_timeout_ids = torch.logical_and(
                        timed_out_eval_envs,
                        self.eval_goal_stage == 1,
                    ).nonzero(as_tuple=False).squeeze(-1)
                    if len(close_timeout_ids) > 0:
                        self._finalize_eval_stage(
                            close_timeout_ids,
                            1,
                            goal_distance=goal_distance[close_timeout_ids],
                            success_override=torch.zeros(len(close_timeout_ids), dtype=torch.bool, device=self.device),
                        )
                        if self.eval_consecutive_mode:
                            self._record_eval_completion(
                                close_timeout_ids,
                                reason_code=1,
                                stage=1,
                                goal_distance=goal_distance[close_timeout_ids],
                            )
                        self.reset_buf[close_timeout_ids] = 1
        control_dt = max(self.success_hold_steps_dt, 1e-8)
        manual_linear_vel = torch.norm((self.object_pos - self.prev_object_pos) / control_dt, p=2, dim=-1)
        prev_quat_diff = quat_mul(self.object_rot, quat_conjugate(self.prev_object_rot))
        manual_angular_vel = 2.0 * torch.asin(
            torch.clamp(torch.norm(prev_quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
        ) / control_dt
        object_pos_reward = torch.nan_to_num(manual_linear_vel, nan=1.0).clamp(0.0, 1.0)
        object_rot_reward = torch.nan_to_num(manual_angular_vel, nan=10.0).clamp(0.0, 10.0)
        hand_qpos_reward = torch.norm(self.hand_dof_pos - self.init_hand_dof_pos, p=2, dim=-1)
        hand_qpos_reward = torch.nan_to_num(hand_qpos_reward, nan=5.0).clamp(0.0, 5.0)
        stable_contact_reward = (torch.sum(self.contact_info[:,-5:], dim=-1) - self.object_cfg['task']['contact_num']).clamp(-5.0, 0.0)
        goal_distance_reward2 = torch.exp(- (goal_distance / self.object_cfg['task']['success_threshold']) ** 2)
        goal_distance_reward1 = torch.clamp(self.mini_goal_distance - goal_distance, min=0.0)
        self.mini_goal_distance = torch.minimum(self.mini_goal_distance, goal_distance)
        self.episode_goal_distance_reward1_sum += goal_distance_reward1
        keep_success_reward = torch.zeros_like(object_pos_reward)
        target_delta = self.cur_targets[:, self.actuated_dof_indices] - self.prev_reward_targets
        action_rate_reward = torch.sum(target_delta ** 2, dim=-1)
        action_rate_reward = torch.nan_to_num(action_rate_reward, nan=50.0).clamp(0.0, 50.0)
        # ---------- drop_reward --------------
        drop_reward = torch.zeros_like(object_pos_reward)
        drop_reward[self.truncated_envs] = 1.0
        # ---------- success_reward ---------------
        success_reward = self.goal_achieved_step.clone()
        
        reward_terms = {
            'ObjPosDeviation': object_pos_reward,
            'ObjRotDeviation': object_rot_reward,
            'HandQposDeviation': hand_qpos_reward,
            'StableContact': stable_contact_reward,
            'GoalDistance1': goal_distance_reward1,
            'GoalDistance2': goal_distance_reward2,
            'Smooth': action_rate_reward,
            'Drop': drop_reward,
            'Success': success_reward,
        }
        self.rew_buf[:] = sum(
            self.reward_scales_current[k] * r
            for k, r in reward_terms.items()
        )

        if self.eval_mode:
            self.rew_buf[~self.eval_active_mask] = 0.0
        for k, r in reward_terms.items():
            self.extras[k] = self.reward_scales_current[k] * r.mean()
        self.extras["episode_cumulative"] = {
            "counted_successes": self.goal_achieved_step,
        }
        self.extras.update(self._get_reward_curriculum_metrics())

    def _compute_student_encoder_observations(self, policy_obs, privileged_obs):
        init_obs = self._get_init_obs()
        parts = [self.proprioception_buf.reshape(self.num_envs, -1), init_obs]
        student_obs = torch.cat(parts, dim=-1)
        if student_obs.shape[1] != self.student_obs_dim:
            raise ValueError(
                f"Configured studentObsDim={self.student_obs_dim}, but student encoder observations built "
                f"{student_obs.shape[1]} dims for mode={self.student_temporal_obs_mode}"
            )
        return student_obs

    def set_student_encoder_obs_enabled(self, enabled=True):
        self.student_encoder_obs_enabled = bool(enabled)

    def get_student_encoder_observations(self):
        return self.student_obs_buf

    def get_teacher_encoder_observations(self):
        return self.teacher_privileged_obs_buf

    def _set_object_dof_targets(self, env_ids, target_values=None):
        env_ids = self._normalize_env_ids(env_ids)
        if len(env_ids) == 0:
            return

        if target_values is None:
            target_values = self._get_init_dof_tensor(env_ids)
        else:
            target_values = target_values.to(device=self.device, dtype=self.cur_targets.dtype)

        object_slice = slice(self.num_hand_dofs, self.num_hand_dofs + self.obj_dof_pos.shape[1])
        self.prev_targets[env_ids, object_slice] = target_values
        self.cur_targets[env_ids, object_slice] = target_values
        self.init_targets[env_ids, object_slice] = target_values

    def _update_history_buf(self, history_buf, current_obs, env_ids=None, step_update=False):
        if not torch.is_tensor(env_ids):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            else:
                env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return
        if step_update:
            if self.proprio_history_len > 1:
                history_buf[env_ids, :-1] = history_buf[env_ids, 1:].clone()
            selected_obs = current_obs if current_obs.shape[0] == len(env_ids) else current_obs[env_ids]
            history_buf[env_ids, -1] = selected_obs
        else:
            history_buf[env_ids] = current_obs.unsqueeze(1).expand(-1, self.proprio_history_len, -1)

    def _get_hand_proprio_obs(self, env_ids=None):
        hand_obs = unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        if env_ids is not None:
            hand_obs = hand_obs[env_ids]
        if self.joint_noise > 0.0:
            hand_obs = torch.clamp(hand_obs + self.joint_noise * torch.randn_like(hand_obs), -1.0, 1.0)
        return hand_obs

    def _set_proprioception_buf(self, env_ids=None, actions=None):
        step_update = env_ids is None
        if step_update:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        hand_obs = self._get_hand_proprio_obs(env_ids=None if step_update else env_ids)
        if step_update:
            selected_actions = self.actions if actions is None else actions
            current_proprio = torch.cat([hand_obs, selected_actions], dim=-1)
        else:
            selected_actions = self.actions[env_ids] if actions is None else actions
            current_proprio = torch.cat([hand_obs, selected_actions], dim=-1)
        self._update_history_buf(self.proprioception_buf, current_proprio, env_ids, step_update=step_update)

    def _get_hand_base_pose(self, env_ids=None):
        if env_ids is None:
            hand_root_pose = self.root_state_tensor[self.hand_indices, 0:7]
        else:
            hand_root_pose = self.root_state_tensor[self.hand_indices[env_ids], 0:7]
        return hand_root_pose[:, :3], hand_root_pose[:, 3:7]

    def _world_positions_to_hand_base(self, world_positions, env_ids=None):
        hand_base_pos, hand_base_rot = self._get_hand_base_pose(env_ids=env_ids)
        base_rot_inv = quat_conjugate(hand_base_rot)
        if world_positions.dim() == 2:
            return quat_apply(base_rot_inv, world_positions - hand_base_pos)

        expanded_rot = base_rot_inv.unsqueeze(1).expand(-1, world_positions.shape[1], -1)
        local_positions = quat_apply(
            expanded_rot.reshape(-1, 4),
            (world_positions - hand_base_pos.unsqueeze(1)).reshape(-1, 3),
        )
        return local_positions.view_as(world_positions)

    def _world_pose_to_hand_base(self, world_pose, env_ids=None):
        hand_base_pos, hand_base_rot = self._get_hand_base_pose(env_ids=env_ids)
        base_rot_inv = quat_conjugate(hand_base_rot)
        local_pos = quat_apply(base_rot_inv, world_pose[:, :3] - hand_base_pos)
        local_rot = quat_mul(base_rot_inv, world_pose[:, 3:7])
        return torch.cat([local_pos, local_rot], dim=-1)

    def _hand_base_pose_to_world(self, local_pose, env_ids=None):
        hand_base_pos, hand_base_rot = self._get_hand_base_pose(env_ids=env_ids)
        world_pos = quat_apply(hand_base_rot, local_pose[:, :3]) + hand_base_pos
        world_rot = quat_mul(hand_base_rot, local_pose[:, 3:7])
        return torch.cat([world_pos, world_rot], dim=-1)

    def _get_nonfinite_env_mask(self):
        tracked_tensors = [
            self.object_pose,
            self.link1_pose,
            self.hand_dof_pos,
            self.hand_dof_vel,
            self.obj_dof_pos,
            self.obj_dof_vel,
            self.fingertip_pos.reshape(self.num_envs, -1),
        ]
        invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for tensor in tracked_tensors:
            invalid |= torch.any(~torch.isfinite(tensor), dim=1)
        return invalid

    def _build_nonfinite_debug_message(self, *, obs=None, policy_obs=None, privileged_obs=None, contact_aux=None):
        messages = []

        def summarize(name, tensor):
            if tensor is None or tensor.numel() == 0:
                return
            nonfinite_mask = ~torch.isfinite(tensor)
            if not torch.any(nonfinite_mask):
                return
            flat_mask = nonfinite_mask.view(tensor.shape[0], -1)
            bad_rows = torch.any(flat_mask, dim=1)
            bad_env_ids = bad_rows.nonzero(as_tuple=False).squeeze(-1)
            first_env = int(bad_env_ids[0].item())
            first_col = int(flat_mask[first_env].nonzero(as_tuple=False)[0].item())
            first_value = tensor[first_env].view(-1)[first_col].item()
            messages.append(
                f"{name}: bad_envs={int(bad_rows.sum().item())}, "
                f"first_env={first_env}, first_col={first_col}, first_value={first_value}"
            )

        summarize("policy_obs", policy_obs)
        summarize("privileged_obs", privileged_obs)
        summarize("critic_policy_contact_aux", contact_aux)
        summarize("obs", obs)
        summarize("rew_buf", self.rew_buf.unsqueeze(-1))
        summarize("object_pose", self.object_pose)
        summarize("link1_pose", self.link1_pose)
        summarize("hand_dof_pos", self.hand_dof_pos)
        summarize("hand_dof_vel", self.hand_dof_vel)
        summarize("obj_dof_pos", self.obj_dof_pos)
        summarize("obj_dof_vel", self.obj_dof_vel)
        summarize("fingertip_pos", self.fingertip_pos.reshape(self.num_envs, -1))

        if not messages:
            return "Non-finite observations or rewards detected in ArtManip"
        return "Non-finite observations or rewards detected in ArtManip: " + "; ".join(messages)

    def _get_init_obs(self):
        init_obs = torch.cat(
            [
                unscale(self.init_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.init_object_pos,
                self.init_object_rot,
                self.init_link1_pose,
                self.init_fingertip_pos,
                self.init_link0_bbx,
                self.init_link1_bbx,
                # torch.tensor([0.02, 0.016, 0.14], dtype=torch.float32, device=self.device).expand(self.num_envs, 3),
            ],
            dim=-1,
        )
        return init_obs

    def _compute_sapg_priv_observations(self):
        self.contact_info[self.at_reset_ids] = self.init_contact_info[self.at_reset_ids]
        self.fingertip_pos[self.at_reset_ids] = self.init_fingertip_pos[self.at_reset_ids].reshape(len(self.at_reset_ids), 5, 3)
        self.object_pos[self.at_reset_ids] = self.init_object_pos[self.at_reset_ids]
        self.object_rot[self.at_reset_ids] = self.init_object_rot[self.at_reset_ids]
        self.object_pose[self.at_reset_ids] = torch.cat(
            [self.init_object_pos[self.at_reset_ids], self.init_object_rot[self.at_reset_ids]],
            dim=-1,
        )
        self.obj_dof_pos[self.at_reset_ids] = self.init_obj_dof_pos[self.at_reset_ids]
        self.obj_dof_vel[self.at_reset_ids] = 0.0
        self.link1_pose[self.at_reset_ids] = self.init_link1_pose[self.at_reset_ids]
        init_obs = self._get_init_obs()
        privileged_obs = torch.cat(
            [
                self.object_pose,
                self.link1_pose,
                self.object_mass,
                self.object_friction,
                self.object_dof_damping,
                self.object_dof_stiffness,
                self.obj_dof_pos - self.init_obj_dof_pos,
                self.obj_dof_vel,
            ],
            dim=-1,
        )
        policy_obs = torch.cat(
            [
                init_obs,
                self._get_hand_proprio_obs(),
                self.actions,
                self.goal_obj_dof_pos - self.init_obj_dof_pos,
                self.fingertip_pos.reshape(self.num_envs,-1),
            ],
            dim=-1,
        )
        if policy_obs.shape[1] != self.policy_obs_dim:
            raise ValueError(
                f"Configured policyObsDim={self.policy_obs_dim}, but compute_observations built {policy_obs.shape[1]} dims"
            )
        if privileged_obs.shape[1] != self.privileged_obs_dim:
            raise ValueError(
                f"Configured privilegedObsDim={self.privileged_obs_dim}, but compute_observations built {privileged_obs.shape[1]} dims"
            )

        return policy_obs, privileged_obs

    def compute_observations(self):
        self._refresh_gym()
        invalid_env_ids = self._get_nonfinite_env_mask().nonzero(as_tuple=False).squeeze(-1)
        if len(invalid_env_ids) > 0 and self.eval_mode:
            self.debug_reset_cause_invalid[invalid_env_ids] = True
            self.at_reset_ids = invalid_env_ids.clone()
            self.reset_idx(invalid_env_ids)
            self._refresh_gym()

        policy_obs, privileged_obs = self._compute_sapg_priv_observations()
        if self.student_encoder_obs_enabled:
            self._set_proprioception_buf()
            student_obs = self._compute_student_encoder_observations(policy_obs, privileged_obs)
            self.teacher_privileged_obs_buf[:] = privileged_obs
            self.student_obs_buf[:] = student_obs
        critic_policy_contact_aux = self.contact_info[:, -5:]
        obs = torch.cat([policy_obs, privileged_obs, critic_policy_contact_aux], dim=-1)
        if not obs.isfinite().all() or not self.rew_buf.isfinite().all():
            raise RuntimeError(
                self._build_nonfinite_debug_message(
                    obs=obs,
                    policy_obs=policy_obs,
                    privileged_obs=privileged_obs,
                    contact_aux=critic_policy_contact_aux,
                )
            )
        self.obs_buf[:] = obs

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.at_reset_ids = env_ids
        self.reset_idx(env_ids)
        self.compute_observations()
        self.reset_buf[:] = 0
        self.timeout_buf[:] = 0
        return super().reset()

    def reset_idx(self, env_ids, goal_env_ids=None):
        if self.eval_mode:
            if not torch.is_tensor(env_ids):
                env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            else:
                env_ids = env_ids.to(device=self.device, dtype=torch.long)

            started_env_ids = env_ids[self.eval_episode_started[env_ids]]
            if len(started_env_ids) > 0:
                if torch.any(self.truncated_envs[started_env_ids]):
                    truncated_env_ids = started_env_ids[self.truncated_envs[started_env_ids]]
                    invalid_ids = truncated_env_ids[self.debug_reset_cause_invalid[truncated_env_ids]]
                    if len(invalid_ids) > 0:
                        self._record_eval_completion(invalid_ids, reason_code=3)
                    fall_ids = truncated_env_ids[
                        torch.logical_and(
                            self.debug_reset_cause_fall[truncated_env_ids],
                            ~self.debug_reset_cause_invalid[truncated_env_ids],
                        )
                    ]
                    if len(fall_ids) > 0:
                        self._record_eval_completion(fall_ids, reason_code=2)
                    timeout_ids = truncated_env_ids[
                        torch.logical_and(
                            self.debug_reset_cause_timeout[truncated_env_ids],
                            torch.logical_and(
                                ~self.debug_reset_cause_invalid[truncated_env_ids],
                                ~self.debug_reset_cause_fall[truncated_env_ids],
                            ),
                        )
                    ]
                    if len(timeout_ids) > 0:
                        self._record_eval_completion(timeout_ids, reason_code=6)

                remaining_reason_ids = started_env_ids[self.eval_completion_reason[started_env_ids] == 0]
                if len(remaining_reason_ids) > 0:
                    self._record_eval_completion(remaining_reason_ids, reason_code=5)

                open_unfinalized_ids = started_env_ids[~self.eval_open_stage_finalized[started_env_ids]]
                if len(open_unfinalized_ids) > 0:
                    self._finalize_eval_stage(
                        open_unfinalized_ids,
                        0,
                        success_override=torch.zeros(len(open_unfinalized_ids), dtype=torch.bool, device=self.device),
                    )
                close_unfinalized_ids = started_env_ids[~self.eval_close_stage_finalized[started_env_ids]]
                if len(close_unfinalized_ids) > 0:
                    self._finalize_eval_stage(
                        close_unfinalized_ids,
                        1,
                        success_override=torch.zeros(len(close_unfinalized_ids), dtype=torch.bool, device=self.device),
                    )

                self.eval_episode_counts[started_env_ids] += 1
                self.eval_active_mask[started_env_ids] = False

            env_ids = env_ids[self.eval_episode_counts[env_ids] < self.eval_episodes_per_grasp]
            if len(env_ids) == 0:
                return

            self.eval_episode_started[env_ids] = True
            self._reset_eval_episode_state(env_ids)
            self.eval_goal_stage[env_ids] = 0
        else:
            env_ids = self._normalize_env_ids(env_ids)

        if self.randomize:
            self._apply_task_randomizations(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0
        select_poses = self.sample_grasps(env_ids)
        grasp_state = self._unpack_grasp_state_batch(select_poses, env_ids=env_ids)
        pose_is_local = self._get_grasp_state_pose_is_local(env_ids)
        object_pose_world = self._cached_pose_to_world(grasp_state["object_pose"], env_ids, pose_is_local)
        self.root_state_tensor[self.object_indices[env_ids], :7] = object_pose_world
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = 0.0
        self.prev_object_pos[env_ids] = object_pose_world[:, :3]
        self.prev_object_rot[env_ids] = object_pose_world[:, 3:7]
        init_object_pose = self._cached_pose_to_hand_base(grasp_state["object_pose"], env_ids, pose_is_local)
        self.init_object_pos[env_ids] = init_object_pose[:, :3]
        self.init_object_rot[env_ids] = init_object_pose[:, 3:7]
        self.init_link0_bbx[env_ids] = self.instance_link0_bbx[self.env2instance[env_ids]]
        self.init_link1_bbx[env_ids] = self.instance_link1_bbx[self.env2instance[env_ids]]
        cached_link1_pose = self._cached_pose_to_hand_base(grasp_state["link1_pose"], env_ids, pose_is_local)
        self.init_link1_pose[env_ids] = cached_link1_pose
        
        object_dof_values = grasp_state["object_dof_pos"]
        self.obj_dof_pos[env_ids, :] = object_dof_values
        self.obj_dof_state_vel[env_ids, :] = 0.0
        self.obj_dof_vel[env_ids, :] = 0.0
        self.prev_obj_dof_pos[env_ids, :] = object_dof_values
        self.init_obj_dof_pos[env_ids, :] = object_dof_values
        self.goal_obj_dof_pos[env_ids, :] = object_dof_values
        if not torch.any(grasp_state["fingertip_pos"] != 0.0):
            if not getattr(self, "allow_missing_cached_fingertip_pos", False):
                raise ValueError(
                    "Cached grasp state is missing fingertip_pos data. "
                    "Please regenerate the grasp caches with fingertip positions included."
                )
            self.init_fingertip_pos[env_ids, :] = 0.0
        else:
            init_fingertip_pos_local = self._cached_positions_to_hand_base(
                grasp_state["fingertip_pos"].view(len(env_ids), -1, 3),
                env_ids,
                pose_is_local,
            )
            self.init_fingertip_pos[env_ids, :] = init_fingertip_pos_local.reshape(len(env_ids), -1)
        self.init_contact_info[env_ids, :] = grasp_state["contact_info"]
        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset hand
        hand_dof_pos = grasp_state["hand_dof_pos"]
        hand_dof_target = grasp_state["hand_dof_target"]
        self.hand_dof_pos[env_ids, :] = hand_dof_pos
        self.hand_dof_vel[env_ids, :] = 0.0
        self.init_hand_dof_pos[env_ids,:] = hand_dof_pos
        zero_actions = torch.zeros((len(env_ids), self.cfg["env"]["numActions"]), dtype=torch.float, device=self.device)
        self.actions[env_ids, :] = zero_actions
        object_target = self._get_init_dof_tensor(env_ids)
        self._set_object_dof_targets(env_ids, object_target)
        self.prev_targets[env_ids, :self.num_hand_dofs] = hand_dof_target
        self.cur_targets[env_ids, :self.num_hand_dofs] = hand_dof_target
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.prev_targets))
        self.init_targets[env_ids, :self.num_hand_dofs]= hand_dof_target
        self.prev_reward_targets[env_ids, :] = hand_dof_target
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self._set_proprioception_buf(env_ids, actions=zero_actions)

        self.progress_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.episode_goal_distance_reward1_sum[env_ids] = 0.0
        self.training_success_hold_time[env_ids] = 0.0
        self.training_goal_success_counted[env_ids] = False
        # randomize start object poses
        if self.eval_mode:
            self.goal_timer[env_ids] = 0.0
            self.goal_refresh_interval[env_ids] = float("inf")
            self._set_eval_goal_targets(env_ids, 0)
        else:
            self.goal_timer[env_ids] = 0.0
            self.goal_refresh_interval[env_ids] = self._sample_goal_refresh_intervals(len(env_ids))
            self.set_reset_goal(env_ids)

    def _sample_goal_refresh_intervals(self, count):
        if count == 0:
            return torch.zeros(0, device=self.device, dtype=torch.float)
        # Training goal switching uses a per-goal sampled dwell time in the
        # success region before switching to the next goal.
        interval_span = self.success_hold_duration_max - self.success_hold_duration_min
        if interval_span == 0.0:
            return torch.full((count,), self.success_hold_duration_min, device=self.device, dtype=torch.float)
        return torch.rand(count, device=self.device) * interval_span + self.success_hold_duration_min

    def _get_next_goal_offsets(self, env_ids):
        if len(env_ids) == 0:
            return torch.zeros((0, self.goal_obj_dof_pos.shape[1]), device=self.device, dtype=torch.float)

        self.goal_indices[env_ids] = (self.goal_indices[env_ids] + 1) % self.goal_offsets.numel()
        selected_goal_offsets = self.goal_offsets[self.goal_indices[env_ids]].view(-1, 1)
        return selected_goal_offsets.expand(-1, self.goal_obj_dof_pos.shape[1])

    def _reset_training_goal_success_state(self, env_ids):
        if len(env_ids) == 0:
            return
        self.goal_achieved_latch[env_ids] = False
        self.training_success_hold_time[env_ids] = 0.0
        self.training_goal_success_counted[env_ids] = False

    def update_goal(self, env_ids):
        if self.eval_mode:
            self._set_eval_goal_targets(env_ids, 0)
            return
        self._reset_training_goal_success_state(env_ids)
        self.goal_timer[env_ids] = 0.0
        new_goal_offsets = self._get_next_goal_offsets(env_ids)
        self.goal_obj_dof_pos[env_ids] = self.init_obj_dof_pos[env_ids] + new_goal_offsets
        self.mini_goal_distance[env_ids] = torch.norm(self.obj_dof_pos[env_ids] - self.goal_obj_dof_pos[env_ids], p=1, dim=-1)

    def set_reset_goal(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_training_goal_success_state(env_ids)
        self.goal_timer[env_ids] = 0.0
        self.goal_indices[env_ids] = 0
        reset_goal_offsets = self.goal_offsets[0].view(1, 1).expand(len(env_ids), self.goal_obj_dof_pos.shape[1])
        self.goal_obj_dof_pos[env_ids] = self.init_obj_dof_pos[env_ids] + reset_goal_offsets
        self.mini_goal_distance[env_ids] = torch.norm(
            self.obj_dof_pos[env_ids] - self.goal_obj_dof_pos[env_ids], p=1, dim=-1
        )

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets_copy = self.cur_targets.clone()
        object_slice = slice(self.num_hand_dofs, self.num_hand_dofs + self.obj_dof_pos.shape[1])
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.hand_dof_lower_limits[self.actuated_dof_indices], self.hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            desired_targets = scale(
                self.actions,
                self.hand_dof_lower_limits[self.actuated_dof_indices],
                self.hand_dof_upper_limits[self.actuated_dof_indices],
            )
            smoothed_targets = self.act_moving_average * desired_targets + (
                1.0 - self.act_moving_average
            ) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = smoothed_targets
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.hand_dof_lower_limits[self.actuated_dof_indices], self.hand_dof_upper_limits[self.actuated_dof_indices])
        self.cur_targets[:, object_slice] = self.prev_targets[:, object_slice]
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.prev_targets[:, object_slice] = self.cur_targets[:, object_slice]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero(as_tuple=False).squeeze(-1)
            if len(force_indices) > 0:
                force_shape = self.rb_forces[force_indices][:, self.object_rb_handles, :].shape
                object_masses = self.object_mass[force_indices].unsqueeze(-1)
                self.rb_forces[force_indices[:, None], self.object_rb_handles, :] = (
                    torch.randn(force_shape, device=self.device) * object_masses * self.force_scale
                )
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

        if self.object_is_lighter:
            self.dof_actuation_forces.zero_()
            object_slice = slice(self.num_hand_dofs, self.num_hand_dofs + self.obj_dof_pos.shape[1])
            active_mask = torch.logical_and(self.obj_dof_pos < -0.6, self.obj_dof_pos >= -1.4)
            lighter_object_force = torch.where(
                active_mask,
                torch.empty_like(self.obj_dof_pos).uniform_(-1.0, 0.0),
                torch.zeros_like(self.obj_dof_pos),
            )
            self.dof_actuation_forces[:, object_slice] = lighter_object_force
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_actuation_forces))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        self.reset_buf[:] = 0
        if hasattr(self, "object_pos") and hasattr(self, "object_rot"):
            self.prev_object_pos[:] = self.object_pos
            self.prev_object_rot[:] = self.object_rot
        self._refresh_gym()
        self._update_obj_dof_velocity()
        self._get_dones()
        self.compute_reward(self.actions)
        self.prev_reward_targets[:] = self.cur_targets[:, self.actuated_dof_indices]
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.at_reset_ids = env_ids.clone()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if not hasattr(self, "root_state_tensor"):
            return

        self.world_object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.world_object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.world_object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.world_object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.world_object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        world_link1_states = self.rigid_body_states[:, self.object_link1_rb_handle]
        fingertip_states = self.rigid_body_states[:, self.fingertip_handles]
        self.world_fingertip_pos = fingertip_states[:, :, 0:3]
        self.world_fingertip_rot = fingertip_states[:, :, 3:7]

        hand_base_pos, hand_base_rot = self._get_hand_base_pose()
        hand_base_rot_inv = quat_conjugate(hand_base_rot)
        self.object_pos = quat_apply(hand_base_rot_inv, self.world_object_pos - hand_base_pos)
        self.object_rot = quat_mul(hand_base_rot_inv, self.world_object_rot)
        self.object_pose = torch.cat([self.object_pos, self.object_rot], dim=-1)
        self.object_linvel = quat_apply(hand_base_rot_inv, self.world_object_linvel)
        self.object_angvel = quat_apply(hand_base_rot_inv, self.world_object_angvel)
        self.link1_pose = self._world_pose_to_hand_base(world_link1_states[:, 0:7])
        self.fingertip_pos = self._world_positions_to_hand_base(self.world_fingertip_pos)
        self.contact_info = get_binary_contact(self.contact_forces[:, self.force_handles], threshold=self.binary_tactile_threshold)



    def _load_init_states(self):
        self.valid_states = []
        self.train_valid_states = []
        self.test_valid_states = []
        self.selected_valid_states = []
        self.success_valid_states = []
        self.instance_grasp_state_pose_frames = []
        self.loaded_grasp_split_keys = set()
        for instance_index, instance_id in enumerate(self.instance_id_list):
            root_states, train_states, test_states, selected_states, success_states, pose_frame = (
                self._load_grasp_state_pools(instance_id, instance_index)
            )
            self.train_valid_states.append(train_states)
            self.test_valid_states.append(test_states)
            self.valid_states.append(root_states)
            self.selected_valid_states.append(selected_states)
            self.success_valid_states.append(success_states)
            self.instance_grasp_state_pose_frames.append(pose_frame)
        self.instance_grasp_state_pose_is_local = torch.tensor(
            [frame == "hand_base" for frame in self.instance_grasp_state_pose_frames],
            device=self.device,
            dtype=torch.bool,
        )
        self.grasp_state_dim = self._infer_loaded_grasp_state_dim()
        self._refresh_grasp_split_tensors()

    def _infer_loaded_grasp_state_dim(self):
        for state_pools in (
            self.valid_states,
            self.train_valid_states,
            self.test_valid_states,
            self.selected_valid_states,
            self.success_valid_states,
        ):
            for states in state_pools:
                if states.dim() == 2:
                    return int(states.shape[1])
        raise RuntimeError("No grasp state pools were initialized.")

    def _validate_grasp_state_pools(self):
        for pool_name, state_pools in (
            ("valid_states", self.valid_states),
            ("train_valid_states", self.train_valid_states),
            ("test_valid_states", self.test_valid_states),
            ("selected_valid_states", self.selected_valid_states),
            ("success_valid_states", self.success_valid_states),
        ):
            for i, states in enumerate(state_pools):
                assert states.dim() == 2, f"{pool_name}[{i}] should be 2D"
                assert states.shape[1] == self.grasp_state_dim, (
                    f"{pool_name}[{i}].shape[1] = {states.shape[1]}, "
                    f"expected {self.grasp_state_dim}"
                )

    def _refresh_grasp_split_tensors(self):
        self._validate_grasp_state_pools()
        self.instance2grasplen = torch.tensor(
            [states.shape[0] for states in self.train_valid_states],
            device=self.device,
            dtype=torch.long,
        )
        self.test_instance2grasplen = torch.tensor(
            [states.shape[0] for states in self.test_valid_states],
            device=self.device,
            dtype=torch.long,
        )
        self.valid_instance2grasplen = torch.tensor(
            [states.shape[0] for states in self.valid_states],
            device=self.device,
            dtype=torch.long,
        )
        self.selected_instance2grasplen = torch.tensor(
            [states.shape[0] for states in self.selected_valid_states],
            device=self.device,
            dtype=torch.long,
        )
        self.success_instance2grasplen = torch.tensor(
            [states.shape[0] for states in self.success_valid_states],
            device=self.device,
            dtype=torch.long,
        )
        self.all_valid_states = torch.cat(self.train_valid_states, dim=0)
        self.all_test_valid_states = torch.cat(self.test_valid_states, dim=0)
        self.all_root_valid_states = torch.cat(self.valid_states, dim=0)
        self.all_selected_valid_states = torch.cat(self.selected_valid_states, dim=0)
        self.all_success_valid_states = torch.cat(self.success_valid_states, dim=0)
        self.instance_offsets = torch.zeros_like(self.instance2grasplen)
        self.instance_offsets[1:] = torch.cumsum(self.instance2grasplen[:-1], dim=0)
        self.test_instance_offsets = torch.zeros_like(self.test_instance2grasplen)
        self.test_instance_offsets[1:] = torch.cumsum(self.test_instance2grasplen[:-1], dim=0)
        self.valid_instance_offsets = torch.zeros_like(self.valid_instance2grasplen)
        self.valid_instance_offsets[1:] = torch.cumsum(self.valid_instance2grasplen[:-1], dim=0)
        self.selected_instance_offsets = torch.zeros_like(self.selected_instance2grasplen)
        self.selected_instance_offsets[1:] = torch.cumsum(self.selected_instance2grasplen[:-1], dim=0)
        self.success_instance_offsets = torch.zeros_like(self.success_instance2grasplen)
        self.success_instance_offsets[1:] = torch.cumsum(self.success_instance2grasplen[:-1], dim=0)

    def _resolve_grasp_state_pose_frame(self, instance_dir):
        metadata_path = os.path.join(instance_dir, "grasp_state_metadata.json")
        if not os.path.exists(metadata_path):
            return "sim_world"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        pose_frame = str(metadata.get("pose_frame", "sim_world")).strip().lower()
        if pose_frame not in {"sim_world", "hand_base"}:
            raise ValueError(
                f"Unsupported grasp pose frame '{pose_frame}' in {metadata_path}. "
                "Expected 'sim_world' or 'hand_base'."
            )
        return pose_frame

    def _get_grasp_cache_instance_dir(self, instance_id):
        return Path("caches") / "initial_grasp" / self.hand_type / self.object_cache_dir_name / str(instance_id)

    def _load_grasp_split_states(self, instance_id, grasp_split):
        instance_dir = self._get_grasp_cache_instance_dir(instance_id)
        split_path = resolve_grasp_cache_path(instance_dir, grasp_split)
        return self._load_grasp_state_file(split_path)

    def _set_instance_grasp_split_states(self, instance_index, grasp_split, states):
        split = normalize_grasp_split(grasp_split)
        if split == "selected":
            self.selected_valid_states[instance_index] = states
        elif split == "success":
            self.success_valid_states[instance_index] = states
        elif split == "test":
            self.test_valid_states[instance_index] = states
        elif split == "valid":
            self.valid_states[instance_index] = states
        else:
            self.train_valid_states[instance_index] = states

    def _ensure_grasp_split_loaded(self, instance_index, grasp_split):
        split = normalize_grasp_split(grasp_split)
        key = (int(instance_index), split)
        if key in self.loaded_grasp_split_keys:
            return

        states = self._load_grasp_split_states(self.instance_id_list[instance_index], split)
        if states.dim() != 2:
            raise ValueError(f"{split} grasp states for instance {self.instance_id_list[instance_index]} should be 2D")
        if states.shape[1] != self.grasp_state_dim:
            raise ValueError(
                f"{split} grasp states for instance {self.instance_id_list[instance_index]} have width "
                f"{states.shape[1]}, expected {self.grasp_state_dim}. Please regenerate that grasp split."
            )
        self._set_instance_grasp_split_states(instance_index, split, states)
        self.loaded_grasp_split_keys.add(key)

    def _load_grasp_state_pools(self, instance_id, instance_index):
        instance_dir = self._get_grasp_cache_instance_dir(instance_id)
        pose_frame = self._resolve_grasp_state_pose_frame(instance_dir)

        active_split = normalize_grasp_split(self.runtime_grasp_split)
        active_states = self._load_grasp_split_states(instance_id, active_split)
        empty_states = active_states[:0].clone()
        valid_states = empty_states
        train_states = empty_states
        test_states = empty_states
        selected_states = empty_states
        success_states = empty_states
        if active_split == "selected":
            selected_states = active_states
        elif active_split == "success":
            success_states = active_states
        elif active_split == "test":
            test_states = active_states
        elif active_split == "valid":
            valid_states = active_states
        else:
            train_states = active_states
        self.loaded_grasp_split_keys.add((int(instance_index), active_split))
        return valid_states, train_states, test_states, selected_states, success_states, pose_frame

    def _load_grasp_state_file(self, path):
        grasp_states = np.load(path)
        if grasp_states.ndim == 1:
            grasp_states = grasp_states[None, :]
        return torch.from_numpy(grasp_states).float().to(self.device)

    def _get_grasp_state_pose_is_local(self, env_ids):
        if hasattr(self, "instance_grasp_state_pose_is_local"):
            return self.instance_grasp_state_pose_is_local[self.env2instance[env_ids]]
        return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

    def _cached_pose_to_world(self, cached_pose, env_ids, pose_is_local):
        world_pose = cached_pose.clone()
        if torch.any(pose_is_local):
            local_env_ids = env_ids[pose_is_local]
            world_pose[pose_is_local] = self._hand_base_pose_to_world(
                cached_pose[pose_is_local],
                env_ids=local_env_ids,
            )
        return world_pose

    def _cached_pose_to_hand_base(self, cached_pose, env_ids, pose_is_local):
        local_pose = cached_pose.clone()
        if torch.any(~pose_is_local):
            world_env_ids = env_ids[~pose_is_local]
            local_pose[~pose_is_local] = self._world_pose_to_hand_base(
                cached_pose[~pose_is_local],
                env_ids=world_env_ids,
            )
        return local_pose

    def _cached_positions_to_hand_base(self, cached_positions, env_ids, pose_is_local):
        local_positions = cached_positions.clone()
        if torch.any(~pose_is_local):
            world_env_ids = env_ids[~pose_is_local]
            local_positions[~pose_is_local] = self._world_positions_to_hand_base(
                cached_positions[~pose_is_local],
                env_ids=world_env_ids,
            )
        return local_positions

    def _get_init_dof_tensor(self, env_ids):
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        batch_size = len(env_ids)
        object_dof_dim = self.obj_dof_pos.shape[1]
        raw_init_dof = self.init_dof

        if isinstance(raw_init_dof, str):
            init_dof_mode = raw_init_dof.strip().lower()
            if init_dof_mode == "lower":
                init_values = self.object_dof_lower_limits[env_ids]
            elif init_dof_mode == "upper":
                init_values = self.object_dof_upper_limits[env_ids]
            else:
                raise ValueError(
                    f"Unsupported object.asset.initDof='{raw_init_dof}'. Expected a number, 'lower', or 'upper'."
                )
        else:
            init_values = torch.full(
                (batch_size,),
                float(raw_init_dof),
                dtype=self.obj_dof_pos.dtype,
                device=self.device,
            )

        return init_values.view(batch_size, 1).expand(-1, object_dof_dim)

    def _get_full_grasp_state_dim(self):
        return (
            2 * self.num_hand_dofs
            + 7
            + 7
            + self.obj_dof_pos.shape[1]
            + self.init_fingertip_pos.shape[1]
            + self.init_contact_info.shape[1]
        )

    def _unpack_grasp_state_batch(self, grasp_states, env_ids=None):
        if grasp_states.dim() != 2:
            raise ValueError(f"grasp_states should be 2D, got shape {tuple(grasp_states.shape)}")

        batch_size = grasp_states.shape[0]
        row_dim = grasp_states.shape[1]
        object_dof_dim = self.obj_dof_pos.shape[1]
        fingertip_dim = self.init_fingertip_pos.shape[1]
        contact_dim = self.init_contact_info.shape[1]
        full_state_dim = self._get_full_grasp_state_dim()

        if env_ids is None:
            env_ids = torch.arange(batch_size, device=self.device, dtype=torch.long)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        if row_dim < self.num_hand_dofs + 7:
            raise ValueError(
                f"Invalid grasp state width {row_dim}; expected at least {self.num_hand_dofs + 7}"
            )

        if row_dim == full_state_dim:
            cursor = 0
            hand_dof_pos = grasp_states[:, cursor:cursor + self.num_hand_dofs]
            cursor += self.num_hand_dofs
            hand_dof_target = grasp_states[:, cursor:cursor + self.num_hand_dofs]
            cursor += self.num_hand_dofs
            object_pose = grasp_states[:, cursor:cursor + 7]
            cursor += 7
            link1_pose = grasp_states[:, cursor:cursor + 7]
            cursor += 7
            object_dof_values = grasp_states[:, cursor:cursor + object_dof_dim]
            cursor += object_dof_dim
            fingertip_pos = grasp_states[:, cursor:cursor + fingertip_dim]
            cursor += fingertip_dim
            contact_info = grasp_states[:, cursor:cursor + contact_dim]
            return {
                "hand_dof_pos": hand_dof_pos,
                "hand_dof_target": hand_dof_target,
                "object_pose": object_pose,
                "link1_pose": link1_pose,
                "object_dof_pos": object_dof_values,
                "fingertip_pos": fingertip_pos,
                "contact_info": contact_info,
            }
        raise ValueError(
            f"Invalid grasp state width {row_dim}; expected exactly {full_state_dim}. "
            "Please regenerate the grasp caches with the current binary-contact layout."
        )


    def sample_grasps(self, env_ids):
        """
        Args:
            env_ids: (k,) long tensor or list/ndarray

        Returns:
            sampled_grasps: (k, c) tensor
        """
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if self.eval_mode and self.eval_grasp_states is not None:
            return self.eval_grasp_states[env_ids]
        if self.runtime_grasp_fixed_state is not None:
            return self.runtime_grasp_fixed_state.unsqueeze(0).expand(len(env_ids), -1)

        if self.runtime_grasp_split == "selected":
            state_pool = self.all_selected_valid_states
            grasp_lens = self.selected_instance2grasplen[self.env2instance[env_ids]]
            grasp_offsets = self.selected_instance_offsets[self.env2instance[env_ids]]
            empty_pool_name = "selected"
        elif self.runtime_grasp_split == "success":
            state_pool = self.all_success_valid_states
            grasp_lens = self.success_instance2grasplen[self.env2instance[env_ids]]
            grasp_offsets = self.success_instance_offsets[self.env2instance[env_ids]]
            empty_pool_name = "success"
        elif self.runtime_grasp_split == "test":
            state_pool = self.all_test_valid_states
            grasp_lens = self.test_instance2grasplen[self.env2instance[env_ids]]
            grasp_offsets = self.test_instance_offsets[self.env2instance[env_ids]]
            empty_pool_name = "test"
        elif self.runtime_grasp_split == "valid":
            state_pool = self.all_root_valid_states
            grasp_lens = self.valid_instance2grasplen[self.env2instance[env_ids]]
            grasp_offsets = self.valid_instance_offsets[self.env2instance[env_ids]]
            empty_pool_name = "valid"
        else:
            state_pool = self.all_valid_states
            grasp_lens = self.instance2grasplen[self.env2instance[env_ids]]
            grasp_offsets = self.instance_offsets[self.env2instance[env_ids]]
            empty_pool_name = "train"

        instance_ids = self.env2instance[env_ids]  # (k,)
        if torch.any(grasp_lens <= 0):
            empty_instances = torch.unique(instance_ids[grasp_lens <= 0]).detach().cpu().tolist()
            empty_instance_ids = [self.instance_id_list[idx] for idx in empty_instances]
            raise ValueError(f"No {empty_pool_name} grasp states available for instances: {empty_instance_ids}")
        local_rand_ids = (torch.rand(len(env_ids), device=self.device) * grasp_lens).long()
        global_ids = grasp_offsets + local_rand_ids
        sampled_grasps = state_pool[global_ids]
        return sampled_grasps

#####################################################################
###=========================jit functions=========================###
#####################################################################


def get_binary_contact(force_vector, threshold=0.001):
        """Convert net contact-force vectors to binary contact indicators."""
        force_magnitude = torch.norm(force_vector, p=2, dim=-1)
        if not torch.is_tensor(threshold):
            threshold = torch.as_tensor(threshold, dtype=force_magnitude.dtype, device=force_magnitude.device)
        else:
            threshold = threshold.to(device=force_magnitude.device, dtype=force_magnitude.dtype)
        while threshold.dim() < force_magnitude.dim():
            threshold = threshold.unsqueeze(-1)
        contact_binary = (force_magnitude > threshold).float()
        return contact_binary
