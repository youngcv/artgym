# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.artmanip import ArtManip
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import quat_conjugate, quat_mul
from scipy.spatial.transform import Rotation as R


class ArtGrasp(ArtManip):
    """Validation-only grasp generation task.

    This task intentionally reuses ArtManip's simulation/reset behavior so the
    grasp rollout matches the real training environment, but it does not use
    policy observations, rewards, or any training-specific logging logic.
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        instance_id_list = sorted(cfg["object"]["asset"]["instance_id_list"])
        assert len(instance_id_list) == 1, "ArtGrasp only supports exactly one instance."

        self.instance_id = instance_id_list[0]
        self.hand_type = cfg["hand"]["type"]
        self.object_type = cfg["object"]["asset"]["type"]
        self.object_cache_dir_name = Path(str(cfg["object"]["asset"]["asset_root"])).name
        self.allow_missing_cached_fingertip_pos = True
        self.validation_batch_size = int(cfg["env"]["numEnvs"])
        self.validation_batch_cursor = 0
        self.validation_total_candidates = 0
        self.validation_total_batches = 0
        self.validation_qpos_all = None
        self.validation_opos_all = None
        self.validation_saved_state_batches = []
        self.validation_saved_image_count = 0
        self.folder_path = os.path.join(
            "caches", "initial_grasp", self.hand_type, self.object_cache_dir_name, self.instance_id
        )
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        self.pos_devia_threshold = self.object_cfg["task"]["pos_devia_threshold"]
        self.orn_devia_threshold = self.object_cfg["task"]["rot_devia_threshold"]
        self.valid_grasp_path = os.path.join(self.folder_path, "valid_grasps.npy")
        self.valid_split_grasp_path = os.path.join(self.folder_path, "valid", "valid_grasps.npy")
        self.train_grasp_path = os.path.join(self.folder_path, "train", "valid_grasps.npy")
        self.test_grasp_path = os.path.join(self.folder_path, "test", "valid_grasps.npy")
        self.valid_num_path = os.path.join(self.folder_path, "valid_num.jsonl")
        self.grasp_state_metadata_path = os.path.join(self.folder_path, "grasp_state_metadata.json")
        self.temp_grasp_vis_dir = os.path.join(self.folder_path, "_grasp_visualization_tmp")
        self.valid_grasp_vis_dir = os.path.join(self.folder_path, "valid", "grasp_visualization")
        self.train_grasp_vis_dir = os.path.join(self.folder_path, "train", "grasp_visualization")
        self.test_grasp_vis_dir = os.path.join(self.folder_path, "test", "grasp_visualization")
        self.force_links = self.hand_cfg['force_links']
        self.contact_force_threshold = 1e-6
        self.split_seed = 0
        self.filter_unique_grasps = bool(cfg["env"].get("filterUniqueGrasps", False))
        self.unique_pos_threshold = float(cfg["env"].get("uniqueGraspPosThreshold", 0.005))
        self.unique_rot_threshold = float(cfg["env"].get("uniqueGraspRotThreshold", 0.05))
        self.no_split_grasps = bool(cfg["env"].get("noSplitGrasps", False))
        self._prepare_output_dirs()
        self._init_contact_validation_handles()

    def _filter_instance_ids_for_runtime_grasp_split(self, instance_ids):
        # ArtGrasp creates grasp caches, so it must not require a pre-existing
        # train/test/valid cache before the validation env can be constructed.
        return instance_ids

    def _prepare_output_dirs(self):
        os.makedirs(self.folder_path, exist_ok=True)
        if os.path.exists(self.valid_grasp_path):
            os.remove(self.valid_grasp_path)
        if os.path.exists(os.path.dirname(self.valid_split_grasp_path)):
            shutil.rmtree(os.path.dirname(self.valid_split_grasp_path))
        if os.path.exists(os.path.dirname(self.train_grasp_path)):
            shutil.rmtree(os.path.dirname(self.train_grasp_path))
        if os.path.exists(os.path.dirname(self.test_grasp_path)):
            shutil.rmtree(os.path.dirname(self.test_grasp_path))
        if os.path.exists(self.valid_num_path):
            os.remove(self.valid_num_path)
        if os.path.exists(self.temp_grasp_vis_dir):
            shutil.rmtree(self.temp_grasp_vis_dir)
        if self.enable_camera:
            os.makedirs(self.temp_grasp_vis_dir, exist_ok=True)

    def _write_grasp_state_metadata(self):
        with open(self.grasp_state_metadata_path, "w", encoding="utf-8") as f:
            payload = {"pose_frame": "hand_base"}
            if self.filter_unique_grasps:
                payload["unique_filter"] = {
                    "enabled": True,
                    "pos_threshold": self.unique_pos_threshold,
                    "rot_threshold": self.unique_rot_threshold,
                }
            payload["split_mode"] = "valid_only" if self.no_split_grasps else "train_test"
            json.dump(payload, f, indent=2)

    def _refresh_validation_state(self):
        self._refresh_gym()

    def _update_validation_resets(self):
        self.reset_buf[:] = self.check_termination()

    def check_valid(self):
        cond1 = torch.norm(self.object_pos - self.init_object_pos, p=2, dim=-1) < self.pos_devia_threshold
        quat_diff = quat_mul(self.object_rot, quat_conjugate(self.init_object_rot))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
        cond2 = rot_dist < self.orn_devia_threshold
        cond3 = self._check_contact_valid()
        valid = cond1 & cond2 & cond3
        
        return valid
    
    def _init_contact_validation_handles(self):
        if len(self.force_links) != 5:
            raise ValueError(
                f'ArtGrasp contact validation expects exactly 5 tactile force links, got {len(self.force_links)}.'
            )

        env = self.envs[0]
        object_actor_handle = self.gym.find_actor_handle(env, 'object')
        if object_actor_handle < 0:
            raise RuntimeError('Failed to find the object actor handle for contact validation.')

        self.object_link0_handle = self.gym.find_actor_rigid_body_index(
            env, object_actor_handle, 'link_0', gymapi.IndexDomain.DOMAIN_ENV
        )
        self.object_link1_handle = self.gym.find_actor_rigid_body_index(
            env, object_actor_handle, 'link_1', gymapi.IndexDomain.DOMAIN_ENV
        )
        if self.object_link0_handle < 0 or self.object_link1_handle < 0:
            raise RuntimeError('Failed to find object rigid bodies link_0/link_1 for contact validation.')

        self.required_link0_force_handles = {
            int(handle) for handle in self.force_handles[1:].detach().cpu().tolist()
        }
        self.required_link1_force_handle = int(self.force_handles[0].item())

    def _check_contact_valid(self):
        self.gym.fetch_results(self.sim, True)
        env_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for env_id, env in enumerate(self.envs):
            contacts = self.gym.get_env_rigid_contacts(env)
            link0_touched = set()
            link1_touched = False

            for contact in contacts:
                if float(self._contact_field(contact, 'lambda')) <= self.contact_force_threshold:
                    continue

                body0 = int(self._contact_field(contact, 'body0'))
                body1 = int(self._contact_field(contact, 'body1'))
                if body0 < 0 or body1 < 0:
                    continue

                if body0 == self.object_link0_handle and body1 in self.required_link0_force_handles:
                    link0_touched.add(body1)
                elif body1 == self.object_link0_handle and body0 in self.required_link0_force_handles:
                    link0_touched.add(body0)

                if body0 == self.object_link1_handle and body1 == self.required_link1_force_handle:
                    link1_touched = True
                elif body1 == self.object_link1_handle and body0 == self.required_link1_force_handle:
                    link1_touched = True
            env_valid[env_id] = (
                len(link0_touched) >= self.object_cfg['task']['contact_num'] - 1 and link1_touched
            )
        return env_valid
    
    @staticmethod
    def _contact_field(contact, name):
        if hasattr(contact, name):
            return getattr(contact, name)
        if isinstance(contact, np.void):
            return contact[name]
        if isinstance(contact, dict):
            return contact[name]
        return contact[name]
    
    def check_termination(self):
        return self.progress_buf >= self.max_episode_length

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.at_reset_ids = env_ids
        self.reset_idx(env_ids)
        self._refresh_validation_state()
        self.reset_buf.zero_()
        self.timeout_buf.zero_()
        self.rew_buf.zero_()
        self.extras = {}
        return VecTask.reset(self)

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf.zero_()
        self._refresh_validation_state()
        self._update_validation_resets()
        self.rew_buf.zero_()
        self.extras = {}
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

    def update_low_level_control(self):
        self._refresh_gym()
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def sample_grasps(self, env_ids):
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        return self.saved_grasping_states[0][env_ids]

    def _unpack_grasp_state_batch(self, grasp_states, env_ids=None):
        candidate_state_dim = self.num_hand_dofs + 7 + self.obj_dof_pos.shape[1]
        if grasp_states.dim() == 2 and grasp_states.shape[1] == candidate_state_dim:
            hand_dof_pos = grasp_states[:, :self.num_hand_dofs]
            object_pose = grasp_states[:, self.num_hand_dofs:self.num_hand_dofs + 7]
            object_dof_pos = grasp_states[:, self.num_hand_dofs + 7:]
            batch_size = grasp_states.shape[0]
            zero_cache = self._get_empty_cached_state_tensors(batch_size, grasp_states.dtype)
            return {
                "hand_dof_pos": hand_dof_pos,
                "hand_dof_target": hand_dof_pos,
                "object_pose": object_pose,
                # Candidate validation batches do not have cached link_1 pose yet.
                # Reuse the object pose here only to keep the reset snapshot finite;
                # the real link_1 pose is saved into the finalized valid-grasp cache.
                "link1_pose": object_pose.clone(),
                "object_dof_pos": object_dof_pos,
                "fingertip_pos": zero_cache["fingertip_pos"],
                "contact_info": zero_cache["contact_info"],
            }
        return super()._unpack_grasp_state_batch(grasp_states, env_ids=env_ids)

    def _get_empty_cached_state_tensors(self, batch_size, dtype):
        return {
            "fingertip_pos": torch.zeros(
                (batch_size, self.init_fingertip_pos.shape[1]),
                dtype=dtype,
                device=self.device,
            ),
            "contact_info": torch.zeros(
                (batch_size, self.init_contact_info.shape[1]),
                dtype=dtype,
                device=self.device,
            ),
        }

    def _build_batch_states(self, batch_idx):
        start = batch_idx * self.validation_batch_size
        end = start + self.validation_batch_size
        qpos = np.asarray(self.validation_qpos_all[start:end])
        opos = np.asarray(self.validation_opos_all[start:end])
        opos = transform_local_matrices_to_world(opos, self.validation_hand_pose)
        batch_env_ids = torch.arange(qpos.shape[0], device=self.device, dtype=torch.long)
        oqpos = self._get_init_dof_tensor(batch_env_ids).detach().cpu().numpy().astype(np.float32)
        return np.concatenate([qpos, opos, oqpos], axis=-1).astype(np.float32)

    def _set_batch(self, batch_idx):
        batch_states = self._build_batch_states(batch_idx)
        self.saved_grasping_states = [torch.from_numpy(batch_states).float().to(self.device)]
        self.grasp_len_per_instance = [batch_states.shape[0]]
        self.instance_grasp_state_pose_is_local = torch.zeros(1, dtype=torch.bool, device=self.device)

    def _save_valid_images(self, valid_env_ids):
        if not self.enable_camera or len(valid_env_ids) == 0:
            return

        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        for env_id in valid_env_ids:
            env_index = int(env_id.item())
            env = self.envs[env_index]
            color_image = self.gym.get_camera_image(self.sim, env, self.cam_handle_list[env_index], gymapi.IMAGE_COLOR)
            saved_image = color_image.reshape(self.image_height, self.image_width, -1)[..., :3]
            saved_file = os.path.join(self.temp_grasp_vis_dir, f"{str(self.validation_saved_image_count).zfill(5)}.png")
            plt.imsave(saved_file, saved_image)
            self.validation_saved_image_count += 1

    def _get_saved_state_dim(self):
        return (
            2 * self.num_hand_dofs
            + 7
            + 7
            + self.obj_dof_pos.shape[1]
            + self.fingertip_pos.shape[1] * self.fingertip_pos.shape[2]
            + self.contact_info.shape[1]
        )

    def _object_pose_start(self):
        return 2 * self.num_hand_dofs

    def _compute_unique_keep_indices(self, saved_grasping_states):
        total_count = saved_grasping_states.shape[0]
        if total_count == 0:
            return np.empty((0,), dtype=np.int64)

        pose_start = self._object_pose_start()
        object_pose = saved_grasping_states[:, pose_start:pose_start + 7]
        positions = np.asarray(object_pose[:, :3], dtype=np.float32)
        quats = np.asarray(object_pose[:, 3:7], dtype=np.float32)

        quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quat_norms = np.clip(quat_norms, a_min=1e-12, a_max=None)
        quats = quats / quat_norms

        kept_indices = []
        kept_positions = []
        kept_quats = []

        for idx in range(total_count):
            pos = positions[idx]
            quat = quats[idx]

            if not kept_indices:
                kept_indices.append(idx)
                kept_positions.append(pos)
                kept_quats.append(quat)
                continue

            kept_positions_arr = np.asarray(kept_positions, dtype=np.float32)
            pos_dist = np.linalg.norm(kept_positions_arr - pos[None, :], axis=1)
            pos_close = pos_dist <= self.unique_pos_threshold
            if not np.any(pos_close):
                kept_indices.append(idx)
                kept_positions.append(pos)
                kept_quats.append(quat)
                continue

            kept_quats_arr = np.asarray(kept_quats, dtype=np.float32)
            quat_dot = np.abs(np.sum(kept_quats_arr * quat[None, :], axis=1))
            quat_dot = np.clip(quat_dot, -1.0, 1.0)
            rot_dist = 2.0 * np.arccos(quat_dot)
            rot_close = rot_dist <= self.unique_rot_threshold

            if np.any(pos_close & rot_close):
                continue

            kept_indices.append(idx)
            kept_positions.append(pos)
            kept_quats.append(quat)

        return np.asarray(kept_indices, dtype=np.int64)

    def _collect_current_saved_states(self):
        object_pose_local = self._world_pose_to_hand_base(
            self.root_state_tensor[self.object_indices, :7]
        )
        link1_pose_local = self._world_pose_to_hand_base(
            self.rigid_body_states[:, self.object_link1_rb_handle, :7]
        )
        fingertip_pos_local = self._world_positions_to_hand_base(self.world_fingertip_pos)
        return torch.cat(
            [
                self.hand_dof_pos,
                self.cur_targets[:, :self.num_hand_dofs],
                object_pose_local,
                link1_pose_local,
                self.obj_dof_pos,
                fingertip_pos_local.reshape(self.num_envs, -1),
                self.contact_info.reshape(self.num_envs, -1),
            ],
            dim=1,
        )

    def _split_saved_grasps(self, saved_grasping_states):
        total_count = saved_grasping_states.shape[0]
        if total_count == 0:
            empty = saved_grasping_states[:0]
            empty_indices = np.empty((0,), dtype=np.int64)
            return empty, empty, empty_indices, empty_indices
        if total_count == 1:
            empty_indices = np.empty((0,), dtype=np.int64)
            return saved_grasping_states, saved_grasping_states[:0], np.array([0], dtype=np.int64), empty_indices

        rng = np.random.default_rng(self.split_seed)
        shuffled_indices = rng.permutation(total_count)
        test_count = int(round(total_count * 0.2))
        test_count = max(1, min(total_count - 1, test_count))
        train_count = total_count - test_count
        train_indices = shuffled_indices[:train_count]
        test_indices = shuffled_indices[train_count:]
        return saved_grasping_states[train_indices], saved_grasping_states[test_indices], train_indices, test_indices

    def _move_split_visualizations(self, train_indices, test_indices):
        if not self.enable_camera:
            return

        os.makedirs(self.train_grasp_vis_dir, exist_ok=True)
        os.makedirs(self.test_grasp_vis_dir, exist_ok=True)

        for split_indices, target_dir in (
            (train_indices, self.train_grasp_vis_dir),
            (test_indices, self.test_grasp_vis_dir),
        ):
            for new_index, source_index in enumerate(split_indices.tolist()):
                source_file = os.path.join(self.temp_grasp_vis_dir, f"{str(source_index).zfill(5)}.png")
                if not os.path.exists(source_file):
                    continue
                target_file = os.path.join(target_dir, f"{str(new_index).zfill(5)}.png")
                shutil.copy2(source_file, target_file)

        shutil.rmtree(self.temp_grasp_vis_dir, ignore_errors=True)

    def _move_split_visualizations_from_source_indices(self, train_source_indices, test_source_indices):
        if not self.enable_camera:
            return

        os.makedirs(self.train_grasp_vis_dir, exist_ok=True)
        os.makedirs(self.test_grasp_vis_dir, exist_ok=True)

        for split_indices, target_dir in (
            (train_source_indices, self.train_grasp_vis_dir),
            (test_source_indices, self.test_grasp_vis_dir),
        ):
            for new_index, source_index in enumerate(split_indices.tolist()):
                source_file = os.path.join(self.temp_grasp_vis_dir, f"{str(int(source_index)).zfill(5)}.png")
                if not os.path.exists(source_file):
                    continue
                target_file = os.path.join(target_dir, f"{str(new_index).zfill(5)}.png")
                shutil.copy2(source_file, target_file)

        shutil.rmtree(self.temp_grasp_vis_dir, ignore_errors=True)

    def _move_valid_visualizations(self, indices):
        if not self.enable_camera:
            return

        os.makedirs(self.valid_grasp_vis_dir, exist_ok=True)
        for new_index, source_index in enumerate(indices.tolist()):
            source_file = os.path.join(self.temp_grasp_vis_dir, f"{str(int(source_index)).zfill(5)}.png")
            if not os.path.exists(source_file):
                continue
            target_file = os.path.join(self.valid_grasp_vis_dir, f"{str(new_index).zfill(5)}.png")
            shutil.copy2(source_file, target_file)

        shutil.rmtree(self.temp_grasp_vis_dir, ignore_errors=True)

    def _finalize_saved_grasps(self):
        state_dim = self._get_saved_state_dim()
        if self.validation_saved_state_batches:
            saved_grasping_states = np.concatenate(self.validation_saved_state_batches, axis=0)
        else:
            saved_grasping_states = np.empty((0, state_dim), dtype=np.float32)

        original_count = int(saved_grasping_states.shape[0])
        if self.filter_unique_grasps:
            keep_indices = self._compute_unique_keep_indices(saved_grasping_states)
            saved_grasping_states = saved_grasping_states[keep_indices]
        else:
            keep_indices = None

        self._write_grasp_state_metadata()
        np.save(self.valid_grasp_path, saved_grasping_states)
        if self.no_split_grasps:
            os.makedirs(os.path.dirname(self.valid_split_grasp_path), exist_ok=True)
            np.save(self.valid_split_grasp_path, saved_grasping_states)
            if keep_indices is None:
                self._move_valid_visualizations(np.arange(saved_grasping_states.shape[0], dtype=np.int64))
            else:
                self._move_valid_visualizations(keep_indices)
        else:
            train_states, test_states, train_indices, test_indices = self._split_saved_grasps(saved_grasping_states)
            os.makedirs(os.path.dirname(self.train_grasp_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.test_grasp_path), exist_ok=True)
            np.save(self.train_grasp_path, train_states)
            np.save(self.test_grasp_path, test_states)
            if keep_indices is None:
                self._move_split_visualizations(train_indices, test_indices)
            else:
                self._move_split_visualizations_from_source_indices(keep_indices[train_indices], keep_indices[test_indices])
        data = {self.instance_id: int(saved_grasping_states.shape[0])}
        with open(self.valid_num_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

        if self.filter_unique_grasps:
            print(
                f"Saved {saved_grasping_states.shape[0]} unique valid grasps "
                f"(from {original_count}) to {self.valid_grasp_path}"
            )
        else:
            print(f"Saved {saved_grasping_states.shape[0]} valid grasps to {self.valid_grasp_path}")
        if self.no_split_grasps:
            print(f"Saved {saved_grasping_states.shape[0]} valid-only grasps to {self.valid_split_grasp_path}")
        else:
            print(f"Saved {train_states.shape[0]} training grasps to {self.train_grasp_path}")
            print(f"Saved {test_states.shape[0]} test grasps to {self.test_grasp_path}")

    def reset_idx(self, env_ids):
        if self.check_termination().all():
            valid = self.check_valid()
            valid_env_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)

            if len(valid_env_ids) > 0:
                all_states = self._collect_current_saved_states()
                self.validation_saved_state_batches.append(all_states[valid].cpu().numpy())
                self._save_valid_images(valid_env_ids)

            self.validation_batch_cursor += 1
            if self.validation_batch_cursor >= self.validation_total_batches:
                self._finalize_saved_grasps()
                raise SystemExit

            self._set_batch(self.validation_batch_cursor)

        super().reset_idx(env_ids)

    def _load_init_states(self):
        qpos_path = os.path.join(self.folder_path, "qpos.npy")
        opos_path = os.path.join(self.folder_path, "opos.npy")

        qpos = np.load(qpos_path, mmap_mode="r")
        opos = np.load(opos_path, mmap_mode="r")

        assert qpos.shape[0] == opos.shape[0], (
            f"qpos/opos length mismatch in {self.folder_path}: {qpos.shape[0]} vs {opos.shape[0]}"
        )
        assert qpos.shape[0] % self.validation_batch_size == 0, (
            f"len(qpos)={qpos.shape[0]} cannot be divided by batch_size={self.validation_batch_size}"
        )

        self.validation_qpos_all = qpos
        self.validation_opos_all = opos
        self.validation_total_candidates = int(qpos.shape[0])
        self.validation_total_batches = self.validation_total_candidates // self.validation_batch_size
        self.validation_hand_pose = np.array(
            [
                self.hand_start_pose.p.x,
                self.hand_start_pose.p.y,
                self.hand_start_pose.p.z,
                self.hand_start_pose.r.x,
                self.hand_start_pose.r.y,
                self.hand_start_pose.r.z,
                self.hand_start_pose.r.w,
            ],
            dtype=np.float32,
        )

        self._set_batch(0)


def transform_local_matrices_to_world(local_matrices, hand_pose_7d):
    hand_pos = hand_pose_7d[:3]
    hand_quat = hand_pose_7d[3:]

    t_world_hand = np.eye(4, dtype=np.float32)
    t_world_hand[:3, :3] = R.from_quat(hand_quat).as_matrix()
    t_world_hand[:3, 3] = hand_pos

    t_world_local = t_world_hand @ local_matrices
    xyz_world = t_world_local[:, :3, 3]
    rot_matrices = t_world_local[:, :3, :3]
    quats_world = R.from_matrix(rot_matrices).as_quat()
    return np.concatenate([xyz_world, quats_world], axis=-1)
