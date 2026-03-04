"""Sub-module containing Flat-Patch based velocity command generators."""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from mjlab.utils.lab_api import math as math_utils
from mjlab.utils.lab_api.math import quat_apply_inverse as quat_rotate_inverse
from mjlab.utils.lab_api.math import wrap_to_pi, yaw_quat

from instinct_mjlab.managers import CommandTerm
from instinct_mjlab.terrains import TerrainImporter

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

    from .commands_cfg import PoseVelocityCommandCfg


class PoseVelocityCommand(CommandTerm):
    """Velocity command based on the 2D flat patch command generator."""

    cfg: PoseVelocityCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: PoseVelocityCommandCfg, env: ManagerBasedRlEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Entity = env.scene[cfg.entity_name]

        # crete buffers to store the command
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.max_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_exp_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_exp_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        # debug ratios to localize "standing still" behavior
        self.metrics["command_nonzero_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_near_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["standing_env_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["random_velocity_env_ratio"] = torch.zeros(self.num_envs, device=self.device)

        # obtain the terrain asset
        self.terrain: TerrainImporter = env.scene["terrain"]

        self.lin_vel_x_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.lin_vel_y_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.ang_vel_z_range = torch.zeros(self.num_envs, 2, device=self.device)

        self.random_lin_vel_x_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_lin_vel_y_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_ang_vel_z_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_velocity_indices = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.random_lin_vel_x = torch.zeros(self.num_envs, device=self.device)
        self.random_lin_vel_y = torch.zeros(self.num_envs, device=self.device)
        self.random_ang_vel_z = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.velocity_ranges is not None:
            terrain_generator_cfg = self.terrain.cfg.terrain_generator
            proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()])
            proportions /= np.sum(proportions)

            # find the sub-terrain index for each column
            # we generate the terrains based on their proportion (not randomly sampled)
            sub_indices = []
            for index in range(terrain_generator_cfg.num_cols):
                sub_index = np.min(np.where(index / terrain_generator_cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
                sub_indices.append(sub_index)
            sub_indices = np.array(sub_indices, dtype=np.int32)
            sub_terrains_names = list(terrain_generator_cfg.sub_terrains.keys())
            for key, value in self.cfg.velocity_ranges.items():
                if key in sub_terrains_names:
                    terrain_type_index = sub_terrains_names.index(key)
                    type_indices = np.where(sub_indices == terrain_type_index)[0]
                    for type_indice in type_indices:
                        env_indices = torch.where(self.terrain.terrain_types == type_indice)[0]
                        self.lin_vel_x_range[env_indices, 0] = value["lin_vel_x"][0]
                        self.lin_vel_x_range[env_indices, 1] = value["lin_vel_x"][1]
                        self.lin_vel_y_range[env_indices, 0] = value["lin_vel_y"][0]
                        self.lin_vel_y_range[env_indices, 1] = value["lin_vel_y"][1]
                        self.ang_vel_z_range[env_indices, 0] = value["ang_vel_z"][0]
                        self.ang_vel_z_range[env_indices, 1] = value["ang_vel_z"][1]
                else:
                    raise RuntimeError(f"Terrain type {key} not found in the terrain generator sub-terrain names.")

            if self.cfg.random_velocity_terrain is not None:
                for key in self.cfg.random_velocity_terrain:
                    terrain_type_index = sub_terrains_names.index(key)
                    type_indices = np.where(sub_indices == terrain_type_index)[0]
                    for type_indice in type_indices:
                        env_indices = torch.where(self.terrain.terrain_types == type_indice)[0]
                        self.random_velocity_indices[env_indices] = True

        self.random_lin_vel_x_range[:, 0] = self.cfg.ranges.lin_vel_x[0]
        self.random_lin_vel_x_range[:, 1] = self.cfg.ranges.lin_vel_x[1]
        self.random_lin_vel_y_range[:, 0] = self.cfg.ranges.lin_vel_y[0]
        self.random_lin_vel_y_range[:, 1] = self.cfg.ranges.lin_vel_y[1]
        self.random_ang_vel_z_range[:, 0] = self.cfg.ranges.ang_vel_z[0]
        self.random_ang_vel_z_range[:, 1] = self.cfg.ranges.ang_vel_z[1]

        # obtain the valid targets from the terrain
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    @property
    def pose_command(self) -> torch.Tensor:
        """The desired base pose command in the base frame. Shape is (num_envs, 3)."""
        return torch.cat([self.pos_command_b[:, :2], self.vel_command_b[:, 0:1]], dim=1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1)
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]) / max_command_step
        )
        lin_vel_error = torch.sum(
            torch.square(self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2]),
            dim=1,
        )
        self.metrics["tracking_exp_vel_xy"] += (
            torch.exp(-lin_vel_error / self.cfg.lin_vel_metrics_std**2) / self._env.max_episode_length
        )
        angular_vel_error = torch.square(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
        self.metrics["tracking_exp_vel_yaw"] += (
            torch.exp(-angular_vel_error / self.cfg.ang_vel_metrics_std**2) / self._env.max_episode_length
        )
        target_vec = self.pos_command_w - self.robot.data.root_link_pos_w[:, :3]
        target_dist = torch.norm(target_vec[:, :2], dim=1)
        cmd_lin_norm = torch.norm(self.vel_command_b[:, :2], dim=1)
        cmd_nonzero = torch.logical_or(cmd_lin_norm > 1e-6, torch.abs(self.vel_command_b[:, 2]) > 1e-6)
        self.metrics["command_nonzero_ratio"] += cmd_nonzero.float() / self._env.max_episode_length
        self.metrics["target_near_ratio"] += (
            (target_dist <= self.cfg.target_dis_threshold).float() / self._env.max_episode_length
        )
        self.metrics["standing_env_ratio"] += self.is_standing_env.float() / self._env.max_episode_length
        self.metrics["random_velocity_env_ratio"] += (
            self.random_velocity_indices.float() / self._env.max_episode_length
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets from the terrain
        ids = torch.randint(0, self.valid_targets.shape[2], size=(len(env_ids),), device=self.device)
        self.pos_command_w[env_ids] = self.valid_targets[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], ids
        ]

        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.max_command_b[env_ids, 0] = self.lin_vel_x_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.lin_vel_x_range[env_ids, 1] - self.lin_vel_x_range[env_ids, 0]
        )
        # -- linear velocity - y direction
        self.max_command_b[env_ids, 1] = self.lin_vel_y_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.lin_vel_y_range[env_ids, 1] - self.lin_vel_y_range[env_ids, 0]
        )
        # -- ang vel yaw - rotation around z
        self.max_command_b[env_ids, 2] = self.ang_vel_z_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.ang_vel_z_range[env_ids, 1] - self.ang_vel_z_range[env_ids, 0]
        )
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # Only update random velocities for envs that are currently being resampled AND are marked for random velocity
        # Create a mask for the current batch of env_ids
        current_batch_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        current_batch_mask[env_ids] = True

        # Find intersection: envs in this batch that are also random velocity envs
        update_mask = current_batch_mask & self.random_velocity_indices
        random_velocity_env_ids = update_mask.nonzero(as_tuple=False).flatten()

        if len(random_velocity_env_ids) > 0:
            self.random_lin_vel_x[random_velocity_env_ids] = self.random_lin_vel_x_range[
                random_velocity_env_ids, 0
            ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
                self.random_lin_vel_x_range[random_velocity_env_ids, 1]
                - self.random_lin_vel_x_range[random_velocity_env_ids, 0]
            )
            self.random_lin_vel_y[random_velocity_env_ids] = self.random_lin_vel_y_range[
                random_velocity_env_ids, 0
            ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
                self.random_lin_vel_y_range[random_velocity_env_ids, 1]
                - self.random_lin_vel_y_range[random_velocity_env_ids, 0]
            )
            self.random_ang_vel_z[random_velocity_env_ids] = self.random_ang_vel_z_range[
                random_velocity_env_ids, 0
            ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
                self.random_ang_vel_z_range[random_velocity_env_ids, 1]
                - self.random_ang_vel_z_range[random_velocity_env_ids, 0]
            )
            self.random_ang_vel_z *= torch.abs(self.random_ang_vel_z) > 0.5

    def _update_command(self):
        """Re-target the position command to the current root state."""
        target_vec = self.pos_command_w - self.robot.data.root_link_pos_w[:, :3]
        target_dist = torch.norm(target_vec[:, :2], dim=1)
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_link_quat_w), target_vec)
        self.vel_command_b[:, :2] = self.pos_command_b[:, :2] * self.cfg.velocity_control_stiffness

        # set heading command to point towards target
        target_vec = self.pos_command_w - self.robot.data.root_link_pos_w
        target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])

        # compute errors to find the closest direction to the current heading
        # this is done to avoid the discontinuity at the -pi/pi boundary
        self.heading_command_w = wrap_to_pi(target_direction - self.robot.data.heading_w)

        self.vel_command_b[:, 2] = self.heading_command_w * self.cfg.heading_control_stiffness

        # scale linear velocity so the dominant axis hits its limit and
        # the other axis preserves its ratio
        vx = self.vel_command_b[:, 0]
        vy = self.vel_command_b[:, 1]
        min_x = (
            -self.max_command_b[:, 0]
            if not self.cfg.only_positive_lin_vel_x
            else torch.zeros_like(self.max_command_b[:, 0])
        )
        min_y = -self.max_command_b[:, 1]
        max_x = self.max_command_b[:, 0]
        max_y = self.max_command_b[:, 1]
        eps = 1e-6

        abs_vx = vx.abs()
        abs_vy = vy.abs()

        if not self.cfg.only_positive_lin_vel_x:
            # clamp each axis independently
            clamped_vx = torch.clamp(abs_vx, min=min_x, max=max_x)
            clamped_vy = torch.clamp(abs_vy, min=min_y, max=max_y)

            # compute scale for whichever axis is dominant
            scale_x = clamped_vx / (abs_vx + eps)
            scale_y = clamped_vy / (abs_vy + eps)
            scale = torch.where(abs_vx >= abs_vy, scale_x, scale_y)

            # apply scale and restore sign
            self.vel_command_b[:, 0] = vx * scale
            self.vel_command_b[:, 1] = vy * scale

        else:
            self.vel_command_b[:, 0] = torch.clamp(vx, min=min_x, max=max_x)
            self.vel_command_b[:, 1] = torch.clamp(vy, min=min_y, max=max_y)

        self.vel_command_b[:, 2] = torch.clamp(
            self.vel_command_b[:, 2],
            self.cfg.ranges.ang_vel_z[0],
            self.cfg.ranges.ang_vel_z[1],
        )
        self.vel_command_b[:] *= (target_dist > self.cfg.target_dis_threshold).unsqueeze(-1)
        self.vel_command_b[:, :2] *= (
            (torch.norm(self.vel_command_b[:, :2], dim=1) > self.cfg.lin_vel_threshold).float().unsqueeze(-1)
        )
        self.vel_command_b[:, 2] *= (torch.abs(self.vel_command_b[:, 2]) > self.cfg.ang_vel_threshold).float()
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        random_velocity_env_ids = self.random_velocity_indices.nonzero(as_tuple=False).flatten()
        self.vel_command_b[random_velocity_env_ids, 0] = self.random_lin_vel_x[random_velocity_env_ids]
        self.vel_command_b[random_velocity_env_ids, 1] = self.random_lin_vel_y[random_velocity_env_ids]
        self.vel_command_b[random_velocity_env_ids, 2] = self.random_ang_vel_z[random_velocity_env_ids]

    def _debug_vis_impl(self, visualizer) -> None:
        """Draw target positions and velocity arrows using MuJoCo geometry.
        
        This replaces the legacy marker path with native MuJoCo visualization.
        """
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        # Convert to numpy for visualization
        pos_commands_w = self.pos_command_w.cpu().numpy()
        vel_commands_b = self.vel_command_b.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_ws = self.robot.data.root_link_quat_w.cpu().numpy()
        lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()

        # Visualization parameters
        goal_radius = self.cfg.target_dis_threshold
        goal_height = 0.1
        patch_height = 0.05
        arrow_z_offset = 0.5
        arrow_scale = 0.5

        for batch in env_indices:
            # Skip if robot appears uninitialized (at origin)
            if np.linalg.norm(base_pos_ws[batch]) < 1e-6:
                continue

            # 1. Draw target position as red cylinder
            goal_pos = pos_commands_w[batch]
            goal_start = goal_pos.copy()
            goal_end = goal_pos.copy()
            goal_end[2] += goal_height
            visualizer.add_cylinder(
                start=goal_start,
                end=goal_end,
                radius=goal_radius,
                color=(1.0, 0.0, 0.0, 0.6),
                label=f"goal_{batch}",
            )

            # 2. Draw commanded velocity arrow (green)
            base_pos = base_pos_ws[batch]
            arrow_start = base_pos.copy()
            arrow_start[2] += arrow_z_offset
            
            # Convert velocity command from body frame to world frame
            vel_cmd_b = vel_commands_b[batch]
            quat_w = base_quat_ws[batch]
            # Simple rotation: only yaw matters for horizontal velocity
            yaw = np.arctan2(2.0 * (quat_w[0] * quat_w[3] + quat_w[1] * quat_w[2]),
                           1.0 - 2.0 * (quat_w[2]**2 + quat_w[3]**2))
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            vel_cmd_w = np.array([
                cos_yaw * vel_cmd_b[0] - sin_yaw * vel_cmd_b[1],
                sin_yaw * vel_cmd_b[0] + cos_yaw * vel_cmd_b[1],
                0.0
            ])
            
            arrow_end = arrow_start + vel_cmd_w * arrow_scale
            visualizer.add_arrow(
                start=arrow_start,
                end=arrow_end,
                color=(0.1, 1.0, 0.1, 0.8),  # Bright green
                width=0.02,
                label=f"cmd_vel_{batch}",
            )

            # 3. Draw actual velocity arrow (blue)
            lin_vel_b = lin_vel_bs[batch]
            vel_actual_w = np.array([
                cos_yaw * lin_vel_b[0] - sin_yaw * lin_vel_b[1],
                sin_yaw * lin_vel_b[0] + cos_yaw * lin_vel_b[1],
                0.0
            ])
            
            arrow_end_actual = arrow_start + vel_actual_w * arrow_scale
            visualizer.add_arrow(
                start=arrow_start,
                end=arrow_end_actual,
                color=(0.1, 0.1, 1.0, 0.8),  # Bright blue
                width=0.02,
                label=f"actual_vel_{batch}",
            )

            # 4. Optionally draw all flat patches (if patch_vis is enabled)
            if self.cfg.patch_vis:
                valid_targets = self.valid_targets.cpu().numpy()
                terrain_level = self.terrain.terrain_levels[batch].item()
                terrain_type = self.terrain.terrain_types[batch].item()
                patches = valid_targets[terrain_level, terrain_type]
                
                for i, patch_pos in enumerate(patches):
                    if np.linalg.norm(patch_pos) < 1e-6:  # Skip invalid patches
                        continue
                    patch_start = patch_pos.copy()
                    patch_end = patch_pos.copy()
                    patch_end[2] += patch_height
                    visualizer.add_cylinder(
                        start=patch_start,
                        end=patch_end,
                        radius=goal_radius,
                        color=(0.0, 0.0, 1.0, 0.3),
                        label=f"patch_{batch}_{i}",
                    )
