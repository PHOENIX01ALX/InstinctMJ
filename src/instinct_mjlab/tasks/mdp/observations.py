"""Observation helpers for Instinct Mj perceptive tasks."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F

from mjlab.managers import ManagerTermBase, SceneEntityCfg
from mjlab.sensor import CameraSensor, PinholeCameraPatternCfg, RayCastSensor
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.observations import (
  motion_anchor_ori_b,
  motion_anchor_pos_b,
)
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse


_RAY_DIRECTIONS_CACHE: dict[tuple[int, int, float, str], torch.Tensor] = {}


def _resolve_raycast_pattern(sensor: RayCastSensor) -> tuple[int, int, float]:
  pattern = sensor.cfg.pattern
  if not isinstance(pattern, PinholeCameraPatternCfg):
    raise TypeError(
      "Perceptive raycast depth only supports PinholeCameraPatternCfg. "
      f"Got: {type(pattern).__name__}"
    )
  return pattern.width, pattern.height, pattern.fovy


def _resolve_motion_command(env, command_name: str = "motion") -> MotionCommand:
  command = env.command_manager.get_term(command_name)
  return cast(MotionCommand, command)


def _get_pinhole_ray_directions(
  width: int,
  height: int,
  fovy: float,
  *,
  device: torch.device,
) -> torch.Tensor:
  cache_key = (width, height, float(fovy), str(device))
  if cache_key in _RAY_DIRECTIONS_CACHE:
    return _RAY_DIRECTIONS_CACHE[cache_key]

  v_fov_rad = math.radians(fovy)
  aspect = width / height
  h_fov_rad = 2.0 * math.atan(math.tan(v_fov_rad / 2.0) * aspect)

  u = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
  v = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
  grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")

  ray_x = grid_u.flatten() * math.tan(h_fov_rad / 2.0)
  ray_y = grid_v.flatten() * math.tan(v_fov_rad / 2.0)
  ray_z = -torch.ones_like(ray_x)

  directions = torch.stack([ray_x, ray_y, ray_z], dim=1)
  directions = directions / directions.norm(dim=1, keepdim=True)
  _RAY_DIRECTIONS_CACHE[cache_key] = directions
  return directions


def _depth_from_raycast_distance_to_image_plane(
  sensor: RayCastSensor,
  *,
  depth_clipping_behavior: str = "max",
) -> torch.Tensor:
  """Compute `distance_to_image_plane` from `RayCastSensor` outputs.

  Returns channel-first depth image in meters with shape `[num_envs, 1, H, W]`.
  """
  width, height, fovy = _resolve_raycast_pattern(sensor)
  distances = sensor.data.distances.clone()
  if distances.shape[1] != width * height:
    raise ValueError(
      f"Ray count mismatch: got {distances.shape[1]}, expected {width * height}"
    )

  directions = _get_pinhole_ray_directions(
    width,
    height,
    fovy,
    device=distances.device,
  )
  forward_component = -directions[:, 2].unsqueeze(0)
  depth = distances * forward_component

  missed = distances < 0.0
  if depth_clipping_behavior == "max":
    depth[missed] = sensor.cfg.max_distance
    depth = torch.clamp(depth, max=sensor.cfg.max_distance)
  elif depth_clipping_behavior == "zero":
    depth[missed] = 0.0
    depth[depth > sensor.cfg.max_distance] = 0.0
  elif depth_clipping_behavior == "none":
    depth[missed] = torch.nan
  else:
    raise ValueError(
      "Unsupported depth_clipping_behavior: "
      f"{depth_clipping_behavior!r}. Expected one of ['max', 'zero', 'none']."
    )

  depth = depth.view(sensor.data.distances.shape[0], height, width, 1)
  return depth.permute(0, 3, 1, 2)


def _normalize_crop_resize_depth(
  depth: torch.Tensor,
  *,
  min_depth: float,
  max_depth: float,
  crop_top: int,
  crop_bottom: int,
  crop_left: int,
  crop_right: int,
  output_height: int,
  output_width: int,
) -> torch.Tensor:
  image = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=min_depth)
  image = torch.clamp(image, min=min_depth, max=max_depth)

  denom = max(max_depth - min_depth, 1.0e-6)
  image = (image - min_depth) / denom

  height = image.shape[-2]
  width = image.shape[-1]
  row_start = min(max(crop_top, 0), height)
  row_stop = max(row_start, height - max(crop_bottom, 0))
  col_start = min(max(crop_left, 0), width)
  col_stop = max(col_start, width - max(crop_right, 0))
  if row_start < row_stop and col_start < col_stop:
    image = image[..., row_start:row_stop, col_start:col_stop]

  return F.interpolate(
    image,
    size=(output_height, output_width),
    mode="bilinear",
    align_corners=False,
  )


def _resolve_update_period_steps(params: dict, env) -> int:
  if "update_period_steps" in params:
    return max(int(params["update_period_steps"]), 1)

  update_period_s = params.get("update_period_s")
  if update_period_s is None:
    return 1

  step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
  if step_dt <= 0.0:
    return 1

  return max(int(round(float(update_period_s) / step_dt)), 1)


def perceptive_depth_image(
  env,
  sensor_name: str,
  min_depth: float = 0.0,
  max_depth: float = 2.0,
  crop_top: int = 2,
  crop_bottom: int = 2,
  crop_left: int = 2,
  crop_right: int = 2,
  output_height: int = 18,
  output_width: int = 32,
) -> torch.Tensor:
  """Depth image term with raycaster-first and camera-fallback support.

  Preferred path is `RayCastSensor`.
  If a camera sensor is provided, falls back to rendered depth map preprocessing.
  """
  sensor = env.scene[sensor_name]

  if isinstance(sensor, RayCastSensor):
    depth = _depth_from_raycast_distance_to_image_plane(
      sensor, depth_clipping_behavior="max"
    )
    return _normalize_crop_resize_depth(
      depth,
      min_depth=min_depth,
      max_depth=max_depth,
      crop_top=crop_top,
      crop_bottom=crop_bottom,
      crop_left=crop_left,
      crop_right=crop_right,
      output_height=output_height,
      output_width=output_width,
    )

  if isinstance(sensor, CameraSensor):
    depth = sensor.data.depth
    if depth is None:
      raise RuntimeError(f"Camera '{sensor_name}' has no depth data")
    image = depth.permute(0, 3, 1, 2)
    return _normalize_crop_resize_depth(
      image,
      min_depth=min_depth,
      max_depth=max_depth,
      crop_top=crop_top,
      crop_bottom=crop_bottom,
      crop_left=crop_left,
      crop_right=crop_right,
      output_height=output_height,
      output_width=output_width,
    )

  raise TypeError(
    f"Unsupported sensor type for perceptive_depth_image: {type(sensor).__name__}"
  )


def perceptive_depth_image_no_channel(
  env,
  sensor_name: str,
  min_depth: float = 0.0,
  max_depth: float = 2.0,
  crop_top: int = 2,
  crop_bottom: int = 2,
  crop_left: int = 2,
  crop_right: int = 2,
  output_height: int = 18,
  output_width: int = 32,
) -> torch.Tensor:
  """Depth image without channel dim, useful with observation history stacking."""
  image = perceptive_depth_image(
    env=env,
    sensor_name=sensor_name,
    min_depth=min_depth,
    max_depth=max_depth,
    crop_top=crop_top,
    crop_bottom=crop_bottom,
    crop_left=crop_left,
    crop_right=crop_right,
    output_height=output_height,
    output_width=output_width,
  )
  return image.squeeze(1)


def perceptive_joint_pos_ref(env, command_name: str = "motion") -> torch.Tensor:
  command = _resolve_motion_command(env, command_name=command_name)
  return command.joint_pos


def perceptive_joint_vel_ref(env, command_name: str = "motion") -> torch.Tensor:
  command = _resolve_motion_command(env, command_name=command_name)
  return command.joint_vel


def perceptive_position_ref(env, command_name: str = "motion") -> torch.Tensor:
  return motion_anchor_pos_b(env, command_name=command_name)


def perceptive_rotation_ref(env, command_name: str = "motion") -> torch.Tensor:
  return motion_anchor_ori_b(env, command_name=command_name)


def perceptive_link_pos_b(env, command_name: str = "motion") -> torch.Tensor:
  command = _resolve_motion_command(env, command_name=command_name)
  return command.body_pos_relative_w


def perceptive_link_rot_b(env, command_name: str = "motion") -> torch.Tensor:
  command = _resolve_motion_command(env, command_name=command_name)
  orientation = matrix_from_quat(command.body_quat_relative_w)
  return orientation[..., :2].reshape(orientation.shape[0], orientation.shape[1], 6)


def parkour_amp_reference_projected_gravity(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
) -> torch.Tensor:
  """Projected gravity in the motion-reference base frame for AMP reference states."""
  motion_reference = env.scene[asset_cfg.name]
  base_quat_w = motion_reference.reference_frame.base_quat_w[:, 0]
  gravity_w = torch.zeros(
    base_quat_w.shape[0],
    3,
    device=base_quat_w.device,
    dtype=base_quat_w.dtype,
  )
  gravity_w[:, 2] = -1.0
  return quat_apply_inverse(base_quat_w, gravity_w)


def parkour_amp_reference_joint_pos_rel(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
  robot_name: str = "robot",
) -> torch.Tensor:
  """Reference joint positions relative to robot default joint positions."""
  motion_reference = env.scene[asset_cfg.name]
  robot = env.scene[robot_name]
  joint_pos = motion_reference.reference_frame.joint_pos[:, 0, asset_cfg.joint_ids]
  joint_pos_rel = joint_pos - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  return joint_pos_rel * motion_reference.reference_frame.joint_pos_mask[:, 0, asset_cfg.joint_ids]


def parkour_amp_reference_joint_vel_rel(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
  robot_name: str = "robot",
) -> torch.Tensor:
  """Reference joint velocities relative to robot default joint velocities."""
  motion_reference = env.scene[asset_cfg.name]
  robot = env.scene[robot_name]
  joint_vel = motion_reference.reference_frame.joint_vel[:, 0, asset_cfg.joint_ids]
  joint_vel = joint_vel * motion_reference.reference_frame.joint_vel_mask[:, 0, asset_cfg.joint_ids]
  return joint_vel - robot.data.default_joint_vel[:, asset_cfg.joint_ids]


def parkour_amp_reference_base_lin_vel(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
) -> torch.Tensor:
  """Reference base linear velocity in the motion-reference base frame."""
  motion_reference = env.scene[asset_cfg.name]
  base_quat_w = motion_reference.reference_frame.base_quat_w[:, 0]
  base_lin_vel_w = motion_reference.reference_frame.base_lin_vel_w[:, 0]
  return quat_apply_inverse(base_quat_w, base_lin_vel_w)


def parkour_amp_reference_base_ang_vel(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
) -> torch.Tensor:
  """Reference base angular velocity in the motion-reference base frame."""
  motion_reference = env.scene[asset_cfg.name]
  base_quat_w = motion_reference.reference_frame.base_quat_w[:, 0]
  base_ang_vel_w = motion_reference.reference_frame.base_ang_vel_w[:, 0]
  return quat_apply_inverse(base_quat_w, base_ang_vel_w)


class _PerceptiveRaycastNoisedBase(ManagerTermBase):
  """Stateful raycast depth base class with configurable update period."""

  def __init__(self, cfg, env):
    super().__init__(env)
    params = cfg.params
    self.sensor_name = str(params.get("sensor_name", "perceptive_depth"))
    self.min_depth = float(params.get("min_depth", 0.0))
    self.max_depth = float(params.get("max_depth", 2.0))
    self.crop_top = int(params.get("crop_top", 2))
    self.crop_bottom = int(params.get("crop_bottom", 2))
    self.crop_left = int(params.get("crop_left", 2))
    self.crop_right = int(params.get("crop_right", 2))
    self.output_height = int(params.get("output_height", 18))
    self.output_width = int(params.get("output_width", 32))
    self.update_period_steps = _resolve_update_period_steps(params=params, env=env)

    self._frame_cache: torch.Tensor | None = None
    self._frame_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._frame_step_count = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      self._frame_initialized[:] = False
      self._frame_step_count.zero_()
      if self._frame_cache is not None:
        self._frame_cache.zero_()
      return

    self._frame_initialized[env_ids] = False
    self._frame_step_count[env_ids] = 0
    if self._frame_cache is not None:
      self._frame_cache[env_ids] = 0.0

  def _compute_depth_image(self, env) -> torch.Tensor:
    return perceptive_depth_image(
      env=env,
      sensor_name=self.sensor_name,
      min_depth=self.min_depth,
      max_depth=self.max_depth,
      crop_top=self.crop_top,
      crop_bottom=self.crop_bottom,
      crop_left=self.crop_left,
      crop_right=self.crop_right,
      output_height=self.output_height,
      output_width=self.output_width,
    )

  def _sample_depth_image(self, env) -> torch.Tensor:
    refresh = ~self._frame_initialized
    if self.update_period_steps <= 1:
      refresh = torch.ones_like(refresh)
    else:
      refresh = refresh | (
        self._frame_step_count.remainder(self.update_period_steps) == 0
      )

    if torch.any(refresh):
      image = self._compute_depth_image(env)
      if self._frame_cache is None:
        self._frame_cache = torch.zeros_like(image)
      self._frame_cache[refresh] = image[refresh]
      self._frame_initialized[refresh] = True

    self._frame_step_count += 1
    if self._frame_cache is None:
      raise RuntimeError("Depth frame cache is not initialized.")
    return self._frame_cache


class PerceptiveRaycastNoised(_PerceptiveRaycastNoisedBase):
  """Stateful noised depth term matching raycaster update-period semantics."""

  def __call__(
    self,
    env,
    sensor_name: str | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    crop_top: int | None = None,
    crop_bottom: int | None = None,
    crop_left: int | None = None,
    crop_right: int | None = None,
    output_height: int | None = None,
    output_width: int | None = None,
    update_period_s: float | None = None,
    update_period_steps: int | None = None,
  ) -> torch.Tensor:
    del sensor_name
    del min_depth
    del max_depth
    del crop_top
    del crop_bottom
    del crop_left
    del crop_right
    del output_height
    del output_width
    del update_period_s
    del update_period_steps
    return self._sample_depth_image(env)


class PerceptiveRaycastNoisedHistory(_PerceptiveRaycastNoisedBase):
  """Stateful noised depth-history term with skip-frame semantics.

  Implements `distance_to_image_plane_noised_history`
  + `history_skip_frames` without relying on observation-manager history.
  """

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    params = cfg.params
    self.history_length = max(int(params.get("history_length", 10)), 1)
    self.history_skip_frames = max(int(params.get("history_skip_frames", 0)), 0)

    self._history: torch.Tensor | None = None
    self._history_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    super().reset(env_ids)
    if self._history is None:
      return
    if env_ids is None:
      self._history_initialized[:] = False
      self._history.zero_()
      return
    self._history_initialized[env_ids] = False
    self._history[env_ids] = 0.0

  def __call__(
    self,
    env,
    sensor_name: str | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    crop_top: int | None = None,
    crop_bottom: int | None = None,
    crop_left: int | None = None,
    crop_right: int | None = None,
    output_height: int | None = None,
    output_width: int | None = None,
    history_length: int | None = None,
    history_skip_frames: int | None = None,
    update_period_s: float | None = None,
    update_period_steps: int | None = None,
  ) -> torch.Tensor:
    del sensor_name
    del min_depth
    del max_depth
    del crop_top
    del crop_bottom
    del crop_left
    del crop_right
    del output_height
    del output_width
    del history_length
    del history_skip_frames
    del update_period_s
    del update_period_steps

    image = self._sample_depth_image(env)
    frame = image.squeeze(1)

    if self._history is None:
      self._history = torch.zeros(
        self.num_envs,
        self.history_length,
        frame.shape[-2],
        frame.shape[-1],
        device=frame.device,
        dtype=frame.dtype,
      )

    initialized_env_ids = torch.where(self._history_initialized)[0]
    if initialized_env_ids.numel() > 0:
      self._history[initialized_env_ids, :-1] = self._history[initialized_env_ids, 1:]
      self._history[initialized_env_ids, -1] = frame[initialized_env_ids]

    uninitialized_env_ids = torch.where(~self._history_initialized)[0]
    if uninitialized_env_ids.numel() > 0:
      repeated = frame[uninitialized_env_ids].unsqueeze(1).repeat(
        1, self.history_length, 1, 1
      )
      self._history[uninitialized_env_ids] = repeated
      self._history_initialized[uninitialized_env_ids] = True

    history = self._history
    if self.history_skip_frames > 0:
      history = history[:, :: self.history_skip_frames]
    return history
