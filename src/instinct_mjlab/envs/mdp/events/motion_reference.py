from __future__ import annotations

import math
import torch
import tqdm
from typing import TYPE_CHECKING

from instinct_mjlab.motion_reference import MotionReferenceManager
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api import math as math_utils

if TYPE_CHECKING:
    from instinct_mjlab.envs.mdp import BeyondMimicAdaptiveWeighting
    from instinct_mjlab.motion_reference import MotionReferenceData
    from mjlab.entity import Entity as Articulation
    from mjlab.entity import Entity as RigidObject
    from mjlab.envs import ManagerBasedRlEnv as ManagerBasedEnv


@requires_model_fields("body_gravcomp", recompute=RecomputeLevel.set_const_fixed)
def virtualize_articulation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot_reference"),
):
    """Virtualize the robot reference (Articulation) in the simulation by removing all
    their collision shapes and setting their mass to zero.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids (num_envs)
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(dtype=torch.long, device=env.device)

    # resolve body indices (global ids in MuJoCo model arrays)
    if asset_cfg.body_ids == slice(None):
        body_ids = asset.indexing.body_ids
    else:
        body_ids_local = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device=env.device)
        body_ids = asset.indexing.body_ids[body_ids_local]

    # MuJoCo does not provide a legacy-backend-style per-body gravity toggle.
    # gravcomp=1.0 compensates full gravity for the selected bodies.
    env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
    env.sim.model.body_gravcomp[env_grid, body_grid] = 1.0


def match_motion_ref_with_scene(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """Match information between motion reference and the scene, use at 'startup' stage."""
    print("[Event] Match motion reference with scene.")
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    motion_ref.match_scene(env.scene)


def reset_robot_state_by_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_offset: list = [0.0, 0.0, 0.1],
    dof_vel_ratio: float = 0.5,
    base_lin_vel_ratio: float = 0.5,
    base_ang_vel_ratio: float = 0.5,
    randomize_pose_range: dict[str, tuple[float, float]] = {},
    randomize_velocity_range: dict[str, tuple[float, float]] = {},
    randomize_joint_pos_range: tuple[float, float] = (0.0, 0.0),
):
    """Reset robot state based on motion reference with optional randomization.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to reset.
        motion_ref_cfg: Motion reference configuration.
        asset_cfg: Robot asset configuration.
        position_offset: Position offset to apply [x, y, z].
        dof_vel_ratio: Ratio to scale joint velocities.
        base_lin_vel_ratio: Ratio to scale base linear velocities.
        base_ang_vel_ratio: Ratio to scale base angular velocities.
        randomize_pose_range: Optional pose randomization ranges for ["x", "y", "z", "roll", "pitch", "yaw"].
        randomize_velocity_range: Optional velocity randomization ranges for ["x", "y", "z", "roll", "pitch", "yaw"].
        randomize_joint_pos_range: Optional joint position randomization range (min, max).
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]

    # reset the motion reference object
    # motion reference (as sensor) is already reset(ed) in scene.reset(...)
    # motion_ref.reset(env_ids)

    # reset the robot state based on motion reference data
    data: MotionReferenceData = motion_ref.data  # triggering the motion reference to update the data.
    motion_ref_init_state = motion_ref.get_init_reference_state(env_ids)

    # reset the robot state
    base_pos_w = motion_ref_init_state.base_pos_w
    base_pos_w = motion_ref_init_state.base_pos_w + torch.tensor(
        position_offset, device=motion_ref_init_state.base_pos_w.device
    ).unsqueeze(0)

    # apply randomizations on the reset states if specified
    if randomize_pose_range:
        # Apply pose randomization similar to reset_root_state_uniform
        range_list = [randomize_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=base_pos_w.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=base_pos_w.device
        )

        # Add position randomization
        base_pos_w += rand_samples[:, 0:3]

        # Add orientation randomization
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        base_quat_w = math_utils.quat_mul(orientations_delta, motion_ref_init_state.base_quat_w)
    else:
        base_quat_w = motion_ref_init_state.base_quat_w

    # write the root pose to the simulation
    asset.write_root_link_pose_to_sim(
        torch.cat(
            [
                base_pos_w,
                base_quat_w,
            ],
            dim=-1,
        ),
        env_ids=env_ids,
    )

    # apply velocity randomization if specified
    base_lin_vel_w = motion_ref_init_state.base_lin_vel_w * base_lin_vel_ratio
    base_ang_vel_w = motion_ref_init_state.base_ang_vel_w * base_ang_vel_ratio
    if randomize_velocity_range:
        # Apply velocity randomization
        range_list = [randomize_velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=base_lin_vel_w.device)
        vel_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=base_lin_vel_w.device
        )

        # Add velocity randomization
        base_lin_vel_w += vel_samples[:, 0:3]
        base_ang_vel_w += vel_samples[:, 3:6]

    # write the root velocity to the simulation
    asset.write_root_link_velocity_to_sim(
        torch.cat(
            [
                base_lin_vel_w,
                base_ang_vel_w,
            ],
            dim=-1,
        ),
        env_ids=env_ids,
    )

    # apply joint position randomization if specified
    joint_pos = motion_ref_init_state.joint_pos
    joint_vel = motion_ref_init_state.joint_vel * dof_vel_ratio
    if randomize_joint_pos_range != (0.0, 0.0):
        # Apply joint position randomization
        joint_pos_noise = math_utils.sample_uniform(
            randomize_joint_pos_range[0], randomize_joint_pos_range[1], joint_pos.shape, device=joint_pos.device
        )
        joint_pos += joint_pos_noise

    # write the joint state to the simulation
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids,
    )


def beyondmimic_bin_fail_counter_smoothing(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    curriculum_name: str,
):
    # Acquire the curriculum term instance, which should be a ManagerTermBase
    curriculum: BeyondMimicAdaptiveWeighting = env.curriculum_manager._term_cfgs[
        env.curriculum_manager._term_names.index(curriculum_name)
    ].func

    curriculum.motion_bin_fail_counter._concatenated_tensor.mul_(1 - curriculum.adaptive_alpha)
    curriculum.motion_bin_fail_counter._concatenated_tensor.add_(
        curriculum.adaptive_alpha * curriculum.current_motion_bin_fail_counter._concatenated_tensor
    )
    curriculum.current_motion_bin_fail_counter._concatenated_tensor.fill_(0)
