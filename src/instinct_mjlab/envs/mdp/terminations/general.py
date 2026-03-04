""" Additinoal common termination functions that are not implemented in mjlab. """

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from mjlab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRLEnv

    from instinct_mjlab.motion_reference import MotionReferenceManager


def dataset_exhausted(
    env: ManagerBasedRLEnv,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reset_without_notice: bool = False,
    print_reason: bool = False,
) -> torch.Tensor:
    """Check if the dataset is exhausted.

    Args:
        env: The environment object.
        reset_without_notice: whether to reset the environment without returning True.
    Returns:
        True if the dataset is exhausted, False otherwise.
    """
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
    return_ = torch.logical_not(
        motion_reference.data.validity[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]
    )  # shape: [N,]
    if print_reason and return_.any():
        print("dataset_exhausted: ", return_.sum())
    if reset_without_notice:
        motion_reference.reset(env_ids=return_.nonzero(as_tuple=True)[0])
        return_[:] = False
    return return_


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 3.0,
    print_reason: bool = False,
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    # In mjlab, Scene stores config in _cfg.
    terrain_type = env.scene._cfg.terrain.terrain_type
    if terrain_type == "plane":
        return torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.bool
        )  # we have infinite terrain because it is a plane
    elif terrain_type in ("generator", "hacked_generator"):
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: Entity = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_link_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_link_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return_ = torch.logical_or(x_out_of_bounds, y_out_of_bounds)
        if print_reason and return_.any():
            print(f"The base is out of the terrain border:", return_.sum())
        return return_
    else:
        raise ValueError(
            "Received unsupported terrain type, must be one of: "
            "'plane', 'generator', 'hacked_generator'."
        )


def abnormal_lin_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 40.0,  # [m/s]
):
    asset = env.scene[asset_cfg.name]
    return torch.norm(asset.data.root_link_lin_vel_w, dim=-1) > max_value


def abnormal_ang_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 40.0,  # [rad/s]
):
    asset = env.scene[asset_cfg.name]
    return torch.norm(asset.data.root_link_ang_vel_w, dim=-1) > max_value


def abnormal_joint_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 40.0,  # [rad/s]
):
    asset = env.scene[asset_cfg.name]
    return torch.any(torch.abs(asset.data.joint_vel) > max_value, dim=-1)


def abnormal_joint_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 4000.0,  # [rad/s^2]
):
    asset = env.scene[asset_cfg.name]
    return torch.any(torch.abs(asset.data.joint_acc) > max_value, dim=-1)


class illegal_reset_contact(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(env)
        self.threshold = cfg.params["threshold"]
        self.sensor_cfg = cfg.params.get("sensor_cfg", None)
        self.sensor_name = cfg.params.get("sensor_name", None)
        self.asset_cfg = cfg.params.get("asset_cfg", None)
        self.print_reason = cfg.params.get("print_reason", False)
        self.episode_length_threshold = cfg.params.get("episode_length_threshold", 1)
        self.illegal_contact_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_cfg: SceneEntityCfg | None = None,
        sensor_name: str | None = None,
        asset_cfg: SceneEntityCfg | None = None,
        print_reason: bool = False,
        episode_length_threshold: int = 1,
    ) -> torch.Tensor:
        """Timeout if the robot is reset with some undesired penetration with the environment.
        within the first episode_length_threshold steps.
        """
        if sensor_name is not None:
            contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
            force_mag = torch.max(torch.norm(contact_sensor.data.force_history, dim=-1), dim=-1)[0]  # [B, N]
            if asset_cfg is not None:
                force_mag = force_mag[:, asset_cfg.body_ids]
            contacts = torch.any(force_mag > threshold, dim=1)
        else:
            contact_sensor = env.scene.sensors[sensor_cfg.name]
            force_mag = torch.max(torch.norm(contact_sensor.data.force_history, dim=-1), dim=-1)[0]  # [B, N]
            force_mag = force_mag[:, sensor_cfg.body_ids]
            contacts = torch.any(force_mag > threshold, dim=1)
        self.illegal_contact_counter += contacts.int()
        return_ = torch.logical_and(
            self.illegal_contact_counter >= episode_length_threshold,
            env.episode_length_buf <= episode_length_threshold,
        )
        if return_.any() and print_reason:
            print(f"illegal_reset_contact: {return_.sum()} envs")
        return return_

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.illegal_contact_counter[env_ids] = 0
