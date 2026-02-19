from dataclasses import field, dataclass, MISSING
import math
import os

import mjlab.envs.mdp as mdp
import mjlab.sim as sim_utils
from mjlab.assets import ArticulationCfg, AssetBaseCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg
from mjlab.managers import ObservationGroupCfg as ObsGroupCfg
from mjlab.managers import ObservationTermCfg as ObsTermCfg
from mjlab.managers import RewardTermCfg as RewTermCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers import TerminationTermCfg as DoneTermCfg
from mjlab.scene import SceneCfg as InteractiveSceneCfg
from mjlab.sensors import ContactMatch, ContactSensorCfg, ObjRef, RayCasterCfg, patterns
from mjlab.terrains import FlatPatchSamplingCfg
from mjlab.utils.noise import UniformNoiseCfg

import instinct_mjlab.envs.mdp as instinct_mdp
from instinct_mjlab.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg
from instinct_mjlab.monitors import (
    MonitorTermCfg,
    MotionReferenceMonitorTerm,
    ShadowingJointPosMonitorTerm,
    ShadowingJointVelMonitorTerm,
    ShadowingLinkPosMonitorTerm,
    ShadowingPositionMonitorTerm,
    ShadowingRotationMonitorTerm,
)
from instinct_mjlab.motion_reference import MotionReferenceManagerCfg
from instinct_mjlab.sensors.grouped_ray_caster import GroupedRayCasterCameraCfg
from instinct_mjlab.sensors.noisy_camera import NoisyGroupedRayCasterCameraCfg
from instinct_mjlab.tasks.shadowing import mdp as shadowing_mdp
from instinct_mjlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinct_mjlab.terrains.terrain_importer_cfg import TerrainImporterCfg
from instinct_mjlab.terrains.trimesh import MotionMatchedTerrainCfg
from instinct_mjlab.utils.noise import (
    CropAndResizeCfg,
    DepthArtifactNoiseCfg,
    DepthContourNoiseCfg,
    DepthNormalizationCfg,
    DepthSkyArtifactNoiseCfg,
    GaussianBlurNoiseCfg,
    RangeBasedGaussianNoiseCfg,
    StereoTooCloseNoiseCfg,
)

# PROPRIO_HISTORY_LENGTH = 0
PROPRIO_HISTORY_LENGTH = 8


@dataclass(kw_only=True)
class PerceptiveShadowingSceneCfg(InteractiveSceneCfg):
    """Configuration for the BeyondMimic scene with necessary scene entities as motion reference."""

    env_spacing: float = 4.0


    # robots
    robot: ArticulationCfg = None  # Set by concrete env cfg subclass

    # robot reference articulation
    robot_reference: ArticulationCfg = None

    # motion reference
    motion_reference: MotionReferenceManagerCfg = None  # Set by concrete env cfg subclass

    # terrain
    terrain: object = field(default_factory=lambda: TerrainImporterCfg(
        prim_path="/World/ground",
        # terrain_type="plane",
        # terrain_generator=None,
        terrain_type="hacked_generator",
        terrain_generator=FiledTerrainGeneratorCfg(
            size=(9, 12),
            border_width=0.0,
            border_height=0.0,
            num_rows=7,
            num_cols=7,
            sub_terrains={
                "motion_matched": MotionMatchedTerrainCfg(
                    proportion=1.0,
                    path="PLACEHOLDER",  # Will be overridden in concrete env cfg __post_init__
                    metadata_yaml="PLACEHOLDER",  # Will be overridden in concrete env cfg __post_init__
                ),
            },
        ),
        physics_material=None,
        visual_material=None,
    ))


    # lights
    light: object = field(default_factory=lambda: AssetBaseCfg(
        prim_path="/World/light",
        spawn=None,
    ))

    sky_light: object = field(default_factory=lambda: AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=None,
    ))


    # sensors
    height_scanner: object = field(default_factory=lambda: RayCasterCfg(
        name="height_scanner",
        frame=ObjRef(type="body", name="torso_link", entity="robot"),
        pattern=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        ray_alignment="yaw",
        max_distance=30.0,
        debug_vis=False,
    ))

    camera: object = field(default_factory=lambda: NoisyGroupedRayCasterCameraCfg(
        name="camera",
        frame=ObjRef(type="body", name="torso_link", entity="robot"),
        pattern=patterns.PinholeCameraPatternCfg(
            width=int(480 / 10),
            height=int(270 / 10),
            fovy=58.0,
        ),
        mesh_prim_paths=["/World/ground/", "/World/envs/env_.*/Robot/.*"],
        aux_mesh_and_link_names={
            "torso_link_rev_1_0": None,
            "waist_yaw_link_rev_1_0": "waist_yaw_link",
            "waist_roll_link_rev_1_0": "waist_roll_link",
            "head_link": "head_link",
            "left_rubber_hand": "left_rubber_hand",
            "right_rubber_hand": "right_rubber_hand",
        },
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(
                0.04764571478 + 0.0039635 - 0.0042 * math.cos(math.radians(48)),
                0.015,
                0.46268178553 - 0.044 + 0.0042 * math.sin(math.radians(48)) + 0.016,
            ),
            rot=(
                math.cos(math.radians(0.5) / 2) * math.cos(math.radians(48) / 2),
                math.sin(math.radians(0.5) / 2),
                math.sin(math.radians(48) / 2),
                0.0,
            ),
            convention="world",
        ),
        attach_yaw_only=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            width=int(480 / 10),
            height=int(270 / 10),
            fovy=58.0,  # fovy
        ),
        data_types=["distance_to_image_plane"],
        noise_pipeline={
            # "depth_contour_noise": DepthContourNoiseCfg(
            #     contour_threshold=1.8,  # in [m]
            #     maxpool_kernel_size=1,
            # ),
            # "depth_artifact_noise": DepthArtifactNoiseCfg(),
            # "reflection_artifact_noise": DepthArtifactNoiseCfg(noise_value=30.0),
            # "stereo_noise": RangeBasedGaussianNoiseCfg(
            #     max_value=1.2,
            #     min_value=0.12,
            #     noise_std=0.02,
            # ),
            # "sky_artifact_noise": DepthSkyArtifactNoiseCfg(),
            # "gaussian_blur_noise": GaussianBlurNoiseCfg(
            #     kernel_size=3,
            #     sigma=0.5,
            # ),
            # "stereo_too_close_noise": StereoTooCloseNoiseCfg(),
            # These last two noise model will affect the processing on the onboard device.
            "normalize": DepthNormalizationCfg(
                depth_range=(0.0, 2.0),
                normalize=True,
            ),
            "crop_and_resize": CropAndResizeCfg(
                crop_region=(2, 2, 2, 2),
                resize_shape=(18, 32),
            ),
        },
        # data_histories={"distance_to_image_plane": 5},
        debug_vis=False,
        depth_clipping_behavior="max",  # clip to the maximum value
        min_distance=0.05,
    ))

    contact_forces: object = field(default_factory=lambda: ContactSensorCfg(
        name="contact_forces",
        primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        history_length=3,
        track_air_time=True,
    ))

    undesired_contact_forces: object = field(default_factory=lambda: ContactSensorCfg(
        name="undesired_contact_forces",
        primary=ContactMatch(
            mode="body",
            pattern=".*",
            entity="robot",
            exclude=(
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    ))


    def __post_init__(self):
        # Bridge Isaac Lab-style class attributes into mjlab entities dict / sensors tuple
        if self.robot is not None:
            self.entities["robot"] = self.robot
        if self.robot_reference is not None:
            self.entities["robot_reference"] = self.robot_reference
        sensor_list = []
        if self.contact_forces is not None:
            sensor_list.append(self.contact_forces)
        if self.undesired_contact_forces is not None:
            sensor_list.append(self.undesired_contact_forces)
        if self.height_scanner is not None:
            sensor_list.append(self.height_scanner)
        if self.camera is not None:
            sensor_list.append(self.camera)
        if self.motion_reference is not None:
            sensor_list.append(self.motion_reference)
        self.sensors = tuple(sensor_list)

        if self.motion_reference is None or self.motion_reference.reference_entity_name is None:
            if "robot_reference" in self.entities:
                del self.entities["robot_reference"]
            if hasattr(self, "robot_reference"):
                delattr(self, "robot_reference")


def make_perceptive_commands() -> dict[str, instinct_mdp.ShadowingCommandBaseCfg]:
    """Perceptive shadowing command configuration."""
    return {
        "position_ref_command": instinct_mdp.PositionRefCommandCfg(
            realtime_mode=True,
            current_state_command=False,
            anchor_frame="robot",
        ),
        "position_b_ref_command": instinct_mdp.PositionRefCommandCfg(
            realtime_mode=True,
            current_state_command=False,
            anchor_frame="reference",
        ),
        "rotation_ref_command": instinct_mdp.RotationRefCommandCfg(
            realtime_mode=True,
            current_state_command=False,
            in_base_frame=True,
            rotation_mode="tannorm",
        ),
        "joint_pos_ref_command": instinct_mdp.JointPosRefCommandCfg(current_state_command=False),
        "joint_vel_ref_command": instinct_mdp.JointVelRefCommandCfg(current_state_command=False),
    }



def make_actions() -> dict[str, mdp.JointPositionActionCfg]:
    """Action specifications for the MDP."""
    return {
        "joint_pos": mdp.JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
        ),
    }



def make_observations() -> dict[str, ObsGroupCfg]:
    """Observation specifications for the perceptive shadowing MDP."""

    # Policy observations
    policy_terms = {
        # Currently, just a dummy observation
        "joint_pos_ref": ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"}),
        "joint_vel_ref": ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"}),
        "position_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "position_b_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
        ),
        "rotation_ref": ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        ),

        # height_scan = ObsTermCfg(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=[-20.0, 20.0],
        # )
        "depth_image": ObsTermCfg(
            func=instinct_mdp.visualizable_image,
            # params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane"},
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane_noised"},
        ),

        # proprioception
        "projected_gravity": ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        # base_lin_vel = ObsTermCfg(func=mdp.base_lin_vel)
        "base_ang_vel": ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_pos": ObsTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_vel": ObsTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "last_action": ObsTermCfg(
            func=mdp.last_action,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
    }

    # Critic observations for BeyondMimic.
    critic_terms = {
        # BeyondMimic specific reference observations
        "joint_pos_ref": ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"}),
        "joint_vel_ref": ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"}),
        "position_ref": ObsTermCfg(func=mdp.generated_commands, params={"command_name": "position_ref_command"}),

        # proprioception
        "link_pos": ObsTermCfg(
            func=instinct_mdp.link_pos_b,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=MISSING, preserve_order=True)},
        ),
        "link_rot": ObsTermCfg(
            func=instinct_mdp.link_tannorm_b,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=MISSING, preserve_order=True)},
        ),

        "height_scan": ObsTermCfg(
            func=mdp.height_scan,
            params={"sensor_name": "height_scanner"},
            clip=[-20.0, 20.0],
        ),

        "base_lin_vel": ObsTermCfg(
            func=mdp.base_lin_vel,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "base_ang_vel": ObsTermCfg(
            func=mdp.base_ang_vel,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_pos": ObsTermCfg(
            func=mdp.joint_pos_rel,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_vel": ObsTermCfg(
            func=mdp.joint_vel_rel,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "last_action": ObsTermCfg(
            func=mdp.last_action,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
    }

    return {
        "policy": ObsGroupCfg(
            terms=policy_terms,
            enable_corruption=True,
            concatenate_terms=False,
        ),
        "critic": ObsGroupCfg(
            terms=critic_terms,
            enable_corruption=False,
            concatenate_terms=False,
        ),
    }



def make_rewards() -> dict[str, RewTermCfg]:
    """Reward specifications for the perceptive shadowing MDP."""
    return {
        "base_position_imitation_gauss": RewTermCfg(
            func=instinct_mdp.base_position_imitation_gauss,
            weight=0.5,
            params={
                "std": 0.3,
            },
        ),

        "base_rot_imitation_gauss": RewTermCfg(
            func=instinct_mdp.base_rot_imitation_gauss,
            weight=0.5,
            params={
                "std": 0.4,
                "difference_type": "axis_angle",
            },
        ),

        "link_pos_imitation_gauss": RewTermCfg(
            func=instinct_mdp.link_pos_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "in_base_frame": False,
                "in_relative_world_frame": True,
                "std": 0.3,
            },
        ),

        "link_rot_imitation_gauss": RewTermCfg(
            func=instinct_mdp.link_rot_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "in_base_frame": False,
                "in_relative_world_frame": True,
                "std": 0.4,
            },
        ),

        "link_lin_vel_imitation_gauss": RewTermCfg(
            func=instinct_mdp.link_lin_vel_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "std": 1.0,
            },
        ),

        "link_ang_vel_imitation_gauss": RewTermCfg(
            func=instinct_mdp.link_ang_vel_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "std": 3.14,
            },
        ),

        "action_rate_l2": RewTermCfg(func=mdp.action_rate_l2, weight=-0.1),

        "joint_limit": RewTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        ),

        "undesired_contacts": RewTermCfg(
            func=instinct_mdp.undesired_contacts,
            weight=-0.1,
            params={
                "sensor_name": "undesired_contact_forces",
                "threshold": 1.0,
            },
        ),

        "applied_torque_limits_by_ratio": RewTermCfg(
            func=instinct_mdp.applied_torque_limits_by_ratio,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*ankle.*",
                        ".*wrist.*",
                    ],
                )
            },
        ),
    }



def make_events() -> dict[str, EventTermCfg]:
    """Event specifications for the perceptive shadowing MDP."""
    return {
        # domain rand
        "physics_material": EventTermCfg(
            func=mdp.randomize_field,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                "field": "geom_friction",
                "ranges": {
                    0: (1.25, 2.0),
                    1: (1.2, 1.8),
                    2: (0.0, 0.5),
                },
                "operation": "abs",
                "distribution": "uniform",
            },
        ),

        "add_joint_default_pos": EventTermCfg(
            func=instinct_mdp.randomize_default_joint_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "offset_distribution_params": (-0.01, 0.01),
                "operation": "add",
                "distribution": "uniform",
            },
        ),

        "base_com": EventTermCfg(
            func=instinct_mdp.randomize_rigid_body_coms,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "coms_x_distribution_params": (-0.025, 0.025),
                "coms_y_distribution_params": (-0.05, 0.05),
                "coms_z_distribution_params": (-0.05, 0.05),
                "distribution": "uniform",
            },
        ),

        "randomize_ray_offsets": EventTermCfg(
            func=instinct_mdp.randomize_ray_offsets,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("camera"),
                "offset_pose_ranges": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (-0.01, 0.01),
                    "roll": (-math.radians(2), math.radians(2)),
                    "pitch": (-math.radians(10), math.radians(10)),
                    "yaw": (-math.radians(2), math.radians(2)),
                },
                "distribution": "uniform",
            },
        ),

        "randomize_actuator_gains": EventTermCfg(
            func=mdp.randomize_pd_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.8, 1.2),
                "kd_range": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        ),

        "randomize_rigid_body_mass": EventTermCfg(
            func=mdp.randomize_field,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "torso_link",
                        "left_ankle.*",
                        "right_ankle.*",
                        "left_wrist.*",
                        "right_wrist.*",
                    ],
                ),
                "field": "body_mass",
                "ranges": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        ),

        "match_motion_ref_with_scene": EventTermCfg(
            func=instinct_mdp.match_motion_ref_with_scene,
            mode="startup",
            params={
                "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            },
        ),

        "reset_robot": EventTermCfg(
            func=instinct_mdp.reset_robot_state_by_reference,
            mode="reset",
            params={
                "motion_ref_cfg": SceneEntityCfg("motion_reference"),
                "asset_cfg": SceneEntityCfg("robot"),
                # reset with position offset to put the robot_reference in scene.
                "position_offset": [0.0, 0.0, 0.0],
                "dof_vel_ratio": 1.0,
                "base_lin_vel_ratio": 1.0,
                "base_ang_vel_ratio": 1.0,
                # Pose randomization (+-5cm position, +-6degrees rotation)
                "randomize_pose_range": {
                    "x": (-0.15, 0.15),
                    "y": (-0.15, 0.15),
                    "z": (-0.0, 0.0),
                },
                # Velocity randomization (+-0.1 m/s linear, +-0.1 rad/s angular)
                "randomize_velocity_range": {},
                # Joint position randomization (+-0.1 rad)
                "randomize_joint_pos_range": (-0.1, 0.1),
            },
        ),

        "bin_fail_counter_smoothing": EventTermCfg(
            func=instinct_mdp.beyondmimic_bin_fail_counter_smoothing,
            mode="interval",
            interval_range_s=(0.02, 0.02),  # every environment step
            params={
                "curriculum_name": "beyond_adaptive_sampling",
            },
        ),

        # "push_robot": EventTermCfg(
        #     func=mdp.push_by_setting_velocity,
        #     mode="interval",
        #     interval_range_s=(0.5, 2.0),
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "velocity_range": {
        #             "x": (-0.5, 0.5),
        #             "y": (-0.5, 0.5),
        #             "z": (-0.2, 0.2),
        #             "roll": (-0.52, 0.52),
        #             "pitch": (-0.52, 0.52),
        #             "yaw": (-0.78, 0.78),
        #         },
        #     },
        # ),
    }


def make_curriculum() -> dict[str, CurriculumTermCfg]:
    """Curriculum specifications for the perceptive shadowing MDP."""
    return {
        "beyond_adaptive_sampling": CurriculumTermCfg(  # type: ignore
            func=instinct_mdp.BeyondConcatMotionAdaptiveWeighting,
        ),
    }



def make_terminations() -> dict[str, DoneTermCfg]:
    """Termination specifications for the perceptive shadowing MDP."""
    return {
        "time_out": DoneTermCfg(func=mdp.time_out, time_out=True),

        "illegal_reset_contact": DoneTermCfg(
            func=instinct_mdp.illegal_reset_contact,
            time_out=True,
            params={
                "sensor_name": "undesired_contact_forces",
                "threshold": 500,
                "episode_length_threshold": 2,
            },
        ),

        "base_pos_too_far": DoneTermCfg(
            func=instinct_mdp.pos_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "distance_threshold": 0.25,
                "check_at_keyframe_threshold": -1,
                "print_reason": False,
                "height_only": True,
            },
        ),

        "base_pg_too_far": DoneTermCfg(
            func=instinct_mdp.projected_gravity_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "projected_gravity_threshold": 0.8,
                "check_at_keyframe_threshold": -1,
                "z_only": False,
                "print_reason": False,
            },
        ),

        "link_pos_too_far": DoneTermCfg(
            func=instinct_mdp.link_pos_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg(
                    "motion_reference",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                    ],
                    preserve_order=True,
                ),
                "distance_threshold": 0.25,
                "in_base_frame": False,
                "check_at_keyframe_threshold": -1,
                "height_only": True,
                "print_reason": False,
            },
        ),

        "dataset_exhausted": DoneTermCfg(
            func=instinct_mdp.dataset_exhausted,
            time_out=True,
            params={
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "print_reason": False,
            },
        ),

        "out_of_border": DoneTermCfg(
            func=instinct_mdp.terrain_out_of_bounds,
            time_out=True,
            params={"asset_cfg": SceneEntityCfg("robot"), "print_reason": False, "distance_buffer": 0.1},
        ),
    }



def make_monitors() -> dict[str, MonitorTermCfg]:
    """Monitor specifications for the perceptive shadowing MDP."""
    return {
        "dataset": MonitorTermCfg(
            func=MotionReferenceMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("motion_reference"),
                sample_stat_interval=500,
                top_n_samples=5,
            ),
        ),

        "shadowing_position": MonitorTermCfg(
            func=ShadowingPositionMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                in_base_frame=True,
                check_at_keyframe_threshold=0.03,
            ),
        ),

        "shadowing_rotation": MonitorTermCfg(
            func=ShadowingRotationMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                masking=True,
            ),
        ),

        "shadowing_joint_pos": MonitorTermCfg(
            func=ShadowingJointPosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                masking=True,
            ),
        ),

        "shadowing_joint_vel": MonitorTermCfg(
            func=ShadowingJointVelMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                masking=True,
            ),
        ),

        "shadowing_link_pos_b": MonitorTermCfg(
            func=ShadowingLinkPosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                in_base_frame=True,
                masking=True,
            ),
        ),

        "shadowing_link_pos_w": MonitorTermCfg(
            func=ShadowingLinkPosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                in_base_frame=False,
                masking=True,
            ),
        ),
    }



@dataclass(kw_only=True)
class PerceptiveShadowingEnvCfg(InstinctLabRLEnvCfg):
    scene: PerceptiveShadowingSceneCfg = field(default_factory=lambda: PerceptiveShadowingSceneCfg())
    commands: dict = field(default_factory=make_perceptive_commands)
    actions: dict = field(default_factory=make_actions)
    observations: dict = field(default_factory=make_observations)
    rewards: dict = field(default_factory=make_rewards)
    events: dict = field(default_factory=make_events)
    curriculum: dict = field(default_factory=make_curriculum)
    terminations: dict = field(default_factory=make_terminations)
    monitors: dict = field(default_factory=make_monitors)

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.mujoco.timestep = 1.0 / 50.0 / self.decimation
        # Contact budget for perceptive shadowing scene (avoid broadphase overflow warnings in play).
        self.sim.nconmax = 100
        self.sim.njmax = 300

        # All managers are already dicts, no conversion needed!
