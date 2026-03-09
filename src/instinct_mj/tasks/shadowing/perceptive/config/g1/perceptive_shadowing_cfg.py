import os
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial

import mjlab.envs.mdp as mdp
import mjlab.sim as sim_utils
import numpy as np
import yaml
from mjlab.managers import EventTermCfg, SceneEntityCfg
from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mj.envs.mdp as instinct_mdp
import instinct_mj.tasks.shadowing.mdp as shadowing_mdp
import instinct_mj.tasks.shadowing.perceptive.perceptive_env_cfg as perceptual_cfg
from instinct_mj.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    G1_MJCF_PATH,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuator_cfgs,
    beyondmimic_g1_29dof_delayed_actuator_cfgs,
)
from instinct_mj.monitors import ActuatorMonitorTerm, MonitorTermCfg, ShadowingBasePosMonitorTerm
from instinct_mj.motion_reference import MotionReferenceManagerCfg
from instinct_mj.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinct_mj.motion_reference.motion_files.terrain_motion_cfg import TerrainMotionCfg as TerrainMotionCfgBase
from instinct_mj.motion_reference.utils import motion_interpolate_bilinear

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

MOTION_FOLDER = "~/Xyk/Datasets/20251116_50cm_kneeClimbStep1"


@dataclass(kw_only=True)
class TerrainMotionCfg(TerrainMotionCfgBase):
    path: object = field(default_factory=lambda: os.path.expanduser(MOTION_FOLDER))

    metadata_yaml: object = field(default_factory=lambda: os.path.expanduser(f"{MOTION_FOLDER}/metadata.yaml"))

    max_origins_per_motion: int = 49

    ensure_link_below_zero_ground: bool = False

    motion_start_from_middle_range: list = field(default_factory=lambda: [0.0, 0.0])

    motion_start_height_offset: float = 0.0

    motion_bin_length_s: float = 1.0

    buffer_device: str = "output_device"

    motion_interpolate_func: object = field(default_factory=lambda: motion_interpolate_bilinear)

    velocity_estimation_method: str = "frontbackward"

    env_starting_stub_sampling_strategy: str = "concat_motion_bins"


@dataclass(kw_only=True)
class AMASSMotionCfg(AmassMotionCfgBase):
    path: object = field(default_factory=lambda: os.path.expanduser(MOTION_FOLDER))

    filtered_motion_selection_filepath: object | None = None

    ensure_link_below_zero_ground: bool = False

    motion_start_from_middle_range: list = field(default_factory=lambda: [0.0, 0.0])

    motion_start_height_offset: float = 0.0

    motion_bin_length_s: float = 1.0

    buffer_device: str = "output_device"

    motion_interpolate_func: object = field(default_factory=lambda: motion_interpolate_bilinear)

    velocity_estimation_method: str = "frontbackward"

    env_starting_stub_sampling_strategy: str = "concat_motion_bins"


motion_reference_cfg = MotionReferenceManagerCfg(
    name="motion_reference",
    entity_name="robot",
    robot_model_path=G1_MJCF_PATH,
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    frame_interval_s=0.1,
    update_period=0.02,
    num_frames=10,
    data_start_from="current_time",
    # set the robot_reference directly at where they are in the scene
    # DO NOT FORGET to change this when in actual training
    visualizing_robot_offset=(2.0, 0.0, 0.0),
    visualizing_robot_from="reference_frame",
    visualizing_marker_types=["relative_links", "links"],
    motion_buffers={
        "TerrainMotion": TerrainMotionCfg(),
    },
    mp_split_method="None",
)
motion_reference_cfg_play = deepcopy(motion_reference_cfg)
motion_reference_cfg_play.debug_vis = True
motion_reference_cfg_play.reference_entity_name = "robot_reference"


@dataclass(kw_only=True)
class G1PerceptiveShadowingEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = field(
        default_factory=lambda: perceptual_cfg.PerceptiveShadowingSceneCfg(
            num_envs=3072,
            entities={"robot": deepcopy(G1_CFG)},
            sensors=perceptual_cfg.make_perceptive_scene_sensors(
                motion_reference=deepcopy(motion_reference_cfg),
            ),
        )
    )

    def __post_init__(self):
        super().__post_init__()

        robot_cfg = self.scene.entities["robot"]
        motion_reference_cfg = next(
            sensor_cfg for sensor_cfg in self.scene.sensors if sensor_cfg.name == "motion_reference"
        )
        camera_cfg = next(sensor_cfg for sensor_cfg in self.scene.sensors if sensor_cfg.name == "camera")

        robot_cfg.articulation.actuators = beyondmimic_g1_29dof_actuator_cfgs
        # self.scene.robot.spawn.rigid_props.max_depenetration_velocity = 0.3
        self.actions["joint_pos"].scale = beyondmimic_action_scale
        # Set contact/constraint capacities explicitly for perceptive shadowing.
        self.sim.njmax = 700
        self.sim.nconmax = 128
        self.sim.mujoco.jacobian = "sparse"
        self.sim.mujoco.ccd_iterations = 128
        self.sim.mujoco.multiccd = False

        MOTION_NAME = list(motion_reference_cfg.motion_buffers.keys())[0]
        motion_buffer = motion_reference_cfg.motion_buffers[MOTION_NAME]
        motion_buffer.metadata_yaml = os.path.join(motion_buffer.path, "metadata.yaml")
        PLANE_TERRAIN = False
        if PLANE_TERRAIN:
            motion_reference_cfg.motion_buffers.pop(MOTION_NAME)
            motion_reference_cfg.motion_buffers["AMASSMotion"] = AMASSMotionCfg()
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
        else:
            terrain_cfg = self.scene.terrain.terrain_generator.sub_terrains["motion_matched"]
            terrain_cfg.path = motion_buffer.path
            terrain_cfg.metadata_yaml = motion_buffer.metadata_yaml
        active_motion_name = list(motion_reference_cfg.motion_buffers.keys())[0]
        active_motion_buffer = motion_reference_cfg.motion_buffers[active_motion_name]

        # match key links for observation terms
        self.observations["critic"].terms["link_pos"].params[
            "asset_cfg"
        ].body_names = motion_reference_cfg.link_of_interests
        self.observations["critic"].terms["link_rot"].params[
            "asset_cfg"
        ].body_names = motion_reference_cfg.link_of_interests

        self.run_name = "g1Perceptive" + "".join(
            [
                (
                    "_concatMotionBins"
                    if active_motion_buffer.env_starting_stub_sampling_strategy == "concat_motion_bins"
                    else "_independentMotionBins"
                ),
            ]
        )


@dataclass(kw_only=True)
class G1PerceptiveShadowingEnvCfg_PLAY(G1PerceptiveShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = field(
        default_factory=lambda: perceptual_cfg.PerceptiveShadowingSceneCfg(
            num_envs=1,
            env_spacing=2.5,
            entities={
                "robot": deepcopy(G1_CFG),
                "robot_reference": deepcopy(G1_CFG),
            },
            sensors=perceptual_cfg.make_perceptive_scene_sensors(
                motion_reference=deepcopy(motion_reference_cfg_play),
            ),
        )
    )

    viewer: ViewerConfig = field(
        default_factory=lambda: ViewerConfig(
            lookat=(0.0, 0.0, 0.0),
            distance=2.1213,
            elevation=45.0,
            azimuth=0.0,
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
        )
    )

    def __post_init__(self):
        super().__post_init__()

        motion_reference_cfg = next(
            sensor_cfg for sensor_cfg in self.scene.sensors if sensor_cfg.name == "motion_reference"
        )
        camera_cfg = next(sensor_cfg for sensor_cfg in self.scene.sensors if sensor_cfg.name == "camera")

        # deactivate adaptive sampling and start from the 0.0s of the motion
        self.curriculum["beyond_adaptive_sampling"] = None
        self.events["bin_fail_counter_smoothing"] = None
        MOTION_NAME = list(motion_reference_cfg.motion_buffers.keys())[0]
        motion_reference_cfg.motion_buffers[MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        motion_reference_cfg.motion_buffers[MOTION_NAME].motion_bin_length_s = None
        motion_reference_cfg.motion_buffers[MOTION_NAME].env_starting_stub_sampling_strategy = "independent"
        # self.scene.motion_reference.motion_buffers[MOTION_NAME].path = (
        #     "/localhdd/Datasets/NoKov-Marslab-Motions-instinctnpz/20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall"
        # )
        motion_buffer = motion_reference_cfg.motion_buffers[MOTION_NAME]
        motion_buffer.metadata_yaml = os.path.join(motion_buffer.path, "metadata.yaml")
        if self.scene.terrain.terrain_type == "hacked_generator":
            terrain_cfg = self.scene.terrain.terrain_generator.sub_terrains["motion_matched"]
            terrain_cfg.path = motion_buffer.path
            terrain_cfg.metadata_yaml = motion_buffer.metadata_yaml
            self.scene.terrain.terrain_generator.num_rows = 6
            self.scene.terrain.terrain_generator.num_cols = 6
        # self.scene.motion_reference.motion_buffers.pop(MOTION_NAME)
        # self.scene.motion_reference.motion_buffers["AMASSMotion"] = AMASSMotionCfg()
        # self.scene.motion_reference.motion_buffers["AMASSMotion"].motion_start_from_middle_range = [0.0, 0.0]
        # self.scene.motion_reference.motion_buffers["AMASSMotion"].motion_bin_length_s = None
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        camera_cfg.debug_vis = True
        self.scene.terrain.collision_debug_vis = False
        self.observations["policy"].terms["depth_image"].params["debug_vis"] = True

        # change reset robot event with more pitch_down randomization (since the robot is facing -y axis)
        # self.events.reset_robot.params["randomize_pose_range"]["roll"] = (0.0, 0.6)

        # remove some terimation terms
        self.terminations["base_pos_too_far"] = None
        self.terminations["base_pg_too_far"] = None
        self.terminations["link_pos_too_far"] = None

        # put the reference in scene and move the robot elsewhere and visualize the reference
        # self.events.reset_robot.params["position_offset"] = [0.0, 1.0, 2.0]
        # self.scene.motion_reference.visualizing_robot_offset = (0.0, 0.0, 0.0)
        # self.viewer.entity_name = "robot_reference"

        # remove some randomizations
        self.events["add_joint_default_pos"] = None
        self.events["base_com"] = None
        self.events["physics_material"] = None
        self.events["push_robot"] = None
        self.events["reset_robot"].params["randomize_pose_range"]["x"] = [0.0] * 2  # (+-0.6)
        self.events["reset_robot"].params["randomize_pose_range"]["y"] = [0.0] * 2  # (+-0.6)
        self.events["reset_robot"].params["randomize_pose_range"]["z"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["roll"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["pitch"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_pose_range"]["yaw"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["x"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["y"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["z"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["roll"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["pitch"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_velocity_range"]["yaw"] = (0.0, 0.0)
        self.events["reset_robot"].params["randomize_joint_pos_range"] = (0.0, 0.0)

        # add some additional monitor terms
        self.monitors["shadowing_position_stats"] = MonitorTermCfg(
            func=ShadowingBasePosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
            ),
        )
        self.monitors["right_ankle_pitch_actuator"] = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_ankle_pitch.*"),
            ),
        )
        self.monitors["left_ankle_pitch_actuator"] = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_ankle_pitch.*"),
            ),
        )
        self.monitors["right_knee_actuator"] = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_knee.*"),
            ),
        )
        self.monitors["left_knee_actuator"] = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_knee.*"),
            ),
        )

        # add another box to the scene (to test visual generalization)
        # self.scene.distractor = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cube",
        #     spawn=sim_utils.MeshCuboidCfg(
        #         size=(4.8, 0.6, 0.5),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.material_cfg,
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.35, 0.25)),
        # )
        # camera_cfg.mesh_prim_paths.append("/cube")
        # self.scene.distractor = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cube",
        #     spawn=sim_utils.MeshCuboidCfg(
        #         size=(4.8, 2.6, 0.5),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.material_cfg,
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -2.35, 0.25)),
        # )
        # camera_cfg.mesh_prim_paths.append("/cube")
        # self.scene.distractor = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cube",
        #     spawn=sim_utils.MeshCuboidCfg(
        #         size=(0.1, 4.6, 1.8),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.material_cfg,
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.8, -1.35, 0.25)),
        # )
        # camera_cfg.mesh_prim_paths.append("/cube")

        # self.scene.distractor1 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cone1",
        #     spawn=sim_utils.MeshConeCfg(
        #         radius=0.22,
        #         height=0.55,
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.material_cfg,
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.0, 0.3)),
        # )
        # camera_cfg.mesh_prim_paths.append("/cone1")

        # see the reference robot
        # camera_cfg.mesh_prim_paths.append("/robot_reference")
