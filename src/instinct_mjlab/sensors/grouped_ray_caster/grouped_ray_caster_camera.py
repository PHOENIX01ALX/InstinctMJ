from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
import warp as wp

from mjlab.utils.lab_api import math as math_utils

from .grouped_ray_caster import GroupedRayCaster

if TYPE_CHECKING:
    from .grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass
class CameraData:
    """Data container for the camera sensor."""

    ##
    # Frame state.
    ##

    pos_w: torch.Tensor | None = None
    """Position of the sensor origin in world frame, following ROS convention.

    Shape is (N, 3) where N is the number of sensors.
    """

    quat_w_world: torch.Tensor | None = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in world frame, following the world coordinate frame

    .. note::
        World frame convention follows the camera aligned with forward axis +X and up axis +Z.

    Shape is (N, 4) where N is the number of sensors.
    """

    ##
    # Camera data
    ##

    image_shape: tuple[int, int] | None = None
    """A tuple containing (height, width) of the camera sensor."""

    intrinsic_matrices: torch.Tensor | None = None
    """The intrinsic matrices for the camera.

    Shape is (N, 3, 3) where N is the number of sensors.
    """

    output: dict[str, torch.Tensor] | None = None
    """The retrieved sensor data with sensor types as key."""

    info: list[dict[str, Any]] | None = None
    """The retrieved sensor info with sensor types as key."""

    ##
    # Additional Frame orientation conventions
    ##

    @property
    def quat_w_ros(self) -> torch.Tensor:
        """Quaternion orientation `(w, x, y, z)` of the sensor origin in the world frame, following ROS convention."""
        assert self.quat_w_world is not None
        return math_utils.convert_camera_frame_orientation_convention(self.quat_w_world, origin="world", target="ros")

    @property
    def quat_w_opengl(self) -> torch.Tensor:
        """Quaternion orientation `(w, x, y, z)` of the sensor origin in the world frame, following OpenGL convention."""
        assert self.quat_w_world is not None
        return math_utils.convert_camera_frame_orientation_convention(
            self.quat_w_world, origin="world", target="opengl"
        )


class GroupedRayCasterCamera(GroupedRayCaster):
    """Grouped ray-cast camera sensor."""

    cfg: GroupedRayCasterCameraCfg
    """The configuration parameters."""

    UNSUPPORTED_TYPES: ClassVar[set[str]] = {
        "rgb",
        "instance_id_segmentation",
        "instance_id_segmentation_fast",
        "instance_segmentation",
        "instance_segmentation_fast",
        "semantic_segmentation",
        "skeleton_data",
        "motion_vectors",
        "bounding_box_2d_tight",
        "bounding_box_2d_tight_fast",
        "bounding_box_2d_loose",
        "bounding_box_2d_loose_fast",
        "bounding_box_3d",
        "bounding_box_3d_fast",
    }
    """A set of sensor types that are not supported by the ray-caster camera."""

    def __init__(self, cfg: GroupedRayCasterCameraCfg):
        """Initializes the camera object.

        Args:
            cfg: The configuration parameters.

        Raises:
            ValueError: If the provided data types are not supported by the grouped-ray-caster camera.
        """
        # perform check on supported data types
        self._check_supported_data_types(cfg)
        # initialize base class
        super().__init__(cfg)
        # create empty variables for storing output data
        self._camera_data = CameraData()
        self._frame = torch.empty(0, dtype=torch.long)

        self._offset_quat: torch.Tensor | None = None
        self._offset_pos: torch.Tensor | None = None
        self.ray_hits_w: torch.Tensor | None = None
        self._focal_length: float = 1.0

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Grouped-Ray-Caster-Camera @ '{self.cfg.name}': \n"
            f"\tupdate period (s)    : n/a (manager driven)\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}\n"
            f"\timage shape          : {self.image_shape}"
        )

    """
    Properties
    """

    @property
    def image_shape(self) -> tuple[int, int]:
        """A tuple containing (height, width) of the camera sensor."""
        return (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width)

    @property
    def frame(self) -> torch.Tensor:
        """Frame number when the measurement took place."""
        return self._frame

    """
    Operations.
    NOTE: Since RayCasterCamera is a direct subclass of RayCaster, GroupedRayCasterCamera has to copy some of the code
    from RayCasterCamera. (Code duplication is not ideal, shall be optimized in the future.)
    """

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        # Create all indices buffer
        self._ALL_INDICES = torch.arange(self._view.count, device=device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._view.count, device=device, dtype=torch.long)
        # create buffers
        self._create_buffers()
        # compute intrinsic matrices
        self._compute_intrinsic_matrices()
        # compute ray stars and directions
        assert self._local_offsets is not None and self._local_directions is not None
        self.ray_starts = self._local_offsets.unsqueeze(0).repeat(self._view.count, 1, 1).clone()
        self.ray_directions = self._local_directions.unsqueeze(0).repeat(self._view.count, 1, 1).clone()
        self._num_rays = self.ray_directions.shape[1]
        # create buffer to store ray hits
        self.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=device)
        # set offsets
        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=device), origin=self.cfg.offset.convention, target="world"
        )
        self._offset_quat = quat_w.repeat(self._view.count, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=device, dtype=torch.float32).repeat(
            self._view.count, 1
        )
        self._create_ray_collision_groups()

    def set_intrinsic_matrices(
        self, matrices: torch.Tensor, focal_length: float = 1.0, env_ids: Sequence[int] | None = None
    ):
        """Set the intrinsic matrix of the camera.

        Args:
            matrices: The intrinsic matrices for the camera. Shape is (N, 3, 3).
            focal_length: Focal length to use when computing aperture values (in cm). Defaults to 1.0.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # save new intrinsic matrices and focal length
        assert self._camera_data.intrinsic_matrices is not None
        self._camera_data.intrinsic_matrices[env_ids] = matrices.to(self.device)
        self._focal_length = focal_length
        # recompute ray directions
        assert self.ray_starts is not None and self.ray_directions is not None
        # NOTE(migration): Original IsaacLab called self.cfg.pattern_cfg.func() here.
        # In mjlab, the native pinhole pattern exposes generate_rays(mj_model, device), so
        # we regenerate rays from intrinsic matrices with IsaacLab-equivalent pinhole math.
        pattern_func = getattr(self.cfg.pattern_cfg, "func", None)
        if callable(pattern_func):
            ray_starts, ray_directions = pattern_func(
                self.cfg.pattern_cfg, self._camera_data.intrinsic_matrices[env_ids], self.device
            )
        else:
            ray_starts, ray_directions = self._compute_pinhole_rays_from_intrinsics(
                self._camera_data.intrinsic_matrices[env_ids]
            )
        self.ray_starts[env_ids] = ray_starts
        self.ray_directions[env_ids] = ray_directions

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = self._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # reset the data
        # note: this recomputation is useful if one performs events such as randomizations on the camera poses.
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        assert self._camera_data.pos_w is not None and self._camera_data.quat_w_world is not None
        self._camera_data.pos_w[env_ids] = pos_w
        self._camera_data.quat_w_world[env_ids] = quat_w
        # Reset the frame count
        self._frame[env_ids] = 0

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
        convention: Literal["opengl", "ros", "world"] = "ros",
    ):
        """Set the pose of the camera w.r.t. the world frame using specified convention.

        Since different fields use different conventions for camera orientations, the method allows users to
        set the camera poses in the specified convention. Possible conventions are:

        - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
        - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
        - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

        See :meth:`isaaclab.utils.maths.convert_camera_frame_orientation_convention` for more details
        on the conventions.

        Args:
            positions: The cartesian coordinates (in meters). Shape is (N, 3).
                Defaults to None, in which case the camera position in not changed.
            orientations: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
                Defaults to None, in which case the camera orientation in not changed.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
            convention: The convention in which the poses are fed. Defaults to "ros".

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # get current positions
        pos_w, quat_w = self._compute_view_world_poses(env_ids)
        if positions is not None:
            # transform to camera frame
            pos_offset_world_frame = positions.to(self.device) - pos_w
            assert self._offset_pos is not None
            self._offset_pos[env_ids] = math_utils.quat_apply(math_utils.quat_inv(quat_w), pos_offset_world_frame)
        if orientations is not None:
            # convert rotation matrix from input convention to world
            quat_w_set = math_utils.convert_camera_frame_orientation_convention(
                orientations.to(self.device), origin=convention, target="world"
            )
            assert self._offset_quat is not None
            self._offset_quat[env_ids] = math_utils.quat_mul(math_utils.quat_inv(quat_w), quat_w_set)

        # update the data
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        assert self._camera_data.pos_w is not None and self._camera_data.quat_w_world is not None
        self._camera_data.pos_w[env_ids] = pos_w
        self._camera_data.quat_w_world[env_ids] = quat_w

    def set_world_poses_from_view(
        self, eyes: torch.Tensor, targets: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Set the poses of the camera from the eye position and look-at target position.

        Args:
            eyes: The positions of the camera's eye. Shape is N, 3).
            targets: The target locations to look at. Shape is (N, 3).
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            NotImplementedError: If the stage up-axis is not "Y" or "Z".

        Note:
            In MuJoCo (mjlab), the up-axis is always "Z", so the NotImplementedError will not be raised.
        """
        # camera position and rotation in opengl convention
        orientations = math_utils.quat_from_matrix(
            math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis="Z", device=self.device)
        )
        self.set_world_poses(eyes, orientations, env_ids, convention="opengl")

    """
    Implementation.
    """

    def prepare_rays(self) -> None:
        """PRE-GRAPH: Prepare grouped camera rays from camera world poses."""
        assert self._ray_pnt is not None and self._ray_vec is not None
        assert self.ray_starts is not None and self.ray_directions is not None

        env_ids = self._ALL_INDICES
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        assert self._camera_data.pos_w is not None and self._camera_data.quat_w_world is not None
        self._camera_data.pos_w[env_ids] = pos_w
        self._camera_data.quat_w_world[env_ids] = quat_w
        self._frame += 1

        quat_w_expanded = quat_w.unsqueeze(1).expand(-1, self.num_rays, -1).reshape(-1, 4)
        ray_starts_flat = self.ray_starts.reshape(-1, 3)
        ray_dirs_flat = self.ray_directions.reshape(-1, 3)
        ray_starts_w = math_utils.quat_apply(quat_w_expanded, ray_starts_flat).view(self._view.count, self.num_rays, 3)
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = math_utils.quat_apply(quat_w_expanded, ray_dirs_flat).view(
            self._view.count, self.num_rays, 3
        )

        pnt_torch = wp.to_torch(self._ray_pnt).view(self._view.count, self._num_rays, 3)
        vec_torch = wp.to_torch(self._ray_vec).view(self._view.count, self._num_rays, 3)
        pnt_torch.copy_(ray_starts_w)
        vec_torch.copy_(ray_directions_w)

        self._cached_world_origins = ray_starts_w
        self._cached_world_rays = ray_directions_w
        self._cached_frame_pos = pos_w
        self._cached_frame_mat = math_utils.matrix_from_quat(quat_w)

    def postprocess_rays(self) -> None:
        super().postprocess_rays()
        if self.ray_hits_w is None:
            return
        assert self._cached_world_rays is not None
        assert self._distances is not None
        assert self._normals_w is not None
        assert self._hit_pos_w is not None
        assert self._camera_data.output is not None

        self.ray_hits_w.copy_(self._hit_pos_w)
        ray_directions_w = self._cached_world_rays

        assert self._camera_data.quat_w_world is not None
        quat_w = self._camera_data.quat_w_world
        ray_depth_camera = self._distances.clone()
        ray_depth_camera[ray_depth_camera < 0] = float("inf")
        ray_depth_image = self._distances.clone()
        ray_depth_image[ray_depth_image < 0] = float("nan")

        # update output buffers
        if "distance_to_image_plane" in self.cfg.data_types:
            # note: data is in camera frame so we only take the first component (x-axis of camera frame)
            quat_inv = math_utils.quat_inv(quat_w).unsqueeze(1).expand(-1, self.num_rays, -1).reshape(-1, 4)
            image_ray_vec = (ray_depth_image[:, :, None] * ray_directions_w).reshape(-1, 3)
            distance_to_image_plane = math_utils.quat_apply(quat_inv, image_ray_vec).view(
                self._view.count, self.num_rays, 3
            )[:, :, 0]
            # apply the maximum distance after the transformation
            if self.cfg.depth_clipping_behavior == "max":
                distance_to_image_plane = torch.clip(distance_to_image_plane, max=self.cfg.max_distance)
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = self.cfg.max_distance
            elif self.cfg.depth_clipping_behavior == "zero":
                distance_to_image_plane[distance_to_image_plane > self.cfg.max_distance] = 0.0
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = 0.0
            self._camera_data.output["distance_to_image_plane"] = distance_to_image_plane.view(
                -1, *self.image_shape, 1
            )

        if "distance_to_camera" in self.cfg.data_types:
            if self.cfg.depth_clipping_behavior == "max":
                ray_depth_camera = torch.clip(ray_depth_camera, max=self.cfg.max_distance)
            elif self.cfg.depth_clipping_behavior == "zero":
                ray_depth_camera[ray_depth_camera > self.cfg.max_distance] = 0.0
            self._camera_data.output["distance_to_camera"] = ray_depth_camera.view(-1, *self.image_shape, 1)

        if "normals" in self.cfg.data_types:
            self._camera_data.output["normals"] = self._normals_w.view(-1, *self.image_shape, 3)

    def debug_vis(self, visualizer: "DebugVisualizer") -> None:
        """Debug visualization for the grouped ray-caster camera.

        Uses mjlab DebugVisualizer native API.
        """
        if not self.cfg.debug_vis:
            return
        if self.ray_hits_w is None:
            return
        assert self._camera_data.pos_w is not None and self._camera_data.quat_w_world is not None

        env_ids = list(visualizer.get_env_indices(self._view.count))
        if not env_ids:
            return

        meansize = max(float(visualizer.meansize), 1e-6)
        point_radius = 0.003 * meansize
        frame_scale = 0.15 * meansize
        frame_axis_radius = 0.01 * meansize

        viz_points = self.ray_hits_w[env_ids].reshape(-1, 3)
        viz_points = viz_points[torch.isfinite(viz_points).all(dim=1)]
        max_points = 256
        if viz_points.shape[0] > max_points:
            stride = max(1, viz_points.shape[0] // max_points)
            viz_points = viz_points[::stride]

        for point in viz_points:
            visualizer.add_sphere(
                center=point,
                radius=point_radius,
                color=(1.0, 0.1, 0.1, 0.6),
            )

        camera_pos = self._camera_data.pos_w[env_ids]
        camera_quat = self._camera_data.quat_w_world[env_ids]
        camera_rot = math_utils.matrix_from_quat(camera_quat)
        for idx in range(len(env_ids)):
            visualizer.add_frame(
                position=camera_pos[idx],
                rotation_matrix=camera_rot[idx],
                scale=frame_scale,
                axis_radius=frame_axis_radius,
                alpha=0.9,
            )

    """
    Private Helpers
    """

    def _check_supported_data_types(self, cfg: GroupedRayCasterCameraCfg):
        """Checks if the data types are supported by the grouped-ray-caster camera."""
        # check if there is any intersection in unsupported types
        # reason: we cannot obtain this data from simplified warp-based ray caster
        common_elements = set(cfg.data_types) & GroupedRayCasterCamera.UNSUPPORTED_TYPES
        if common_elements:
            raise ValueError(
                f"GroupedRayCasterCamera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types cannot be obtained in a fast way using ''warp''."
                "\n\tHint: If you need to work with these sensor types, we recommend using the USD camera"
                " interface from the isaaclab.sensors.camera module."
            )

    def _create_buffers(self):
        """Create buffers for storing data."""
        # prepare drift
        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        # create the data object
        # -- pose of the cameras
        self._camera_data.pos_w = torch.zeros((self._view.count, 3), device=self.device)
        self._camera_data.quat_w_world = torch.zeros((self._view.count, 4), device=self.device)
        # -- intrinsic matrix
        self._camera_data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self.device)
        self._camera_data.intrinsic_matrices[:, 2, 2] = 1.0
        self._camera_data.image_shape = self.image_shape
        # -- output data
        # create the buffers to store the annotator data.
        self._camera_data.output = {}
        self._camera_data.info = [{name: None for name in self.cfg.data_types}] * self._view.count
        for name in self.cfg.data_types:
            if name in ["distance_to_image_plane", "distance_to_camera"]:
                shape = (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width, 1)
            elif name in ["normals"]:
                shape = (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width, 3)
            else:
                raise ValueError(f"Received unknown data type: {name}. Please check the configuration.")
            # allocate tensor to store the data
            self._camera_data.output[name] = torch.zeros((self._view.count, *shape), device=self.device)

    def _compute_intrinsic_matrices(self):
        """Computes the intrinsic matrices for the camera based on the config provided."""
        # get the sensor properties
        pattern_cfg = self.cfg.pattern_cfg

        if hasattr(pattern_cfg, "horizontal_aperture") and hasattr(pattern_cfg, "focal_length"):
            # check if vertical aperture is provided
            # if not then it is auto-computed based on the aspect ratio to preserve squared pixels
            if getattr(pattern_cfg, "vertical_aperture", None) is None:
                pattern_cfg.vertical_aperture = pattern_cfg.horizontal_aperture * pattern_cfg.height / pattern_cfg.width

            # compute the intrinsic matrix
            f_x = pattern_cfg.width * pattern_cfg.focal_length / pattern_cfg.horizontal_aperture
            f_y = pattern_cfg.height * pattern_cfg.focal_length / pattern_cfg.vertical_aperture
            c_x = pattern_cfg.horizontal_aperture_offset * f_x + pattern_cfg.width / 2
            c_y = pattern_cfg.vertical_aperture_offset * f_y + pattern_cfg.height / 2
            self._focal_length = pattern_cfg.focal_length
        else:
            # mjlab-native pinhole pattern: infer focal from fovy.
            half_fovy = 0.5 * math.radians(pattern_cfg.fovy)
            f_y = 0.5 * pattern_cfg.height / max(math.tan(half_fovy), 1e-8)
            f_x = f_y
            c_x = pattern_cfg.width / 2
            c_y = pattern_cfg.height / 2
            self._focal_length = 1.0

        assert self._camera_data.intrinsic_matrices is not None
        # allocate the intrinsic matrices
        self._camera_data.intrinsic_matrices[:, 0, 0] = f_x
        self._camera_data.intrinsic_matrices[:, 0, 2] = c_x
        self._camera_data.intrinsic_matrices[:, 1, 1] = f_y
        self._camera_data.intrinsic_matrices[:, 1, 2] = c_y

    def _compute_pinhole_rays_from_intrinsics(self, intrinsic_matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute pinhole rays from intrinsic matrices using IsaacLab-compatible conventions."""
        # get image plane mesh grid
        grid = torch.meshgrid(
            torch.arange(start=0, end=self.cfg.pattern_cfg.width, dtype=torch.int32, device=self.device),
            torch.arange(start=0, end=self.cfg.pattern_cfg.height, dtype=torch.int32, device=self.device),
            indexing="xy",
        )
        pixels = torch.vstack(list(map(torch.ravel, grid))).T
        # convert to homogeneous coordinate system
        pixels = torch.hstack([pixels, torch.ones((len(pixels), 1), device=self.device)])
        # move each pixel coordinate to the center of the pixel
        pixels += torch.tensor([[0.5, 0.5, 0]], device=self.device)
        pixels = pixels.to(dtype=intrinsic_matrices.dtype)
        # get pixel coordinates in camera frame
        pix_in_cam_frame = torch.matmul(torch.inverse(intrinsic_matrices), pixels.T)

        # robotics camera frame is (x forward, y left, z up) from camera frame with (x right, y down, z forward)
        # transform to robotics camera frame
        transform_vec = torch.tensor([1, -1, -1], device=self.device, dtype=intrinsic_matrices.dtype).unsqueeze(
            0
        ).unsqueeze(2)
        pix_in_cam_frame = pix_in_cam_frame[:, [2, 0, 1], :] * transform_vec
        # normalize ray directions
        ray_directions = (pix_in_cam_frame / torch.norm(pix_in_cam_frame, dim=1, keepdim=True)).permute(0, 2, 1)
        # for camera, we always ray-cast from the sensor's origin
        ray_starts = torch.zeros_like(ray_directions, device=self.device)
        return ray_starts, ray_directions

    def _compute_view_world_poses(self, env_ids: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtains the pose of the view the camera is attached to in the world frame.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z).
        """
        assert self._data is not None
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self._frame_type == "body":
            assert self._frame_body_id is not None
            pos_w = self._data.xpos[env_ids, self._frame_body_id]
            quat_w = self._data.xquat[env_ids, self._frame_body_id]
        elif self._frame_type == "site":
            assert self._frame_site_id is not None
            pos_w = self._data.site_xpos[env_ids, self._frame_site_id]
            quat_w = math_utils.quat_from_matrix(self._data.site_xmat[env_ids, self._frame_site_id].view(-1, 3, 3))
        elif self._frame_type == "geom":
            assert self._frame_geom_id is not None
            pos_w = self._data.geom_xpos[env_ids, self._frame_geom_id]
            quat_w = math_utils.quat_from_matrix(self._data.geom_xmat[env_ids, self._frame_geom_id].view(-1, 3, 3))
        else:
            raise RuntimeError(f"Unsupported view type: {self._frame_type}")
        # return the pose
        return pos_w.clone(), quat_w.clone()

    def _compute_camera_world_poses(self, env_ids: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the camera in the world frame.

        This function applies the offset pose to the pose of the view the camera is attached to.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z) in "world" convention.
        """
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # get the pose of the view the camera is attached to
        pos_w, quat_w = self._compute_view_world_poses(env_ids)
        # apply offsets
        # need to apply quat because offset relative to parent frame
        assert self._offset_pos is not None and self._offset_quat is not None
        pos_w += math_utils.quat_apply(quat_w, self._offset_pos[env_ids])
        quat_w = math_utils.quat_mul(quat_w, self._offset_quat[env_ids])

        return pos_w, quat_w

    def _compute_data(self) -> CameraData:
        return self._camera_data
