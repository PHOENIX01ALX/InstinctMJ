"""Configuration for the grouped-ray-cast camera sensor."""

from __future__ import annotations
from dataclasses import dataclass, field

from typing import Literal

from instinct_mjlab.visualization.markers import VisualizationMarkersCfg
from mjlab.sensor import PinholeCameraPatternCfg

from .grouped_ray_caster_camera import GroupedRayCasterCamera
from .grouped_ray_caster_cfg import GroupedRayCasterCfg

@dataclass(kw_only=True)
class GroupedRayCasterCameraCfg(GroupedRayCasterCfg):
    """Configuration for the grouped-ray-cast camera sensor."""

    class_type: type = GroupedRayCasterCamera

    @dataclass(kw_only=True)
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

        convention: Literal["opengl", "ros", "world"] = "ros"
        """The convention in which the frame offset is applied. Defaults to "ros".

        - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Offset is applied in the OpenGL (Usd.Camera) convention.
        - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Offset is applied in the ROS convention.
        - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Offset is applied in the World Frame convention.

        """

    offset: OffsetCfg = field(default_factory=OffsetCfg)
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""

    data_types: list[str] = field(default_factory=lambda: ["distance_to_image_plane"])
    """List of sensor names/types to enable for the camera. Defaults to ["distance_to_image_plane"]."""

    depth_clipping_behavior: Literal["max", "zero", "none"] = "none"
    """Clipping behavior for the camera for values exceed the maximum value. Defaults to "none".

    - ``"max"``: Values are clipped to the maximum value.
    - ``"zero"``: Values are clipped to zero.
    - ``"none``: No clipping is applied. Values will be returned as ``inf`` for ``distance_to_camera`` and ``nan``
      for ``distance_to_image_plane`` data type.
    """

    pattern: PinholeCameraPatternCfg = field(default_factory=PinholeCameraPatternCfg)
    """The pattern that defines the local ray starting positions and directions in a pinhole camera pattern."""

    focal_length: float | None = None
    """Optional focal length for aperture-based intrinsic construction."""

    horizontal_aperture: float | None = None
    """Optional horizontal aperture for aperture-based intrinsic construction."""

    vertical_aperture: float | None = None
    """Optional vertical aperture for aperture-based intrinsic construction."""

    horizontal_aperture_offset: float = 0.0
    """Horizontal aperture offset in aperture-based intrinsic construction."""

    vertical_aperture_offset: float = 0.0
    """Vertical aperture offset in aperture-based intrinsic construction."""

    visualizer_cfg: VisualizationMarkersCfg = field(default_factory=lambda: VisualizationMarkersCfg(
        prim_path="/Visuals/RayCaster",
        markers={
            "hit": {
                "radius": 0.02,
                "color": (1.0, 0.0, 0.0, 1.0),
            },
            "frame": {
                "scale": (0.1, 0.1, 0.1),
            },
        },
    ))

    def __post_init__(self):
        # Camera rays should use full frame orientation.
        self.ray_alignment = "base"
