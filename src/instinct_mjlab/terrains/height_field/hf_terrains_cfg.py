from dataclasses import MISSING, dataclass, field
from typing import List

from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg, SubTerrainCfg


@dataclass(kw_only=True)
class HfTerrainBaseCfg(SubTerrainCfg):
    function: object | None = None
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
    base_thickness_ratio: float = 1.0


@dataclass(kw_only=True)
class HfPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = False


@dataclass(kw_only=True)
class HfInvertedPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    inverted: bool = True


@dataclass(kw_only=True)
class HfPyramidStairsTerrainCfg(HfTerrainBaseCfg):
    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = False


@dataclass(kw_only=True)
class HfInvertedPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg):
    inverted: bool = True


@dataclass(kw_only=True)
class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    obstacle_height_mode: str = "choice"
    obstacle_width_range: tuple[float, float] = MISSING
    obstacle_height_range: tuple[float, float] = MISSING
    num_obstacles: int = MISSING
    platform_width: float = 1.0


@dataclass(kw_only=True)
class HfWaveTerrainCfg(HfTerrainBaseCfg):
    amplitude_range: tuple[float, float] = MISSING
    num_waves: int = 1


@dataclass(kw_only=True)
class HfSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    stone_height_max: float = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    holes_depth: float = -10.0
    platform_width: float = 1.0

from . import hf_terrains

@dataclass
class WallTerrainCfgMixin:
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
    wall_prob: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # Probability of generating walls on [left, right, front, back] sides
    wall_height: float = 5.0  # Height of the walls
    wall_thickness: float = 0.05  # Thickness of the walls

@dataclass(kw_only=True)
class PerlinPlaneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_plane_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    noise_scale: float | List[float] = 0.05
    noise_frequency: int = 20

    fractal_octaves: int = 2

    fractal_lacunarity: float = 2.0

    fractal_gain: float = 0.25

    centering: bool = False  # If True, the noise will be centered around 0

@dataclass(kw_only=True)
class PerlinPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_pyramid_sloped_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = False
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinInvertedPyramidSlopedTerrainCfg(HfInvertedPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_pyramid_sloped_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = True
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_pyramid_stairs_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = False
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinInvertedPyramidStairsTerrainCfg(HfInvertedPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_pyramid_stairs_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = True
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinDiscreteObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_discrete_obstacles_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    obstacle_height_mode: str = "choice"
    obstacle_width_range: tuple[float, float] = MISSING
    obstacle_height_range: tuple[float, float] = MISSING
    num_obstacles: int = MISSING
    platform_width: float = 1.0
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinWaveTerrainCfg(HfWaveTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_wave_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    amplitude_range: tuple[float, float] = MISSING
    num_waves: int = 1
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinSteppingStonesTerrainCfg(HfSteppingStonesTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_stepping_stones_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    stone_height_max: float = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    holes_depth: float = -10.0
    platform_width: float = 1.0
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

# -- Newly added terrain configurations for parkour terrains-- #
@dataclass(kw_only=True)
class PerlinParapetTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a parapet terrain, can be used for jump and hurdle tasks."""

    function: object = hf_terrains.perlin_parapet_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    parapet_height: tuple[float, float] | float = (0.1, 0.3)
    parapet_length: tuple[float, float] | float = (0.1, 0.3)
    parapet_width: float | None = None
    curved_top_rate: float | None = None
    """The rate to generate curved top. If None, the top will be flat."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinGutterTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a gutter parkour terrain."""

    function: object = hf_terrains.perlin_gutter_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    gutter_length: tuple[float, float] | float = (0.5, 1.5)  # the distance between gutters
    gutter_depth: tuple[float, float] | float = (0.1, 0.3)  # the depth of the gutter
    gutter_width: float | None = None  # the length of the gutter
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinStairsUpDownTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs up and down parkour terrain."""

    function: object = hf_terrains.perlin_stairs_up_down_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    per_step_height: tuple[float, float] | float = MISSING
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = MISSING
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = MISSING
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinStairsDownUpTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs down and up parkour terrain."""

    function: object = hf_terrains.perlin_stairs_down_up_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    per_step_height: tuple[float, float] | float = MISSING
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = MISSING
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = MISSING
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinTiltTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilt terrain."""

    function: object = hf_terrains.perlin_tilt_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    wall_height: tuple[float, float] | float = MISSING
    wall_width: float | None = None
    wall_length: tuple[float, float] | float = MISSING
    wall_opening_angle: tuple[float, float] | float = MISSING  # in degrees
    wall_opening_width: tuple[float, float] | float = MISSING
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinTiltedRampTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilted ramp terrain."""

    function: object = hf_terrains.perlin_tilted_ramp_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    tilt_angle: tuple[float, float] | float = MISSING  # in degrees
    tilt_height: tuple[float, float] | float = MISSING
    tilt_width: tuple[float, float] | float = MISSING
    tilt_length: tuple[float, float] | float = MISSING
    switch_spacing: tuple[float, float] | float = MISSING
    spacing_curriculum: bool | None = None
    overlap_size: float | None = None
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinSlopeTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a slope up and down terrain with a flat ground in the middle."""

    function: object = hf_terrains.perlin_slope_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_angle: tuple[float, float] | float = MISSING  # in degrees
    per_slope_length: tuple[float, float] | float = MISSING
    platform_length: float = 1.0
    slope_width: float | None = None
    up_down: bool | None = None  # If True or None, the slope will be up and down, otherwise it will be down and up.
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinCrossStoneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a cross stone terrain."""

    function: object = hf_terrains.perlin_cross_stone_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    stone_size: tuple[float, float] = MISSING
    stone_height: tuple[float, float] | float = MISSING
    stone_spacing: tuple[float, float] | float = MISSING
    ground_depth: float = -0.5
    platform_width: float = 1.5
    xy_random_ratio: float = 0.2
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinSquareGapTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_square_gap_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    gap_distance_range: tuple[float, float] = (0.1, 0.5)
    gap_depth: tuple[float, float] = (0.2, 0.5)
    platform_width: float = 1.5
    border_width: float = 0.0

    perlin_cfg: PerlinPlaneTerrainCfg | None = None
