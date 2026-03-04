from __future__ import annotations
from dataclasses import dataclass, field

from typing import Literal

from mjlab.terrains import TerrainImporterCfg as TerrainImporterCfgBase

from .terrain_importer import TerrainImporter
from .virtual_obstacle import VirtualObstacleCfg


@dataclass(kw_only=True)
class TerrainImporterCfg(TerrainImporterCfgBase):
    class_type: type = TerrainImporter
    """The inherited class to use for the terrain importer."""

    virtual_obstacles: dict[str, VirtualObstacleCfg] = field(default_factory=dict)
    """The virtual obstacles to use for the terrain importer."""

    virtual_obstacle_source: Literal["mesh", "heightfield"] = "mesh"
    """Source used to generate virtual obstacles.

    - ``"mesh"``: Use concatenated terrain mesh surface.
    - ``"heightfield"``: Extract obstacle edges directly from MuJoCo hfield data.
    """

    virtual_obstacle_hfield_method: Literal["mesh_like"] = "mesh_like"
    """Heightfield virtual-obstacle extraction method.

    - ``"mesh_like"``: Reconstruct hfield surface mesh then run mesh-style obstacle extraction.
    """

    virtual_obstacle_hfield_trace_simplify_epsilon: float = 0.03
    """Polyline simplification tolerance (meters) for mesh-like edge tracing."""

    virtual_obstacle_hfield_trace_min_segment_length: float = 0.04
    """Minimum segment length (meters) kept after mesh-like edge tracing."""

    virtual_obstacle_hfield_trace_snap_xy: float | None = None
    """XY snap size (meters) for graph nodes in mesh-like edge tracing.

    If None, a value is estimated from primitive hfield edge spacing.
    """

    virtual_obstacle_hfield_trace_snap_z: float | None = None
    """Z snap size (meters) for graph nodes in mesh-like edge tracing.

    This prevents edges at different heights from being merged into one node.
    If None, a value is estimated from edge segment distribution.
    """

    virtual_obstacle_hfield_trace_collinear_angle_threshold: float = 6.0
    """Angle threshold (degrees) for collinear simplification during tracing.

    Smaller values preserve corners more aggressively.
    """

    virtual_obstacle_hfield_mesh_like_drop_cell_diagonals: bool = True
    """Whether ``mesh_like`` should ignore intra-cell diagonal sharp edges.

    These edges are triangulation artifacts from hfield quads and can create
    fragmented virtual-obstacle segments.
    """

    virtual_obstacle_hfield_mesh_like_trace_segments: bool = True
    """Whether ``mesh_like`` sharp-edge segments should be graph-traced/merged."""

    virtual_obstacle_hfield_mesh_like_min_edge_length: float = 0.0
    """Minimum raw sharp-edge length (meters) kept in ``mesh_like`` before tracing."""

    collision_debug_vis: bool = False
    """Whether to visualize terrain collision geoms by tinting them in purple."""

    collision_debug_rgba: tuple[float, float, float, float] = (0.62, 0.2, 0.9, 0.35)
    """RGBA tint for terrain collision debug visualization."""

    terrain_type: Literal["generator", "plane", "hacked_generator"] = "generator"
    """The type of terrain to generate. Defaults to "generator".

    Available options are "plane" and "generator".

    ## NOTE
    The TerrainImporter also supports "hacked_generator" and routes it
    through the same generator pipeline as "generator".
    """
