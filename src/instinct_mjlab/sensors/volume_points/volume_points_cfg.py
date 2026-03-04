from dataclasses import MISSING, dataclass, field

from instinct_mjlab.visualization.marker_cfg import VisualizationMarkersCfg
from mjlab.sensor import SensorCfg

from .points_generator_cfg import PointsGeneratorCfg
from .volume_points import VolumePoints

VOLUME_POINTS_VISUALIZER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/volumePoints",
    markers={
        "sphere": {
            "radius": 0.01,
            "color": (0.0, 1.0, 0.0, 1.0),
        },
        "sphere_penetrated": {
            "radius": 0.01,
            "color": (1.0, 0.0, 0.0, 1.0),
        },
    },
)

@dataclass(kw_only=True)
class VolumePointsCfg(SensorCfg):
    """Configuration for the volume points sensor."""

    class_type: type = VolumePoints

    entity_name: str = ""
    """Entity name prefix used in MuJoCo body names (e.g. ``robot`` -> ``robot/<body>``)."""

    body_names: str | list[str] = ".*"
    """Body name pattern(s) inside ``entity_name``. Supports regex expressions."""

    filter_prim_paths_expr: list[str] = field(default_factory=list)
    """The list of primitive paths (or expressions) to filter volume points' body with. Defaults to an empty list,
    in which case
    no filtering is applied.

    .. note::
        The expression in the list can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Object`` will be replaced with ``/World/envs/env_.*/Object``.

    """

    points_generator: PointsGeneratorCfg = MISSING
    """ The points generator configuration. The generator function should be callable and accept only its cfg.
    """

    debug_vis: bool = False
    visualizer_cfg: VisualizationMarkersCfg = field(default_factory=lambda: VOLUME_POINTS_VISUALIZER_CFG)

    def build(self) -> VolumePoints:
        return VolumePoints(self)
