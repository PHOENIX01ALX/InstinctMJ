from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace


def _to_marker_obj(value):
  if isinstance(value, dict):
    return SimpleNamespace(**{k: _to_marker_obj(v) for k, v in value.items()})
  return value


@dataclass
class VisualizationMarkersCfg:
  prim_path: str = "/Visuals/Markers"
  markers: dict[str, object] = field(default_factory=dict)

  def __post_init__(self):
    self.markers = {name: _to_marker_obj(cfg) for name, cfg in self.markers.items()}

  def replace(self, **kwargs) -> "VisualizationMarkersCfg":
    cfg = copy.deepcopy(self)
    for key, value in kwargs.items():
      if key == "markers" and isinstance(value, dict):
        setattr(cfg, key, {name: _to_marker_obj(v) for name, v in value.items()})
      else:
        setattr(cfg, key, value)
    return cfg


class VisualizationMarkers:
  def __init__(self, cfg: VisualizationMarkersCfg):
    self.cfg = cfg
    self._visible = True

  def set_visibility(self, visible: bool) -> None:
    self._visible = bool(visible)

  def visualize(self, *args, **kwargs) -> None:
    del args, kwargs
    return None


@dataclass
class _PrimitiveMarkerCfg:
  scale: tuple[float, float, float] = (0.15, 0.15, 0.15)
  color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


FRAME_MARKER_CFG = VisualizationMarkersCfg(
  prim_path="/Visuals/FrameMarker",
  markers={"frame": _PrimitiveMarkerCfg(scale=(0.15, 0.15, 0.15))},
)
RED_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
  prim_path="/Visuals/RedArrowXMarker",
  markers={"arrow": _PrimitiveMarkerCfg(scale=(0.35, 0.05, 0.05), color=(1.0, 0.0, 0.0, 1.0))},
)
BLUE_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
  prim_path="/Visuals/BlueArrowXMarker",
  markers={"arrow": _PrimitiveMarkerCfg(scale=(0.35, 0.05, 0.05), color=(0.0, 0.0, 1.0, 1.0))},
)

