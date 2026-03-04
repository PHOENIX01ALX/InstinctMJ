"""Shared scene visual style utilities for Instinct Mj tasks.

This module centralizes the native-viewer environment style so each task cfg
can apply the same sky/ground look with a single helper call.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import mujoco
from mjlab.scene import SceneCfg
from mjlab.utils.spec_config import MaterialCfg, TextureCfg


@dataclass(frozen=True, kw_only=True)
class SceneVisualStyleCfg:
  """Visual style parameters for MuJoCo scene-level rendering."""

  sky_rgb_top: tuple[float, float, float] = (0.95, 0.97, 0.99)
  sky_rgb_horizon: tuple[float, float, float] = (0.83, 0.89, 0.95)
  # Use flat texture so the ground is pure white without checker stripes.
  ground_builtin: str = "flat"
  ground_mark: str = "none"
  # Bright white ground style for clearer terrain visibility in play mode.
  ground_rgb1: tuple[float, float, float] = (0.98, 0.98, 0.98)
  ground_rgb2: tuple[float, float, float] = (0.98, 0.98, 0.98)
  ground_mark_rgb: tuple[float, float, float] = (0.98, 0.98, 0.98)
  haze_rgba: tuple[float, float, float, float] = (0.95, 0.97, 0.99, 1.0)
  headlight_ambient: tuple[float, float, float] = (0.78, 0.78, 0.78)
  headlight_diffuse: tuple[float, float, float] = (0.68, 0.68, 0.68)
  headlight_specular: tuple[float, float, float] = (0.06, 0.06, 0.06)
  sky_texture_size: tuple[int, int] = (512, 3072)
  ground_texture_size: tuple[int, int] = (300, 300)
  ground_texrepeat: tuple[int, int] = (4, 4)
  ground_reflectance: float = 0.015


INSTINCT_BRIGHT_SCENE_STYLE_CFG = SceneVisualStyleCfg()
"""Default bright sky + bright ground visual style."""


def _run_spec_modifiers(
  spec: mujoco.MjSpec,
  modifiers: tuple[Callable[[mujoco.MjSpec], None], ...],
) -> None:
  for modifier in modifiers:
    modifier(spec)


def attach_scene_spec_modifier(
  scene_cfg: SceneCfg,
  spec_modifier: Callable[[mujoco.MjSpec], None],
) -> None:
  """Attach a scene spec modifier while preserving any existing modifier."""

  if scene_cfg.spec_fn is None:
    scene_cfg.spec_fn = spec_modifier
    return

  scene_cfg.spec_fn = partial(
    _run_spec_modifiers,
    modifiers=(scene_cfg.spec_fn, spec_modifier),
  )


def edit_spec_with_scene_visual_style(
  spec: mujoco.MjSpec,
  *,
  style_name: str,
  style_cfg: SceneVisualStyleCfg = INSTINCT_BRIGHT_SCENE_STYLE_CFG,
  terrain_body_name: str = "terrain",
  preserve_collision_rgba: bool = False,
) -> None:
  """Apply bright skybox + bright ground material to a MuJoCo spec."""

  skybox_texture_name = f"{style_name}_skybox"
  ground_texture_name = f"{style_name}_groundplane"
  ground_material_name = f"{style_name}_groundplane"

  existing_skybox = None
  for tex in spec.textures:
    if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
      existing_skybox = tex
      break

  if existing_skybox is not None:
    existing_skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    existing_skybox.rgb1[:] = style_cfg.sky_rgb_top
    existing_skybox.rgb2[:] = style_cfg.sky_rgb_horizon
    existing_skybox.width = style_cfg.sky_texture_size[0]
    existing_skybox.height = style_cfg.sky_texture_size[1]
  else:
    TextureCfg(
      name=skybox_texture_name,
      type="skybox",
      builtin="gradient",
      rgb1=style_cfg.sky_rgb_top,
      rgb2=style_cfg.sky_rgb_horizon,
      width=style_cfg.sky_texture_size[0],
      height=style_cfg.sky_texture_size[1],
    ).edit_spec(spec)

  TextureCfg(
    name=ground_texture_name,
    type="2d",
    builtin=style_cfg.ground_builtin,
    mark=style_cfg.ground_mark,
    rgb1=style_cfg.ground_rgb1,
    rgb2=style_cfg.ground_rgb2,
    markrgb=style_cfg.ground_mark_rgb,
    width=style_cfg.ground_texture_size[0],
    height=style_cfg.ground_texture_size[1],
  ).edit_spec(spec)
  MaterialCfg(
    name=ground_material_name,
    texuniform=True,
    texrepeat=style_cfg.ground_texrepeat,
    reflectance=style_cfg.ground_reflectance,
    texture=ground_texture_name,
  ).edit_spec(spec)

  spec.visual.rgba.haze[:] = style_cfg.haze_rgba
  spec.visual.headlight.ambient[:] = style_cfg.headlight_ambient
  spec.visual.headlight.diffuse[:] = style_cfg.headlight_diffuse
  spec.visual.headlight.specular[:] = style_cfg.headlight_specular

  terrain_body = None
  for body in spec.bodies:
    if body.name == terrain_body_name:
      terrain_body = body
      break
  if terrain_body is None:
    return

  for geom in terrain_body.geoms:
    if preserve_collision_rgba:
      contype = int(getattr(geom, "contype", 1))
      conaffinity = int(getattr(geom, "conaffinity", 1))
      if contype != 0 or conaffinity != 0:
        continue
    geom.material = ground_material_name
    # Keep border geoms visually consistent with the ground material color.
    geom.rgba[:] = (
      style_cfg.ground_rgb1[0],
      style_cfg.ground_rgb1[1],
      style_cfg.ground_rgb1[2],
      1.0,
    )


def apply_scene_visual_style(
  scene_cfg: SceneCfg,
  *,
  style_name: str,
  style_cfg: SceneVisualStyleCfg = INSTINCT_BRIGHT_SCENE_STYLE_CFG,
  terrain_body_name: str = "terrain",
  preserve_collision_rgba: bool = False,
) -> None:
  """Bind the shared visual style to a scene cfg."""

  attach_scene_spec_modifier(
    scene_cfg,
    partial(
      edit_spec_with_scene_visual_style,
      style_name=style_name,
      style_cfg=style_cfg,
      terrain_body_name=terrain_body_name,
      preserve_collision_rgba=preserve_collision_rgba,
    ),
  )
