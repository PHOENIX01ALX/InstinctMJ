"""Python package serving as the migrated InstinctLab module."""

from __future__ import annotations

def _patch_terrain_pipeline() -> None:
  """Route mjlab scene terrain creation through Instinct_mjlab terrain classes."""
  import mjlab.scene.scene as scene_module
  import mjlab.terrains as terrains_module
  import mjlab.terrains.terrain_importer as terrain_importer_module

  from instinct_mjlab.terrains.terrain_generator import FiledTerrainGenerator
  from instinct_mjlab.terrains.terrain_importer import TerrainImporter as InstinctTerrainImporter

  scene_module.TerrainImporter = InstinctTerrainImporter
  terrain_importer_module.TerrainGenerator = FiledTerrainGenerator
  terrains_module.TerrainImporter = InstinctTerrainImporter
  terrains_module.TerrainGenerator = FiledTerrainGenerator


_patch_terrain_pipeline()
