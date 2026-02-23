from __future__ import annotations

import numpy as np
import torch
import trimesh
import time
from typing import TYPE_CHECKING

from mjlab.terrains import SubTerrainCfg as SubTerrainBaseCfg
from mjlab.terrains import TerrainGenerator
from mjlab.terrains import TerrainImporter as TerrainImporterBase


class Timer:
    """Simple timer context manager."""

    def __init__(self, message: str):
        self.message = message
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.perf_counter() - self._start
        print(f"{self.message}: {elapsed:.4f}s")
        return False

if TYPE_CHECKING:
    from .terrain_importer_cfg import TerrainImporterCfg
    from .virtual_obstacle import VirtualObstacleBase


class TerrainImporter(TerrainImporterBase):
    def __init__(self, cfg: TerrainImporterCfg, device: str):
        self._debug_vis_enabled = False
        self._collision_debug_vis = bool(getattr(cfg, "collision_debug_vis", False))
        self._collision_debug_rgba = tuple(getattr(cfg, "collision_debug_rgba", (0.62, 0.2, 0.9, 0.35)))
        self._virtual_obstacles = {}
        for name, virtual_obstacle_cfg in cfg.virtual_obstacles.items():
            if virtual_obstacle_cfg is None:
                continue
            virtual_obstacle = virtual_obstacle_cfg.class_type(virtual_obstacle_cfg)
            self._virtual_obstacles[name] = virtual_obstacle

        if cfg.terrain_type in ("hacked_generator", "generator"):
            self._hacked_terrain_type = "hacked_generator"
            cfg.terrain_type = "plane"
        super().__init__(cfg, device)

        # Keep flat-patch buffers aligned with generated terrain when hacked_generator is used.
        if getattr(self, "_hacked_terrain_type", None) == "hacked_generator" and self.terrain_generator is not None:
            self._flat_patches = {
                name: torch.from_numpy(arr).to(device=self.device, dtype=torch.float)
                for name, arr in self.terrain_generator.flat_patches.items()
            }
            self._flat_patch_radii = dict(self.terrain_generator.flat_patch_radii)

        terrain_mesh = self._get_terrain_mesh_for_virtual_obstacles()
        if terrain_mesh is not None:
            self._generate_virtual_obstacles(terrain_mesh)
        if self._collision_debug_vis:
            self._apply_collision_debug_visual_style()

    @property
    def virtual_obstacles(self) -> dict[str, VirtualObstacleBase]:
        """Get the virtual obstacles representing the edges.
        TODO: Make the returned value more general.
        """
        # still pointing the same VirtualObstacleBase objects but the dict is a copy.
        return self._virtual_obstacles.copy()

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg] | None:
        """Get the specific configurations for all subterrains."""
        # This is a placeholder. The actual implementation should return the specific configurations.
        return (
            self.terrain_generator.subterrain_specific_cfgs
            if hasattr(self, "terrain_generator") and hasattr(self.terrain_generator, "subterrain_specific_cfgs")
            else None
        )

    """
    Operations - Import.
    """

    def import_ground_plane(self, name: str):
        """
        NOTE:
        This is a hack to fit self-defined tasks and keep InstinctLab's control flow.
        For hacked_generator, terrain_type is routed through "plane" branch while still
        compiling custom generated terrain meshes.
        """
        if getattr(self, "_hacked_terrain_type", None) == "hacked_generator":
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'hacked_generator' but no value provided for 'terrain_generator'.")
            terrain_generator_cls = getattr(
                self.cfg.terrain_generator,
                "class_type",
                TerrainGenerator,
            )
            self.terrain_generator = terrain_generator_cls(
                self.cfg.terrain_generator,
                device=self.device,
            )
            self.terrain_generator.compile(self._spec)
            # Diagnostic: log terrain mesh geom count for debugging.
            terrain_body = self._spec.body("terrain")
            geom_count = sum(1 for _ in terrain_body.geoms)
            print(f"[TerrainImporter] hacked_generator compiled: {geom_count} geom(s) in 'terrain' body")
            if self.terrain_generator.terrain_origins is not None:
                print(f"[TerrainImporter] terrain_origins shape: {self.terrain_generator.terrain_origins.shape}")
            self.configure_env_origins(self.terrain_generator.terrain_origins)
        else:
            super().import_ground_plane(name)

    def _get_terrain_mesh_for_virtual_obstacles(self) -> trimesh.Trimesh | None:
        if self.terrain_generator is None:
            return None
        terrain_mesh = getattr(self.terrain_generator, "terrain_mesh", None)
        return terrain_mesh

    def _generate_virtual_obstacles(self, mesh: trimesh.Trimesh):
        """Generate virtual obstacles from a terrain mesh."""
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())  # remove duplicate faces
        mesh.remove_unreferenced_vertices()
        # Generate virtual obstacles based on the generated terrain mesh.
        # NOTE: generate virtual obstacle first because it might modify the mesh.
        for name, virtual_obstacle in self._virtual_obstacles.items():
            with Timer(f"Generate virtual obstacle {name}"):
                virtual_obstacle.generate(mesh, device=self.device)

    def _apply_collision_debug_visual_style(self) -> None:
        """Tint terrain collision geoms so they are visible in the viewer."""
        terrain_body = self._spec.body("terrain")
        if terrain_body is None:
            return
        for geom in terrain_body.geoms:
            contype = int(getattr(geom, "contype", 1))
            conaffinity = int(getattr(geom, "conaffinity", 1))
            if contype == 0 and conaffinity == 0:
                continue
            geom.rgba[:] = self._collision_debug_rgba

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Set the debug visualization flag.

        Args:
            vis: True to enable debug visualization, False to disable.
        """
        self._debug_vis_enabled = debug_vis
        results = True

        for name, virtual_obstacle in self._virtual_obstacles.items():
            if debug_vis:
                virtual_obstacle.visualize()
            else:
                virtual_obstacle.disable_visualizer()

        return results

    def debug_vis(self, visualizer) -> None:
        """Draw virtual obstacles using viewer-native debug visualizer."""
        if not self._debug_vis_enabled:
            return
        for virtual_obstacle in self._virtual_obstacles.values():
            if hasattr(virtual_obstacle, "debug_vis"):
                virtual_obstacle.debug_vis(visualizer)

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        """Configure the environment origins.

        Args:
            origins: The origins of the environments. Shape is (num_envs, 3).
        """
        if origins is None and getattr(self, "_hacked_terrain_type", None) == "hacked_generator":
            # Keep terrain-based origins configured in hacked_generator path.
            return
        return super().configure_env_origins(origins)
