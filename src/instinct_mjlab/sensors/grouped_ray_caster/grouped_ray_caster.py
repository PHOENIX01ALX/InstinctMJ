from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import warp as wp

from mjlab.sensor import RayCastSensor

if TYPE_CHECKING:
    from .grouped_ray_caster_cfg import GroupedRayCasterCfg


class GroupedRayCaster(RayCastSensor):
    """Ray-caster with per-environment mutable ray offsets/directions."""

    cfg: GroupedRayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: GroupedRayCasterCfg):
        super().__init__(cfg)
        self._num_envs = 0
        self._ALL_INDICES = torch.empty(0, dtype=torch.long)
        self.drift: torch.Tensor | None = None
        self.ray_starts: torch.Tensor | None = None
        self.ray_directions: torch.Tensor | None = None
        self._min_distance = 0.0
        self._mesh_filter_epsilon = 0.0
        self._mesh_filter_max_hops = 1
        self._needs_filter_continue = False
        self._mesh_filter_enabled: bool = False
        self._allowed_geom_lut: torch.Tensor | None = None

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)

        self._min_distance = float(self.cfg.min_distance)
        if self._min_distance < 0.0:
            raise ValueError(f"min_distance must be >= 0.0, got {self.cfg.min_distance}.")

        self._mesh_filter_epsilon = float(self.cfg.mesh_filter_epsilon)
        if self._mesh_filter_epsilon <= 0.0:
            raise ValueError(f"mesh_filter_epsilon must be > 0.0, got {self.cfg.mesh_filter_epsilon}.")

        self._mesh_filter_max_hops = int(self.cfg.mesh_filter_max_hops)
        if self._mesh_filter_max_hops < 1:
            raise ValueError(f"mesh_filter_max_hops must be >= 1, got {self.cfg.mesh_filter_max_hops}.")

        self._num_envs = data.nworld
        self._ALL_INDICES = torch.arange(self._num_envs, device=device, dtype=torch.long)
        self.drift = torch.zeros(self._num_envs, 3, device=device, dtype=torch.float32)

        self.ray_starts = self._local_offsets.unsqueeze(0).repeat(self._num_envs, 1, 1).clone()
        self.ray_directions = self._local_directions.unsqueeze(0).repeat(self._num_envs, 1, 1).clone()
        self._initialize_mesh_path_filter(mj_model, device)
        self._needs_filter_continue = self._mesh_filter_enabled or self._min_distance > 0.0

    def prepare_rays(self) -> None:
        """PRE-GRAPH: Transform per-env local rays to world frame."""
        if self._frame_type == "body":
            frame_pos = self._data.xpos[:, self._frame_body_id]
            frame_mat = self._data.xmat[:, self._frame_body_id].view(-1, 3, 3)
        elif self._frame_type == "site":
            frame_pos = self._data.site_xpos[:, self._frame_site_id]
            frame_mat = self._data.site_xmat[:, self._frame_site_id].view(-1, 3, 3)
        else:  # geom
            frame_pos = self._data.geom_xpos[:, self._frame_geom_id]
            frame_mat = self._data.geom_xmat[:, self._frame_geom_id].view(-1, 3, 3)

        # note: we clone here because we are read-only operations
        frame_pos = frame_pos.clone()
        frame_mat = frame_mat.clone()

        rot_mat = self._compute_alignment_rotation(frame_mat)
        world_offsets = torch.einsum("bij,bnj->bni", rot_mat, self.ray_starts)
        world_origins = frame_pos.unsqueeze(1) + world_offsets
        ray_directions_w = torch.einsum("bij,bnj->bni", rot_mat, self.ray_directions)

        if self.drift is not None:
            # apply drift
            world_origins = world_origins + self.drift.unsqueeze(1)
            frame_pos = frame_pos + self.drift

        pnt_torch = wp.to_torch(self._ray_pnt).view(self._num_envs, self._num_rays, 3)
        vec_torch = wp.to_torch(self._ray_vec).view(self._num_envs, self._num_rays, 3)
        pnt_torch.copy_(world_origins)
        vec_torch.copy_(ray_directions_w)

        self._cached_world_origins = world_origins
        self._cached_world_rays = ray_directions_w
        self._cached_frame_pos = frame_pos
        self._cached_frame_mat = frame_mat

    def postprocess_rays(self) -> None:
        super().postprocess_rays()
        if not self._needs_filter_continue:
            return
        self._apply_hit_filter_and_continue()

    def _initialize_mesh_path_filter(self, mj_model, device: str) -> None:
        exprs = [expr for expr in self.cfg.mesh_prim_paths if expr]
        if len(exprs) == 0:
            self._mesh_filter_enabled = False
            self._allowed_geom_lut = None
            return

        compiled_patterns: list[re.Pattern[str]] = []
        for expr in exprs:
            normalized_expr = expr.replace("{ENV_REGEX_NS}", "/World/envs/env_\\d+")
            compiled_patterns.append(re.compile(normalized_expr))

        body_aliases = self._build_body_aliases()
        allowed_geom_ids: list[int] = []
        for geom_id in range(int(mj_model.ngeom)):
            candidates = self._build_geom_path_candidates(mj_model, geom_id, body_aliases)
            if any(pattern.search(path) is not None for pattern in compiled_patterns for path in candidates):
                allowed_geom_ids.append(geom_id)

        if len(allowed_geom_ids) == 0:
            raise RuntimeError(
                "GroupedRayCaster mesh filter matched no MuJoCo geoms."
                f"\n\tmesh_prim_paths: {self.cfg.mesh_prim_paths}"
            )

        allowed_geom_lut = torch.zeros(int(mj_model.ngeom), device=device, dtype=torch.bool)
        allowed_geom_lut[torch.tensor(sorted(set(allowed_geom_ids)), device=device, dtype=torch.long)] = True
        self._allowed_geom_lut = allowed_geom_lut
        self._mesh_filter_enabled = True

    def _apply_hit_filter_and_continue(self) -> None:
        ray_geomid_torch = wp.to_torch(self._ray_geomid).view(self._num_envs, self._num_rays)
        ray_dist_torch = wp.to_torch(self._ray_dist).view(self._num_envs, self._num_rays)
        ray_normal_torch = wp.to_torch(self._ray_normal).view(self._num_envs, self._num_rays, 3)

        geom_ids = ray_geomid_torch.to(dtype=torch.long)
        hit_mask = self._distances >= 0.0
        if not torch.any(hit_mask):
            return

        if self._mesh_filter_enabled:
            allowed_mask = torch.zeros_like(hit_mask)
            allowed_mask[hit_mask] = self._allowed_geom_lut[geom_ids[hit_mask]]
        else:
            allowed_mask = torch.ones_like(hit_mask)

        reject_mask = hit_mask & ((~allowed_mask) | (self._distances <= self._min_distance))
        if not torch.any(reject_mask):
            return

        final_distances = self._distances.clone()
        final_hit_pos = self._hit_pos_w.clone()
        final_normals = self._normals_w.clone()

        # Rejected hits are treated as "continue ray-casting" candidates.
        final_distances[reject_mask] = -1.0
        final_hit_pos[reject_mask] = self._cached_world_origins[reject_mask]
        final_normals[reject_mask] = 0.0

        world_origins = self._cached_world_origins
        world_rays = self._cached_world_rays
        eps = self._mesh_filter_epsilon
        max_hops = self._mesh_filter_max_hops

        current_origins = self._hit_pos_w.clone()
        current_origins[reject_mask] = self._hit_pos_w[reject_mask] + world_rays[reject_mask] * eps

        traveled = torch.zeros_like(self._distances)
        traveled[reject_mask] = self._distances[reject_mask] + eps
        remaining = self.cfg.max_distance - traveled
        active = reject_mask & (remaining > 0.0)
        if not torch.any(active):
            self._distances.copy_(final_distances)
            self._hit_pos_w.copy_(final_hit_pos)
            self._normals_w.copy_(final_normals)
            return

        ray_pnt_torch = wp.to_torch(self._ray_pnt).view(self._num_envs, self._num_rays, 3)
        ray_vec_torch = wp.to_torch(self._ray_vec).view(self._num_envs, self._num_rays, 3)
        ray_vec_torch.copy_(world_rays)

        for _ in range(max_hops):
            if not torch.any(active):
                break

            ray_pnt_torch.copy_(world_origins)
            ray_pnt_torch[active] = current_origins[active]

            self.raycast_kernel(rc=self._ctx.render_context)

            hop_distances = ray_dist_torch.clone()
            hop_geom_ids = ray_geomid_torch.to(dtype=torch.long)
            hop_normals = ray_normal_torch.clone()

            hop_hit = active & (hop_distances >= 0.0) & (hop_distances <= remaining)
            if not torch.any(hop_hit):
                break

            if self._mesh_filter_enabled:
                hop_allowed = torch.zeros_like(hop_hit)
                hop_allowed[hop_hit] = self._allowed_geom_lut[hop_geom_ids[hop_hit]]
            else:
                hop_allowed = hop_hit

            hop_hit_pos = current_origins + world_rays * hop_distances.unsqueeze(-1)
            total_distances = traveled + hop_distances
            hop_accept = hop_hit & hop_allowed & (total_distances > self._min_distance)
            if torch.any(hop_accept):
                final_distances[hop_accept] = total_distances[hop_accept]
                final_hit_pos[hop_accept] = hop_hit_pos[hop_accept]
                final_normals[hop_accept] = hop_normals[hop_accept]

            hop_reject = hop_hit & (~hop_accept)
            if not torch.any(hop_reject):
                break

            current_origins[hop_reject] = hop_hit_pos[hop_reject] + world_rays[hop_reject] * eps
            traveled[hop_reject] = traveled[hop_reject] + hop_distances[hop_reject] + eps
            remaining[hop_reject] = self.cfg.max_distance - traveled[hop_reject]
            active = hop_reject & (remaining > 0.0)

        self._distances.copy_(final_distances)
        self._hit_pos_w.copy_(final_hit_pos)
        self._normals_w.copy_(final_normals)

    def _build_body_aliases(self) -> dict[str, set[str]]:
        body_aliases: dict[str, set[str]] = {}
        for mesh_name, link_name in self.cfg.aux_mesh_and_link_names.items():
            mesh_name = str(mesh_name).strip()
            if mesh_name == "":
                continue
            link_name = mesh_name if link_name is None else str(link_name).strip()
            if link_name == "":
                continue
            key = link_name.lower()
            if key not in body_aliases:
                body_aliases[key] = set()
            body_aliases[key].add(mesh_name)
        return body_aliases

    def _build_geom_path_candidates(
        self,
        mj_model,
        geom_id: int,
        body_aliases: dict[str, set[str]],
    ) -> list[str]:
        full_geom_name = mj_model.geom(geom_id).name or ""
        body_id = int(mj_model.geom_bodyid[geom_id])
        full_body_name = mj_model.body(body_id).name or ""

        entity_name, local_body_name = self._split_prefixed_name(full_body_name)
        _, local_geom_name = self._split_prefixed_name(full_geom_name)

        body_tokens = [local_body_name]
        body_tokens.extend(sorted(body_aliases.get(local_body_name.lower(), set())))
        body_tokens = [token for token in body_tokens if token != ""]
        if len(body_tokens) == 0:
            body_tokens = [f"body_{body_id}"]

        paths: set[str] = set()
        if full_body_name:
            paths.add(f"/{full_body_name}")
        if full_geom_name:
            paths.add(f"/{full_geom_name}")

        entity_lower = entity_name.lower()
        for body_token in body_tokens:
            if entity_name:
                paths.add(f"/World/{entity_name}/{body_token}")
                paths.add(f"/World/envs/env_0/{entity_name}/{body_token}")
                if local_geom_name:
                    paths.add(f"/World/{entity_name}/{body_token}/{local_geom_name}")
                    paths.add(f"/World/envs/env_0/{entity_name}/{body_token}/{local_geom_name}")

            if entity_lower == "robot":
                paths.add(f"/World/envs/env_0/Robot/{body_token}")
                if local_geom_name:
                    paths.add(f"/World/envs/env_0/Robot/{body_token}/{local_geom_name}")

            if entity_lower in {"terrain", "ground"} or "ground" in full_body_name.lower() or "ground" in full_geom_name.lower():
                paths.add("/World/ground/")
                paths.add(f"/World/ground/{body_token}")
                if local_geom_name:
                    paths.add(f"/World/ground/{body_token}/{local_geom_name}")

        return sorted(path for path in paths if path != "")

    @staticmethod
    def _split_prefixed_name(full_name: str) -> tuple[str, str]:
        if "/" in full_name:
            return full_name.split("/", 1)
        return "", full_name
