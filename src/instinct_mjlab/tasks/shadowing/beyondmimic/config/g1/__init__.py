"""Register Instinct Mj BeyondMimic G1 tasks."""

from instinct_mjlab.tasks.registry import register_instinct_task

from .beyondmimic_plane_cfg import g1_beyondmimic_plane_env_cfg
from .rl_cfgs import g1_beyondmimic_instinct_rl_cfg


register_instinct_task(
  task_id="Instinct-BeyondMimic-Plane-G1-v0",
  env_cfg_factory=lambda: g1_beyondmimic_plane_env_cfg(play=False),
  play_env_cfg_factory=lambda: g1_beyondmimic_plane_env_cfg(play=True),
  instinct_rl_cfg_factory=g1_beyondmimic_instinct_rl_cfg,
)

register_instinct_task(
  task_id="Instinct-BeyondMimic-Plane-G1-Play-v0",
  env_cfg_factory=lambda: g1_beyondmimic_plane_env_cfg(play=True),
  play_env_cfg_factory=lambda: g1_beyondmimic_plane_env_cfg(play=True),
  instinct_rl_cfg_factory=g1_beyondmimic_instinct_rl_cfg,
)
