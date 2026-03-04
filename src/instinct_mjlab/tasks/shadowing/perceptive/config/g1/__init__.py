"""Register Instinct Mj perceptive G1 tasks."""

from instinct_mjlab.tasks.registry import register_instinct_task

from .perceptive_shadowing_cfg import (
  G1PerceptiveShadowingEnvCfg,
  G1PerceptiveShadowingEnvCfg_PLAY,
)
from .perceptive_vae_cfg import (
  G1PerceptiveVaeEnvCfg,
  G1PerceptiveVaeEnvCfg_PLAY,
)
from .rl_cfgs import (
  g1_perceptive_shadowing_instinct_rl_cfg,
  g1_perceptive_vae_instinct_rl_cfg,
)


register_instinct_task(
  task_id="Instinct-Perceptive-Shadowing-G1-v0",
  env_cfg_factory=G1PerceptiveShadowingEnvCfg,
  play_env_cfg_factory=G1PerceptiveShadowingEnvCfg_PLAY,
  instinct_rl_cfg_factory=g1_perceptive_shadowing_instinct_rl_cfg,
)

register_instinct_task(
  task_id="Instinct-Perceptive-Shadowing-G1-Play-v0",
  env_cfg_factory=G1PerceptiveShadowingEnvCfg_PLAY,
  play_env_cfg_factory=G1PerceptiveShadowingEnvCfg_PLAY,
  instinct_rl_cfg_factory=g1_perceptive_shadowing_instinct_rl_cfg,
)

register_instinct_task(
  task_id="Instinct-Perceptive-Vae-G1-v0",
  env_cfg_factory=G1PerceptiveVaeEnvCfg,
  play_env_cfg_factory=G1PerceptiveVaeEnvCfg_PLAY,
  instinct_rl_cfg_factory=g1_perceptive_vae_instinct_rl_cfg,
)

register_instinct_task(
  task_id="Instinct-Perceptive-Vae-G1-Play-v0",
  env_cfg_factory=G1PerceptiveVaeEnvCfg_PLAY,
  play_env_cfg_factory=G1PerceptiveVaeEnvCfg_PLAY,
  instinct_rl_cfg_factory=g1_perceptive_vae_instinct_rl_cfg,
)
