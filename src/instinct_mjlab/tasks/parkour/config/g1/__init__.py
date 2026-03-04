"""Register Instinct Mj parkour G1 tasks."""

# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from instinct_mjlab.tasks.registry import register_instinct_task

from .agents.instinct_rl_amp_cfg import G1ParkourPPORunnerCfg
from .g1_parkour_target_amp_cfg import instinct_g1_parkour_amp_final_cfg


register_instinct_task(
  task_id="Instinct-Parkour-Target-Amp-G1-v0",
  env_cfg_factory=lambda: instinct_g1_parkour_amp_final_cfg(play=False, shoe=True),
  play_env_cfg_factory=lambda: instinct_g1_parkour_amp_final_cfg(play=True, shoe=True),
  instinct_rl_cfg_factory=G1ParkourPPORunnerCfg,
)


register_instinct_task(
  task_id="Instinct-Parkour-Target-Amp-G1-Play-v0",
  env_cfg_factory=lambda: instinct_g1_parkour_amp_final_cfg(play=True, shoe=True),
  play_env_cfg_factory=lambda: instinct_g1_parkour_amp_final_cfg(play=True, shoe=True),
  instinct_rl_cfg_factory=G1ParkourPPORunnerCfg,
)
