from __future__ import annotations

from collections.abc import Sequence

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg
from mjlab.utils.logging import print_info
from mjlab.viewer.debug_visualizer import DebugVisualizer

from instinct_mjlab.managers import MultiRewardCfg, MultiRewardManager
from instinct_mjlab.monitors import MonitorManager


class InstinctRlEnv(ManagerBasedRlEnv):
  """This class adds additional logging mechanism on sensors to get more
  comprehensive running statistics.
  """

  def load_managers(self) -> None:
    # Route Instinct tasks through MultiRewardManager so reward logging matches
    # InstinctLab conventions:
    #   Episode_Reward/rewards_<term>/{max_episode_len_s,sum,timestep}
    reward_group_cfg = self._as_multi_reward_cfg(self.cfg.rewards)
    if reward_group_cfg is not None:
      self.cfg.rewards = {}

    super().load_managers()

    # Replace parent reward manager with MultiRewardManager when requested.
    if reward_group_cfg is not None:
      self.cfg.rewards = reward_group_cfg
      self.reward_manager = MultiRewardManager(
        self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
      )
      print_info(f"[INFO] {self.reward_manager}")

    monitor_cfg = getattr(self.cfg, "monitors", None)
    if monitor_cfg is None:
      monitor_cfg = {}
    self.monitor_manager = MonitorManager(monitor_cfg, self)

  @staticmethod
  def _as_multi_reward_cfg(rewards_cfg):
    """Convert reward config into a multi-reward group config when possible.

    - Keep existing MultiRewardCfg as-is.
    - For flat dict[str, RewardTermCfg], wrap into {"rewards": ...}.
    - For grouped dicts (dict[str, dict[str, RewardTermCfg]]), keep as-is.
    """
    if isinstance(rewards_cfg, MultiRewardCfg):
      return rewards_cfg
    if isinstance(rewards_cfg, dict):
      first_non_none = next(
        (value for value in rewards_cfg.values() if value is not None),
        None,
      )
      if first_non_none is None or isinstance(first_non_none, RewardTermCfg):
        return {"rewards": rewards_cfg}
      return rewards_cfg
    return None

  def setup_manager_visualizers(self) -> None:
    super().setup_manager_visualizers()
    if (
      getattr(self, "monitor_manager", None) is not None
      and hasattr(self.monitor_manager, "debug_vis")
    ):
      self.manager_visualizers["monitor_manager"] = self.monitor_manager

  def step(self, action: torch.Tensor):
    obs, reward, terminated, truncated, extras = super().step(action)
    if getattr(self, "monitor_manager", None) is not None:
      monitor_infos = self.monitor_manager.update(dt=self.step_dt)
      extras.setdefault("step", {})
      extras["step"].update(monitor_infos)
    return obs, reward, terminated, truncated, extras

  def update_visualizers(self, visualizer: DebugVisualizer) -> None:
    super().update_visualizers(visualizer)
    terrain = getattr(self.scene, "terrain", None)
    if terrain is not None and hasattr(terrain, "debug_vis"):
      terrain.debug_vis(visualizer)

  def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
    if isinstance(env_ids, Sequence):
      env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)

    super()._reset_idx(env_ids)

    if getattr(self, "monitor_manager", None) is not None:
      monitor_infos = self.monitor_manager.reset(env_ids, is_episode=True)
      self.extras["log"] = self.extras.get("log", {})
      self.extras["log"].update(monitor_infos)

  """
  Properties.
  """

  @property
  def num_rewards(self) -> int:
    return getattr(self.reward_manager, "num_rewards", 1)
