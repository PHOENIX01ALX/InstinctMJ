"""Adapter that lets instinct_rl train directly on mjlab environments."""

from __future__ import annotations

import torch
from instinct_rl.env import VecEnv

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


class InstinctRlVecEnvWrapper(VecEnv):
  """Wrap `ManagerBasedRlEnv` to match instinct_rl VecEnv contract."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    policy_group: str = "actor",
    critic_group: str | None = "critic",
  ):
    self.env = env
    self.policy_group = policy_group
    self.critic_group = critic_group

    self.num_envs = self.unwrapped.num_envs
    self.device = torch.device(self.unwrapped.device)
    self.max_episode_length = self.unwrapped.max_episode_length
    self.num_actions = self.unwrapped.action_manager.total_action_dim
    self._log_defaults: dict[str, torch.Tensor] = {}

    active_groups = set(self.unwrapped.observation_manager.active_terms.keys())
    if self.critic_group is not None and self.critic_group not in active_groups:
      self.critic_group = None

    self._group_map = {"policy": self.policy_group}
    if self.critic_group is not None:
      self._group_map["critic"] = self.critic_group

    self.num_rewards = int(getattr(self.unwrapped, "num_rewards", 1))
    self.num_obs = self._group_flat_dim(self.policy_group)
    self.num_critic_obs = (
      self._group_flat_dim(self.critic_group)
      if self.critic_group is not None
      else None
    )

    # Reset once because instinct_rl runner does not call reset before rollout.
    self.env.reset()

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    return self.unwrapped.cfg

  @property
  def render_mode(self) -> str | None:
    return self.env.render_mode

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space

  @classmethod
  def class_name(cls) -> str:
    return cls.__name__

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env.unwrapped

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.unwrapped.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:
    self.unwrapped.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.unwrapped.seed(seed)

  def get_obs_segments(self, group_name: str = "policy") -> dict[str, tuple[int, ...]]:
    source_group = self._group_map.get(group_name, group_name)
    active_terms = self.unwrapped.observation_manager.active_terms[source_group]
    term_dims = self.unwrapped.observation_manager.group_obs_term_dim[source_group]
    obs_segments: dict[str, tuple[int, ...]] = {}
    for term_name, term_dim in zip(active_terms, term_dims, strict=False):
      obs_segments[term_name] = term_dim
    return obs_segments

  def get_obs_format(self) -> dict[str, dict[str, tuple[int, ...]]]:
    obs_format: dict[str, dict[str, tuple[int, ...]]] = {}
    active_terms = self.unwrapped.observation_manager.active_terms
    term_dims = self.unwrapped.observation_manager.group_obs_term_dim

    for exposed_name, source_group in self._group_map.items():
      group_format: dict[str, tuple[int, ...]] = {}
      for term_name, dim in zip(
        active_terms[source_group],
        term_dims[source_group],
        strict=False,
      ):
        group_format[term_name] = dim
      obs_format[exposed_name] = group_format

    for group_name, group_term_names in active_terms.items():
      if group_name in obs_format:
        continue
      group_format: dict[str, tuple[int, ...]] = {}
      for term_name, dim in zip(
        group_term_names,
        term_dims[group_name],
        strict=False,
      ):
        group_format[term_name] = dim
      obs_format[group_name] = group_format

    return obs_format

  def get_observations(self) -> tuple[torch.Tensor, dict]:
    obs_dict = self.unwrapped.observation_manager.compute()
    packed_obs = self._pack_observations(obs_dict)
    return packed_obs["policy"], {"observations": packed_obs}

  def reset(self) -> tuple[torch.Tensor, dict]:
    obs_dict, extras = self.env.reset()
    packed_obs = self._pack_observations(obs_dict)
    extras = dict(extras)
    self._stabilize_log(extras)
    extras.setdefault("step", {})
    extras.setdefault("episode", {})
    extras["observations"] = packed_obs
    return packed_obs["policy"], extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
    packed_obs = self._pack_observations(obs_dict)
    dones = (terminated | truncated).to(dtype=torch.long)

    extras = dict(extras)
    self._stabilize_log(extras)
    extras.setdefault("step", {})
    extras.setdefault("episode", {})
    extras["observations"] = packed_obs
    if not self.unwrapped.cfg.is_finite_horizon:
      extras["time_outs"] = truncated

    # MultiRewardManager returns dict[str, Tensor]; stack into (num_envs, N).
    if isinstance(rewards, dict):
      rewards = torch.stack(list(rewards.values()), dim=-1)
    if rewards.ndim == 1:
      rewards = rewards.unsqueeze(1)
    return packed_obs["policy"], rewards, dones, extras

  def close(self) -> None:
    self.env.close()

  def _stabilize_log(self, extras: dict) -> None:
    """Keep extras['log'] keys stable across steps for instinct_rl logging."""
    log_info = extras.get("log")
    if not isinstance(log_info, dict):
      return

    stable_log = dict(log_info)
    for key, value in stable_log.items():
      self._log_defaults[key] = self._make_zero_like(value)
    for key, default_value in self._log_defaults.items():
      stable_log.setdefault(key, default_value)
    extras["log"] = stable_log

  def _make_zero_like(self, value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
      return torch.zeros_like(value)
    if isinstance(value, bool):
      return torch.tensor(False, device=self.device)
    if isinstance(value, int):
      return torch.tensor(0, device=self.device)
    if isinstance(value, float):
      return torch.tensor(0.0, device=self.device)
    return torch.tensor(0.0, device=self.device)

  def _pack_observations(self, obs_dict: dict) -> dict[str, torch.Tensor]:
    packed: dict[str, torch.Tensor] = {}
    for exposed_name, source_group in self._group_map.items():
      packed[exposed_name] = self._flatten_group(
        group_name=source_group, group_obs=obs_dict[source_group]
      )

    for group_name, group_obs in obs_dict.items():
      if group_name in self._group_map.values():
        continue
      packed[group_name] = self._flatten_group(group_name=group_name, group_obs=group_obs)
    return packed

  def _flatten_group(
    self, *, group_name: str, group_obs: torch.Tensor | dict[str, torch.Tensor]
  ) -> torch.Tensor:
    if isinstance(group_obs, torch.Tensor):
      return group_obs.flatten(start_dim=1)

    active_terms = self.unwrapped.observation_manager.active_terms[group_name]
    flattened_terms: list[torch.Tensor] = []
    for term_name in active_terms:
      term_tensor = group_obs[term_name]
      flattened_terms.append(term_tensor.flatten(start_dim=1))
    return torch.cat(flattened_terms, dim=1)

  def _group_flat_dim(self, group_name: str | None) -> int:
    if group_name is None:
      return 0
    term_dims = self.unwrapped.observation_manager.group_obs_term_dim[group_name]
    flat_dim = 0
    for term_dim in term_dims:
      term_flat_dim = 1
      for dim in term_dim:
        term_flat_dim *= int(dim)
      flat_dim += term_flat_dim
    return int(flat_dim)
