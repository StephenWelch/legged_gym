from __future__ import annotations

import numpy as np
import torch
from typing import TypeVar, SupportsFloat, Any, Tuple, Dict, List
from gymnasium import Env
from gymnasium.spaces import Box

from legged_gym.envs.base.legged_robot import LeggedRobot

LeggedObs = TypeVar("LeggedObs")
# LeggedAct = TypeVar("LeggedAct")
LeggedAct = np.ndarray

class LeggedRobotWrapper(Env[LeggedObs, LeggedAct]):
    """
    A Gymnasium-compliant wrapper for a single environment instance inside LeggedRobot.
    """

    def __init__(self, legged_robot: LeggedRobot, env_ids: List[int], device: str):
        self.legged_robot = legged_robot
        self.env_ids = env_ids
        self.device = device

        env_cfg = self.legged_robot.cfg
        assert all([0 <= env_id <= env_cfg.env.num_envs for env_id in env_ids])

        self.env_ids = torch.tensor(self.env_ids, device=self.legged_robot.device).reshape(-1)

        self.observation_space = Box(
            -env_cfg.normalization.clip_observations,
            env_cfg.normalization.clip_observations,
            (env_cfg.env.num_observations,)
        )
        self.action_space = Box(
            -env_cfg.normalization.clip_actions,
            env_cfg.normalization.clip_actions,
            (env_cfg.env.num_actions,)
        )

    def step(self, action: LeggedAct) -> Tuple[LeggedObs, SupportsFloat, bool, bool, Dict[str, Any]]:
        # Move action to environment device
        action = torch.tensor(action, device=self.legged_robot.device)
        action_tensor = torch.zeros((self.legged_robot.num_envs, *self.action_space.shape), device=self.legged_robot.device)
        action_tensor[self.env_ids] = action[range(len(self.env_ids))]
        # Step environment. Privileged obs <-> full environment state
        obs_buf, privileged_obs_buf, reward_buf, reset_buf, extras = self.legged_robot.step(action_tensor)

        obs = obs_buf[self.env_ids].to(self.device)
        reward = reward_buf[self.env_ids].to(self.device)
        term = reset_buf[self.env_ids].to(self.device)
        # Return everything besides the full state
        # TODO implement actual truncation
        return obs, reward, term, torch.zeros_like(term), extras

    def reset(self, *, seed: int | None=None, options: Dict[str, Any] | None=None) -> Tuple[LeggedObs, Dict[str, Any]]:
        # Reset the environment. Takes a *tensor* of environment IDs
        self.legged_robot.reset_idx(self.env_ids)
        self.legged_robot.compute_observations()
        obs = self.legged_robot.obs_buf[self.env_ids].to(self.device)
        # TODO return initial state
        return obs, {}

