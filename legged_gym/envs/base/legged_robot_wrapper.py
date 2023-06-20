from __future__ import annotations

import numpy as np
import torch
from typing import TypeVar, SupportsFloat, Any, Tuple, Dict
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

    def __init__(self, legged_robot: LeggedRobot, env_id: int, device: str):
        self.legged_robot = legged_robot
        self.env_id = env_id
        self.device = device

        env_cfg = self.legged_robot.cfg
        assert env_id in range(env_cfg.env.num_envs)

        self.env_ids = torch.tensor(self.env_id, device=self.legged_robot.device).reshape(1)

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
        # Step environment. Privileged obs <-> full environment state
        obs_buf, privileged_obs_buf, reward_buf, reset_buf, extras = self.legged_robot.step(action)

        obs = obs_buf[self.env_id].to(self.device)
        reward = reward_buf[self.env_id].to(self.device)
        term = reset_buf[self.env_id].to(self.device)
        # Return everything besides the full state
        return obs, reward, term, False, extras

    def reset(self, *, seed: int | None=None, options: Dict[str, Any] | None=None) -> Tuple[LeggedObs, Dict[str, Any]]:
        # Reset the environment. Takes a *tensor* of environment IDs
        self.legged_robot.reset_idx(self.env_ids)
        # TODO return initial state
        return torch.zeros(self.observation_space.shape), {}

