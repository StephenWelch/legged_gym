import os
from dataclasses import dataclass

import isaacgym
import torch
import gymnasium as gym
import numpy as np
import omegaconf
import os
import hydra.experimental

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math

from typing import Optional

from mbrl.models import Model
from torch import Tensor

from legged_gym.envs import * # This line is required :)
from legged_gym.utils import get_args, task_registry
from torch.utils.tensorboard import SummaryWriter
import datetime
EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env = LeggedRobotWrapper(env, 0, 'cpu')

    writer = SummaryWriter()

    @dataclass
    class LeggedRobotObs:
        base_lin_vel: Tensor         # (1, 3)
        base_ang_vel: Tensor         # (1, 3)
        projected_gravity: Tensor    # (1, 3)
        commands: Tensor             # (1, 3)
        dof_pos: Tensor              # (1, 12)
        dof_vel: Tensor              # (1, 12)
        actions: Tensor

        @classmethod
        def from_tensor(cls, env: LeggedRobot, tensor: Tensor):
            return LeggedRobotObs(
                tensor[:, 0:3] / env.obs_scales.lin_vel,
                tensor[:, 3:6] / env.obs_scales.ang_vel,
                tensor[:, 6:9],
                tensor[:, 9:12] / env.commands_scale,
                tensor[:, 12:24] / env.obs_scales.dof_pos,
                tensor[:, 24:36] / env.obs_scales.dof_vel,
                tensor[:, 36:48] # TODO check if this should be unnormalized
            )

    def termination_fn(act: Tensor, next_obs: Tensor) -> Tensor:
        unnormalized_obs = LeggedRobotObs.from_tensor(env.legged_robot, next_obs)
        # Check for tipping over - if magnitude X/Y gravity vector acting on body > threshold
        return (torch.norm(unnormalized_obs.projected_gravity[:, :2]) > 0.8).reshape(-1, 1)

    def reward_fn(act: Tensor, next_obs: Tensor) -> Tensor:
        unnormalized_obs = LeggedRobotObs.from_tensor(env.legged_robot, next_obs)
        reward = 0
        reward += (env.legged_robot.cfg.rewards.scales.tracking_lin_vel * _reward_tracking_lin_vel(unnormalized_obs)).reshape(-1, 1)
        reward += (env.legged_robot.cfg.rewards.scales.lin_vel_z * _reward_lin_vel_z(unnormalized_obs)).reshape(-1, 1)
        if termination_fn(act, next_obs):
            reward += (env.legged_robot.cfg.rewards.scales.termination * _reward_termination(unnormalized_obs)).reshape(-1, 1)
        return reward

    def _reward_lin_vel_z(obs: LeggedRobotObs):
        # Penalize z axis base linear velocity
        return torch.square(obs.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(obs: LeggedRobotObs):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(obs.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(obs: LeggedRobotObs):
        # Penalize non flat base orientation
        return torch.sum(torch.square(obs.projected_gravity[:, :2]), dim=1)

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)

    # def _reward_torques(self):
    #     # Penalize torques
    #     return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(obs: LeggedRobotObs):
        # Penalize dof velocities
        return torch.sum(torch.square(obs.dof_vel), dim=1)

    # def _reward_dof_acc(self):
    #     # Penalize dof accelerations
    #     return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # def _reward_collision(self):
    #     # Penalize collisions on selected bodies
    #     return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
    #                      dim=1)

    def _reward_termination(obs: LeggedRobotObs):
        # Terminal reward / penalty
        return torch.ones(env.legged_robot.num_envs, device=env.legged_robot.sim_device)

    def _reward_dof_pos_limits(obs: LeggedRobotObs):
        # Penalize dof positions too close to the limit
        out_of_limits = -(obs.dof_pos - obs.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (obs.dof_pos - obs.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(obs: LeggedRobotObs):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(obs.dof_vel) - obs.dof_vel_limits * env_cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    # def _reward_torque_limits(self):
    #     # penalize torques too close to the limit
    #     return torch.sum(
    #         (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(obs: LeggedRobotObs):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(obs.commands[:, :2] - obs.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / env_cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(obs: LeggedRobotObs):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(obs.commands[:, 2] - obs.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / env_cfg.rewards.tracking_sigma)

    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts)
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact,
    #                             dim=1)  # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime

    # def _reward_stumble(self):
    #     # Penalize feet hitting vertical surfaces
    #     return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
    #                      5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
                    torch.norm(self.commands[:, :2], dim=1) < 0.1)

    # def _reward_feet_contact_forces(self):
    #     # penalize high contact forces
    #     return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
    #                                  dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    hydra.experimental.initialize(config_path="conf", job_name="pets")
    cfg = hydra.experimental.compose(config_name="main")

    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    # obs_shape = (env_cfg.env.num_observations,)
    # act_shape = (env_cfg.env.num_actions,)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # work_dir = work_dir or os.getcwd()
    work_dir = os.getcwd()
    silent = False
    print(f"Results will be saved at {work_dir}.")

    if silent:
        logger = None
    else:
        logger = mbrl.util.Logger(work_dir)
        logger.register_group(
            mbrl.constants.RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green"
        )

    # -------- Create and populate initial env dataset --------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env),
        {},
        replay_buffer=replay_buffer,
    )
    replay_buffer.save(work_dir)

    # ---------------------------------------------------------
    # ---------- Create model environment and agent -----------
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
    )

    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
    )

    def log_tensorboard(model: Model, n_train_iter: int, epoch: int, total_avg_loss: float, val_score: float, best_val_score: Optional[Tensor]):
        writer.add_scalar("train/total_avg_loss", total_avg_loss, cfg.overrides.num_epochs_train_model * n_train_iter + epoch)
        ...

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    current_trial = 0
    max_total_reward = -np.inf
    while env_steps < cfg.overrides.num_steps:
        obs, _ = env.reset()
        agent.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps_trial = 0
        last_env_steps = env_steps
        while not terminated and not truncated:
            # --------------- Model Training -----------------
            if env_steps % cfg.algorithm.freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                    callback=log_tensorboard
                )

            # --- Doing env step using the agent and adding to model dataset ---
            (
                next_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            env_steps += 1

            if debug_mode:
                print(f"Step {env_steps}: Reward {reward:.3f}.")

        if logger is not None:
            logger.log_data(
                mbrl.constants.RESULTS_LOG_NAME,
                {"env_step": env_steps, "episode_reward": total_reward},
            )
            writer.add_scalar("train/episode_reward", total_reward, env_steps)
            writer.add_scalar("train/episode_length", env_steps - last_env_steps, env_steps)
            for name, value in info["episode"].items():
                writer.add_scalar(f"train/{name}", value, env_steps)
        current_trial += 1
        if debug_mode:
            print(f"Trial: {current_trial }, reward: {total_reward}.")

        max_total_reward = max(max_total_reward, total_reward)

    writer.flush()

    return np.float32(max_total_reward)

if __name__ == '__main__':
    args = get_args()
    train(args)
