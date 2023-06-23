import copy
from collections import defaultdict
from os.path import splitext, basename
from typing import Optional, cast, Any, Tuple, Dict

import isaacgym
import torch
import hydra
import os

from mbrl.models import Model
from omegaconf import OmegaConf

import wandb
import mbrl
import mbrl.algorithms.mbpo
import omegaconf
import gymnasium as gym
import numpy as np
from mbrl.algorithms import mbpo
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party import pytorch_sac_pranz24
from mbrl.third_party.pytorch_sac import VideoRecorder
from torch.utils.tensorboard import SummaryWriter

from legged_gym.envs import *  # This line is required :)
from legged_gym.utils import get_args, task_registry

from torch import Tensor
from dataclasses import dataclass

from legged_gym.utils.helpers import resolve_device_names, get_args_from_cfg, Map


@dataclass
class LeggedRobotObs:
    base_lin_vel: Tensor  # (1, 3)
    base_ang_vel: Tensor  # (1, 3)
    projected_gravity: Tensor  # (1, 3)
    commands: Tensor  # (1, 3)
    dof_pos: Tensor  # (1, 12)
    dof_vel: Tensor  # (1, 12)
    actions: Tensor

    # torques: Tensor

    @classmethod
    def from_tensor(cls, env: LeggedRobot, tensor: Tensor):
        return LeggedRobotObs(
            tensor[..., 0:3] / env.obs_scales.lin_vel,
            tensor[..., 3:6] / env.obs_scales.ang_vel,
            tensor[..., 6:9],
            tensor[..., 9:12] / env.commands_scale,
            tensor[..., 12:24] / env.obs_scales.dof_pos,
            tensor[..., 24:36] / env.obs_scales.dof_vel,
            tensor[..., 36:48],  # TODO check if this should be unnormalized
            # tensor[..., 36:48] / env.cfg.control.action_scale  # TODO for torque control mode only
        )


class SacLoggerAdapter:
    def __init__(self, group_name: str = "", step_name: str = "step"):
        self._data: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self._group_name = group_name
        self._step_name = step_name

    def log(self, key: str, value: Any, step: int):
        self._data[step].update({key: value})

    def commit(self):
        for step, data in self._data.items():
            data.update({self._step_name: step})
            wandb.log({self._group_name: data}, commit=False)
        self._data = defaultdict(dict)


@hydra.main(config_path="conf", config_name="mbpo.yaml")
def train(cfg: OmegaConf):
    legged_gym_cfg = get_args_from_cfg(Map(OmegaConf.to_container(cfg)))

    env, env_cfg = task_registry.make_env(name=cfg.task, args=legged_gym_cfg)
    env = LeggedRobotWrapper(env, 0, 'cpu')

    writer = SummaryWriter()
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"{cfg.task}-{cfg.algorithm.name}",

        # track hyperparameters and run metadata
        config=cfg
    )

    def termination_fn(act: Tensor, next_obs: Tensor) -> Tensor:
        unnormalized_obs = LeggedRobotObs.from_tensor(env.legged_robot, next_obs)
        # Check for tipping over - if magnitude X/Y gravity vector acting on body > threshold
        # return (torch.norm(unnormalized_obs.projected_gravity[:, :2], dim=-1) > 0.8).reshape(-1, 1)
        return (unnormalized_obs.dof_pos[:, 2] < 0.33).reshape(-1, 1)

    train_mbpo(env, env, termination_fn, cfg, work_dir=None)


def train_mbpo(
        env: gym.Env,
        test_env: gym.Env,
        termination_fn: mbrl.types.TermFnType,
        cfg: omegaconf.DictConfig,
        silent: bool = False,
        work_dir: Optional[str] = None,
) -> np.float32:
    writer = SummaryWriter()
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )
    wandb_logger = SacLoggerAdapter(group_name="policy")

    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    # logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    # logger.register_group(
    #     mbrl.constants.RESULTS_LOG_NAME,
    #     mbrl.algorithms.mbpo.MBPO_LOG_FORMAT,
    #     color="green",
    #     dump_frequency=1,
    # )
    save_video = cfg.get("save_video", False)
    video_recorder = VideoRecorder(work_dir if save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
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
    random_explore = cfg.algorithm.random_initial_explore
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env) if random_explore else agent,
        {} if random_explore else {"sample": True, "batched": False},
        replay_buffer=replay_buffer,
    )

    def wandb_callback(model: Model, n_train_iter: int, epoch: int, total_avg_loss: int, val_score: float,
                       best_val_score: Optional[Tensor]):
        wandb.log({
            "model/total_avg_loss": total_avg_loss,
            "model/step": cfg.overrides.epoch_length * epoch + n_train_iter,
            "model/epoch": epoch
        }, commit=False)

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
            cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, None, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        # logger=None if silent else logger,
    )
    best_eval_reward = -np.inf
    epoch = 0
    sac_buffer = None
    while env_steps < cfg.overrides.num_steps:
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
        sac_buffer = mbpo.maybe_replace_sac_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
        )
        obs = None
        terminated = False
        truncated = False
        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or terminated or truncated:
                steps_epoch = 0
                obs, _ = env.reset()
                terminated = False
                truncated = False
            # --- Doing env step and adding to model dataset ---
            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            ) = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            # --------------- Model Training -----------------
            if (env_steps + 1) % cfg.overrides.freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                    callback=wandb_callback
                )

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                mbpo.rollout_model_and_populate_sac_buffer(
                    model_env,
                    replay_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                )

                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                use_real_data = rng.random() < cfg.algorithm.real_data_ratio
                which_buffer = replay_buffer if use_real_data else sac_buffer
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                        which_buffer
                ) < cfg.overrides.sac_batch_size:
                    break  # only update every once in a while

                agent.sac_agent.update_parameters(
                    which_buffer,
                    cfg.overrides.sac_batch_size,
                    updates_made,
                    wandb_logger,
                    reverse_mask=True,
                )
                updates_made += 1
                # if not silent and updates_made % cfg.log_frequency_agent == 0:
                #     logger.dump(updates_made, save=True)
                wandb_logger.log("rollout_length", rollout_length, updates_made)
                wandb_logger.log("sac_buffer_size", len(sac_buffer), updates_made)
                wandb_logger.commit()

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                avg_reward = mbpo.evaluate(
                    test_env, agent, cfg.algorithm.num_eval_episodes, video_recorder
                )
                # logger.log_data(
                #     mbrl.constants.RESULTS_LOG_NAME,
                #     {
                #         "epoch": epoch,
                #         "env_step": env_steps,
                #         "episode_reward": avg_reward,
                #         "rollout_length": rollout_length,
                #     },
                # )

                # writer.add_scalar("eval/episode_reward", avg_reward, env_steps)
                wandb.log({
                    "eval/episode_reward": avg_reward,
                })

                if avg_reward > best_eval_reward:
                    video_recorder.save(f"{epoch}.mp4")
                    best_eval_reward = avg_reward
                    agent.sac_agent.save_checkpoint(
                        ckpt_path=os.path.join(work_dir, "sac.pth")
                    )
                epoch += 1

            env_steps += 1
            obs = next_obs

    wandb.finish()

    return np.float32(best_eval_reward)


if __name__ == '__main__':
    train()
