# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class PandoraFlatCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48
        num_actions = 12

    
    class terrain( LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        mesh_type = 'plane'
        measure_heights = False
          
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.38] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'll1_hip_yaw': 0.,
            'll2_hip_rol': 0.,
            'll3_hip_pit': 0.4,
            'll4_kne_pit': -0.8,
            'll5_ank_pit': 0.4,
            'll6_ank_rol': 0.,

            'rl1_hip_yaw': 0.,
            'rl2_hip_rol': 0.,
            'rl3_hip_pit': 0.4,
            'rl4_kne_pit': -0.8,
            'rl5_ank_pit': 0.4,
            'rl6_ank_rol': 0.
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        lbd=10.0
        stiffness = { 
            'll1_hip_yaw': lbd**2,
            'll2_hip_rol': lbd**2,
            'll3_hip_pit': lbd**2,
            'll4_kne_pit': lbd**2,
            'll5_ank_pit': lbd**2/2,
            'll6_ank_rol': lbd**2/2,

            'rl1_hip_yaw': lbd**2,
            'rl2_hip_rol': lbd**2,
            'rl3_hip_pit': lbd**2,
            'rl4_kne_pit': lbd**2,
            'rl5_ank_pit': lbd**2/2,
            'rl6_ank_rol': lbd**2/2
        }  # [N*m/rad]
        damping = {
            'll1_hip_yaw': 2*lbd,
            'll2_hip_rol': 2*lbd,
            'll3_hip_pit': 2*lbd,
            'll4_kne_pit': 2*lbd,
            'll5_ank_pit': 2*lbd,
            'll6_ank_rol': 2*lbd,

            'rl1_hip_yaw': 2*lbd,
            'rl2_hip_rol': 2*lbd,
            'rl3_hip_pit': 2*lbd,
            'rl4_kne_pit': 2*lbd,
            'rl5_ank_pit': 2*lbd,
            'rl6_ank_rol': 2*lbd
        }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        #file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pandora/urdf/pandora_v1_robot.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pandora/urdf/pandora_v1_robot_colission.urdf'
        name = "pandora_v1"
        foot_name = 'foot'
        terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

#Scenario 1 Complete
    # class rewards( LeggedRobotCfg.rewards ):
    #     only_positive_rewards = False
    #     tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    #     soft_dof_pos_limit = 1.
    #     soft_dof_vel_limit = 1.
    #     soft_torque_limit = 1.
    #     base_height_target = 1.
    #     max_contact_force = 500.

    #     class scales( LeggedRobotCfg.rewards.scales ):
    #         termination = -200.
    #         tracking_lin_vel = 1.0
    #         tracking_ang_vel = 1.0
    #         lin_vel_z = -0.5
    #         ang_vel_xy = -0.0
    #         orientation = -0.
    #         torques = -5.e-6
    #         dof_vel = -0.0
    #         dof_acc = -2.e-7
    #         base_height= -0.
    #         feet_air_time = 5.
    #         collision= -1
    #         stumble = -0. 
    #         action_rate = -0.01
    #         stand_still = -0.
    #         feet_contact_forces = -0.
    #         no_fly = 0.25
    #         dof_pos_limits = -1.

#Scenario 2 Basic
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.
        soft_dof_vel_limit = 0.
        soft_torque_limit = 0.
        base_height_target = 0.
        max_contact_force = 50000.

        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.
            lin_vel_z = 0.
            ang_vel_xy = 0.
            orientation = -0.
            torques = -5.e-6
            dof_vel = -0.0
            dof_acc = -0.
            base_height= -0.
            feet_air_time = 0.
            collision= -0.
            stumble = -0. 
            action_rate = -0.
            stand_still = -0.
            feet_contact_forces = -0.
            no_fly = 0.0
            dof_pos_limits = -0.

#Scenario 3 Partial
    # class rewards( LeggedRobotCfg.rewards ):
    #     only_positive_rewards = False
    #     tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    #     soft_dof_pos_limit = 0.
    #     soft_dof_vel_limit = 0.
    #     soft_torque_limit = 0.
    #     base_height_target = 0.
    #     max_contact_force = 50000.

    #     class scales( LeggedRobotCfg.rewards.scales ):
    #         termination = -200.
    #         tracking_lin_vel = 1.0
    #         tracking_ang_vel = 0.
    #         lin_vel_z = 0.
    #         ang_vel_xy = 0.
    #         orientation = -0.
    #         torques = -5.e-6
    #         dof_vel = -0.0
    #         dof_acc = -0.
    #         base_height= -0.
    #         feet_air_time = 0.
    #         collision= -0.
    #         stumble = -0. 
    #         action_rate = -0.
    #         stand_still = -0.
    #         feet_contact_forces = -0.
    #         no_fly = 0.0
    #         dof_pos_limits = -0.

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.


class PandoraFlatCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_pandora'
        max_iterations = 400
        
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        #load_run = "Apr13_10-53-54_"
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        gamma = 0.99



  