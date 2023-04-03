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
        pos = [0.0, 0.0, 1.35] # x,y,z [m]
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
            'll1_hip_yaw': lbd**2/4,
            'll2_hip_rol': lbd**2/4,
            'll3_hip_pit': lbd**2,
            'll4_kne_pit': lbd**2,
            'll5_ank_pit': lbd**2/4,
            'll6_ank_rol': lbd**2/4,

            'rl1_hip_yaw': lbd**2/4,
            'rl2_hip_rol': lbd**2/4,
            'rl3_hip_pit': lbd**2,
            'rl4_kne_pit': lbd**2,
            'rl5_ank_pit': lbd**2/4,
            'rl6_ank_rol': lbd**2/4
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
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pandora/urdf/pandora_v1_robot.urdf'
        name = "pandora_v1"
        foot_name = 'foot'
        terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -300.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.7
            feet_air_time = 2.
            dof_pos_limits = -1.
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.
            stumble = -125.0 

class PandoraFlatCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_pandora'
        max_iterations = 300

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        



  