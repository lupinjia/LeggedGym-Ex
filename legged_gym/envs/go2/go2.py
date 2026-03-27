from legged_gym import *
from time import time
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import torch_rand_float
from legged_gym.utils.helpers import class_to_dict
from .go2_config import GO2Cfg

class GO2(LeggedRobot):
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_pos[:, [0, 3, 6, 9]] = self.simulator.default_dof_pos[:, [0, 3, 6, 9]] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), 4), self.device)
        dof_pos[:, [1, 4, 7, 10]] = self.simulator.default_dof_pos[:, [1, 4, 7, 10]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)
        dof_pos[:, [2, 5, 8, 11]] = self.simulator.default_dof_pos[:, [2, 5, 8, 11]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)

        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    # Override functions for deployment
    def compute_observations(self):
        # print(f"project gravity: {self.simulator.projected_gravity}")
        # print(f"base ang vel: {self.simulator.base_ang_vel}")
        # print(f"dof pos: {self.simulator.dof_pos[0]}")
        # print(f"default dof pos: {self.simulator.default_dof_pos}")
        # print(f"dof vel: {self.simulator.dof_vel}")
        # print(f"actions: {self.actions}")
        self.obs_buf = torch.cat((
                                self.commands[:, :3] * self.commands_scale,                   # 3
                                self.simulator.projected_gravity,                             # 3
                                self.simulator.base_ang_vel * self.obs_scales.ang_vel,        # 3
                                (self.simulator.dof_pos - self.simulator.default_dof_pos) 
                                    * self.obs_scales.dof_pos, # num_dofs
                                self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
                                self.actions                                                    # num_actions
                                ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - \
                             1) * self.noise_scale_vec

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                (
                    self.simulator.base_lin_vel * self.obs_scales.lin_vel,
                    self.simulator.base_ang_vel * self.obs_scales.ang_vel,
                    self.simulator.projected_gravity,
                    self.commands[:, :3] * self.commands_scale,
                    (self.simulator.dof_pos - self.simulator.default_dof_pos) * \
                     self.obs_scales.dof_pos,
                    self.simulator.dof_vel * self.obs_scales.dof_vel,
                    self.actions,
                    self.last_actions,
                    self.simulator._friction_values,        # 1
                    self.simulator._added_base_mass,        # 1
                    self.simulator._base_com_bias,          # 3
                    self.simulator._rand_push_vels[:, :2],  # 2
                ),
                dim=-1,
            )
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                    1) - 0.5 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
    
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[9:21] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec
    
    def _reset_root_states(self, env_ids):
        # base pos
        if self.simulator.custom_origins:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
            base_pos[:, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
        # base quat
        base_quat = self.simulator.base_init_quat.reshape(1, -1).repeat(len(env_ids), 1)
        # base lin vel
        base_lin_vel = torch_rand_float(-0.0, 0.0, (len(env_ids), 3), self.device)
        # base ang vel
        base_ang_vel = torch_rand_float(-0.0, 0.0, (len(env_ids), 3), self.device)
        self.simulator.reset_root_states(env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel)