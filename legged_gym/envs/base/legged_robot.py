from legged_gym import *
from time import time
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math_utils import wrap_to_pi, torch_rand_float, quat_apply
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params: dict, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, sim_device, headless)
        
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self._pre_sim_step(actions)
        self.simulator.step(actions)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self.simulator.draw_debug_vis() if needed
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.simulator.post_physics_step()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        if self.cfg.sensor.add_depth:
            self.simulator.update_depth_images()
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        if self.debug:
            self.simulator.draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        fail_buf = torch.any(
            torch.norm(self.simulator.link_contact_forces[:, self.simulator.termination_contact_indices, :], dim=-1)
            > 10.0, dim=1)
        # print(f"contact termination: {fail_buf}")
        fail_buf |= self.simulator.projected_gravity[:, 2] > self.cfg.rewards.max_projected_gravity
        # print(f"gravity termination: {self.simulator.projected_gravity[:, 2] > self.cfg.rewards.max_projected_gravity}")
        self.fail_buf += fail_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
        )

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length ==0):
            self._update_command_curriculum(env_ids)

        self._resample_commands(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.simulator.reset_idx(env_ids)

        # reset buffers
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.simulator.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # reset action queue and delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.
            self.action_queue[env_ids] = 0.
            self.action_delay[env_ids] = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                                       self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination(
            ) * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.simulator.base_lin_vel * self.obs_scales.lin_vel,                    # 3
                                    self.simulator.projected_gravity,                                         # 3
                                    self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
                                    self.commands[:, :3] * self.commands_scale,                   # 3
                                    (self.simulator.dof_pos - self.simulator.default_dof_pos) 
                                      * self.obs_scales.dof_pos, # num_dofs
                                    self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
                                    self.actions                                                    # num_actions
                                    ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.simulaor.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - \
                             1) * self.noise_scale_vec

        if self.cfg.domain_rand.randomize_ctrl_delay:
            # normalize to [0, 1]
            ctrl_delay = (self.action_delay /
                          self.cfg.domain_rand.ctrl_delay_step_range[1]).unsqueeze(1)

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
                    self.simulator.dr_friction_values,        # 1
                    self.simulator.dr_added_base_mass,        # 1
                    self.simulator.dr_base_com_bias,          # 3
                    self.simulator.dr_rand_push_vels[:, :2],  # 2
                ),
                dim=-1,
            )
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                    1) - 0.5 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

    def set_viewer_camera(self, pos, lookat):
        """ Set viewer camera position and direction
        """
        self.simulator.set_viewer_camera(eye=pos, target=lookat)

    # ------------- Callbacks (Protected Function) --------------
    
    def _pre_sim_step(self, actions):
        """ Callback called at the beginning of the step function, before stepping the simulation
        """
        clip_actions = self.cfg.normalization.clip_actions
        actions = torch.clip(
            actions, -clip_actions, clip_actions).to(self.device)
        # update history of actions
        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.actions[:] = actions[:]
        # apply action delay by using an action queue
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            actions = self.action_queue[torch.arange(
                self.num_envs), self.action_delay].clone()
        # during training, the camera follows the first environment
        if not self.debug and not self.headless:
            pos = self.simulator.base_pos[0].cpu().numpy() + np.array(self.cfg.viewer.pos)
            lookat = self.simulator.base_pos[0].cpu().numpy() + np.array(self.cfg.viewer.lookat)
            self.set_viewer_camera(pos, lookat)
        
        return actions
    
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.simulator.base_pos[env_ids, :2] - self.simulator.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.simulator._terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(
            self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        
        self.simulator.update_terrain_curriculum(env_ids, move_up, move_down)
    
    def _reset_dofs(self, env_ids):
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_pos[:, :] = self.simulator.default_dof_pos[:] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_actions), self.device)
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
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
        base_lin_vel = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        # base ang vel
        base_ang_vel = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        self.simulator.reset_root_states(env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.simulator.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), self.cfg.commands.ranges.ang_vel_yaw[0], 
                                                                 self.cfg.commands.ranges.ang_vel_yaw[1])

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self.simulator.push_robots()
        
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids),1), self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids),1), self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        # set small commands to zero
        self.commands[env_ids, :3] *= (torch.norm(
            self.commands[env_ids, :3], dim=1) > 0.2).unsqueeze(1)

    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > \
                self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

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
        noise_vec[:3] = noise_scales.lin_vel * \
            noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:24] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.forward_vec[:, 0] = 1.0
        self.fail_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.commands = torch.zeros(
            (self.num_envs, self.cfg.commands.num_commands), device=self.device, dtype=torch.float)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, dtype=torch.float,
                                           requires_grad=False)
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros_like(self.actions)
        self.llast_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # last last actions
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.simulator.feet_indices)), device=self.device, dtype=torch.float)
        self.last_contacts = torch.zeros((self.num_envs, len(self.simulator.feet_indices)), device=self.device, dtype=torch.int)

        # randomize action delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(
                self.num_envs, self.cfg.domain_rand.ctrl_delay_step_range[1]+1, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name =="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.sim.dt * self.cfg.control.decimation
        self.debug = self.cfg.env.debug
        # use self-implemented pd controller
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        # determine privileged observation offset to normalize privileged observations
        self.friction_value_offset = (self.cfg.domain_rand.friction_range[0] + 
                                      self.cfg.domain_rand.friction_range[1]) / 2  # mean value
        self.kp_scale_offset = (self.cfg.domain_rand.kp_range[0] +
                                self.cfg.domain_rand.kp_range[1]) / 2  # mean value
        self.kd_scale_offset = (self.cfg.domain_rand.kd_range[0] +
                                self.cfg.domain_rand.kd_range[1]) / 2  # mean value
        
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.simulator.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.simulator.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.simulator.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.simulator.base_pos[:, 2].unsqueeze(
            1) - self.simulator.measured_heights, dim=1)
        # print(f"base height: {base_height}")
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return rew

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.simulator.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1)
    
    def _reward_dof_power(self):
        # Penalize power consumption
        return torch.sum(torch.abs(self.simulator.torques * self.simulator.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.simulator.last_dof_vel - 
                                       self.simulator.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions - 2*self.last_actions + self.llast_actions), dim=-1)
        return action_smoothness_cost

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        # print(f"contacts: {(torch.norm(self.simulator.link_contact_forces[0, self.simulator.penalized_contact_indices, :], dim=-1) > 0.1)}")
        rew = torch.sum(1.*(torch.norm(
            self.simulator.link_contact_forces[:, self.simulator.penalized_contact_indices, :], 
            dim=-1) > 0.1), dim=1)
        # print(f"collision reward: {rew[0]}")
        return rew

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(self.simulator.torques) - self.simulator.torques_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.simulator.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.simulator.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.simulator.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_dof_vel_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.simulator.dof_vel), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_dof_pos_stand_still(self):
        # Penalize position deviation at zero commands
        return torch.sum(torch.square(self.simulator.dof_pos - self.simulator.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.1)
    
    def _reward_feet_contact_stand_still(self):
        # Encourage feet contact with the ground at zero commands
        contacts = self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2] > 0.1
        full_contact = torch.sum(1.*contacts, dim=1)==len(self.simulator.feet_contact_indices)
        return 1.0*full_contact * (torch.norm(self.commands[:, :3], dim=1) < 0.1)
    
    def _reward_dof_close_to_default(self):
        # Penalize dof position deviation from default
        return torch.sum(torch.square(self.simulator.dof_pos - self.simulator.default_dof_pos), dim=1)

    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        """
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        # print(f"feet pos: {self.simulator.feet_pos[:, :, 2]}")
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.simulator.feet_pos[:, :, 2] -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_foot_landing_vel(self):
        z_vels = self.simulator.feet_vel[:, :, 2]
        contacts = self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2] > 0.1
        about_to_land = ((self.simulator.feet_pos[:, :, 2] -
                          self.cfg.rewards.foot_height_offset) <
                         self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(
            about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward
    
    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
    
    def _reward_foot_acc(self):
        '''reward for foot acceleration'''
        foot_acc = (self.simulator.feet_vel - self.simulator.last_feet_vel) / self.dt
        return torch.sum(torch.square(foot_acc), dim=(1, 2))
