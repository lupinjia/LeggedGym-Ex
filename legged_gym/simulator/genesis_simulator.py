from legged_gym import *
from legged_gym.simulator.simulator import Simulator
from PIL import Image as im
import cv2 as cv
import torch
import numpy as np
import os
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import *
if SIMULATOR == "genesis":
    import genesis as gs

""" ********** Genesis Simulator ********** """
class GenesisSimulator(Simulator):
    def __init__(self, cfg, sim_params: dict, device, headless):
        self._sim_params = sim_params
        super().__init__(cfg, sim_params, device, headless)
    
    #----- Public methods -----#
    def step(self, actions):
        self._last_base_lin_vel[:] = self._base_lin_vel[:]
        self._last_base_ang_vel[:] = self._base_ang_vel[:]
        self._last_feet_vel[:] = self._feet_vel[:]
        self._last_dof_vel[:] = self._dof_vel[:]
        for _ in range(self._cfg.control.decimation):
            self._torques = self._compute_torques(actions)
            self._robot.control_dofs_force(
                self._torques, self._dof_indices)
            self._scene.step()
            self._dof_pos[:] = self._robot.get_dofs_position(
                self._dof_indices)
            self._dof_vel[:] = self._robot.get_dofs_velocity(
                self._dof_indices)

    def post_physics_step(self):
        # prepare quantities
        self._base_pos[:] = self._robot.get_pos()
        self._check_base_pos_out_of_bound()       # check if the pos of the robot is out of terrain bounds
        self._base_pos[:] = self._robot.get_pos()
        self._base_quat_gs[:] = self._robot.get_quat()
        self._base_quat[:,-1] = self._robot.get_quat()[:,0]   # wxyz to xyzw
        self._base_quat[:,:3] = self._robot.get_quat()[:,1:4] # wxyz to xyzw
        self._base_euler[:] = get_euler_xyz(self._base_quat)
        self._base_lin_vel[:] = quat_rotate_inverse(self._base_quat, self._robot.get_vel())
        self._base_ang_vel[:] = quat_rotate_inverse(self._base_quat, self._robot.get_ang())
        self._projected_gravity = quat_rotate_inverse(self._base_quat, self._global_gravity)
        self._dof_pos[:] = self._robot.get_dofs_position(self._dof_indices)
        self._dof_vel[:] = self._robot.get_dofs_velocity(self._dof_indices)
        self._link_contact_forces[:] = self._robot.get_links_net_contact_force()
        self._feet_pos[:] = self._robot.get_links_pos()[:, self._feet_indices, :]
        self._feet_vel[:] = self._robot.get_links_vel()[:, self._feet_indices, :]
        # Link contact state
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = 1. * (torch.norm(
                self._link_contact_forces[:, self._contact_state_link_indices, :], dim=-1) > 1.)
        # update terrain heights info
        if self._cfg.terrain.measure_heights:
            self._update_surrounding_heights()
            if self._cfg.terrain.obtain_terrain_info_around_feet:
                self._calc_terrain_info_around_feet()
        
    def reset_idx(self, env_ids):
        # domain randomization
        if self._cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        if self._cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)
        if self._cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(env_ids)
        if self._cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(env_ids)
        if self._cfg.domain_rand.randomize_joint_friction:
            self._randomize_joint_friction(env_ids)
        if self._cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(env_ids)
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(env_ids)
        
        self._last_dof_vel[env_ids] = 0.
        self._last_feet_vel[env_ids] = 0.
        self._last_base_lin_vel[env_ids] = 0.
        self._last_base_ang_vel[env_ids] = 0.

    def reset_dofs(self, env_ids, dof_pos, dof_vel):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        self._dof_pos[env_ids] = dof_pos[:]
        self._dof_vel[env_ids] = dof_vel[:]
        
        self._robot.set_dofs_position(
            position=self._dof_pos[env_ids],
            dofs_idx_local=self._dof_indices,
            zero_velocity=True,
            envs_idx=env_ids,
        )
        self._robot.zero_all_dofs_velocity(env_ids)

    def reset_root_states(self, env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base pos
        self._base_pos[env_ids, :] = base_pos[:]
        self._robot.set_pos(
            self._base_pos[env_ids], zero_velocity=False, envs_idx=env_ids)

        # base quat
        self._base_quat[env_ids, :] = base_quat[:]
        self._base_quat_gs[env_ids, 0] = self._base_quat[env_ids, 3]  # xyzw to wxyz
        self._base_quat_gs[env_ids, 1:4] = self._base_quat[env_ids, 0:3] # xyzw to wxyz
        self._robot.set_quat(
            self._base_quat_gs[env_ids], zero_velocity=False, envs_idx=env_ids)
        self._robot.zero_all_dofs_velocity(env_ids)

        # update projected gravity
        self._projected_gravity = quat_rotate_inverse(self._base_quat, self._global_gravity)

        # reset root states - velocity
        self._base_lin_vel[env_ids] = base_lin_vel[:]
        self._base_ang_vel[env_ids] = base_ang_vel[:]
        base_vel = torch.concat(
            [self._base_lin_vel[env_ids], self._base_ang_vel[env_ids]], dim=1)
        self._robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=[
                                     0, 1, 2, 3, 4, 5], envs_idx=env_ids)

    def update_sensors(self):
        # Genesis currently exposes depth update via `update_depth_images`
        if self._cfg.sensor.add_depth:
            return self._update_depth_images()

    def update_terrain_curriculum(self, env_ids, move_up, move_down):
        self._terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self._terrain_levels[env_ids] = torch.where(self._terrain_levels[env_ids] >= self._max_terrain_level,
                                                   torch.randint_like(
                                                       self._terrain_levels[env_ids], self._max_terrain_level),
                                                   torch.clip(self._terrain_levels[env_ids], 0))  # (the minumum level is zero)
        self._env_origins[env_ids] = self._terrain_origins[self._terrain_levels[env_ids],
            self._terrain_types[env_ids]]

    def push_robots(self):
        max_push_vel_xy = self._cfg.domain_rand.max_push_vel_xy
        # in Genesis, base link also has DOF, it's 6DOF if not fixed.
        dofs_vel = self._robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
        push_vel = torch_rand_float(-max_push_vel_xy,
                                     max_push_vel_xy, (self._num_envs, 2), self._device)
        self._rand_push_vels[:, :2] = push_vel.detach().clone()
        dofs_vel[:, :2] += push_vel
        self._robot.set_dofs_velocity(dofs_vel)

    def draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height points
        if not self._cfg.terrain.measure_heights:
            return
        self._scene.clear_debug_objects()
        
        # # Height points around feet
        # height_points = torch.zeros(self._num_envs, 9*len(self._feet_indices), 3, device=self._device)
        # foot_points = self._feet_pos + self._cfg.terrain.border_size
        # foot_points = (foot_points/self._cfg.terrain.horizontal_scale).long()
        # px = foot_points[:, :, 0].view(-1)
        # py = foot_points[:, :, 1].view(-1)
        # heights1 = self._height_samples[px-1, py]  # [x-0.1, y]
        # heights2 = self._height_samples[px+1, py]  # [x+0.1, y]
        # heights3 = self._height_samples[px, py-1]  # [x, y-0.1]
        # heights4 = self._height_samples[px, py+1]  # [x, y+0.1]
        # heights5 = self._height_samples[px, py]    # [x, y]
        # heights6 = self._height_samples[px-1, py-1]  # [x-0.1, y-0.1]
        # heights7 = self._height_samples[px+1, py+1]  # [x+0.1, y+0.1]
        # heights8 = self._height_samples[px-1, py+1]  # [x-0.1, y+0.1]
        # heights9 = self._height_samples[px+1, py-1]  # [x+0.1, y-0.1]
        # for i in range(len(self._feet_indices)):
        #     height_points[0, i*9+0, 0] = (px-1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+0, 1] = (py-1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+0, 2] = heights6.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
        #     height_points[0, i*9+1, 0] = (px-1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+1, 1] = py.view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+1, 2] = heights1.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
        #     height_points[0, i*9+2, 0] = px.view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+2, 1] = (py-1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+2, 2] = heights3.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
        #     height_points[0, i*9+3, 0] = px.view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+3, 1] = (py+1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+3, 2] = heights4.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
        #     height_points[0, i*9+4, 0] = px.view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+4, 1] = py.view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+4, 2] = heights5.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
        #     height_points[0, i*9+5, 0] = (px+1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+5, 1] = py.view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+5, 2] = heights2.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
        #     height_points[0, i*9+6, 0] = (px+1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+6, 1] = (py+1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+6, 2] = heights7.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
        #     height_points[0, i*9+7, 0] = (px-1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+7, 1] = (py+1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+7, 2] = heights8.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
        #     height_points[0, i*9+8, 0] = (px+1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+8, 1] = (py-1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
        #     height_points[0, i*9+8, 2] = heights9.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
        
        # # print(f"shape of height_points: ", height_points.shape) # (num_envs, num_points, 3)
        # self._scene.draw_debug_spheres(height_points[0, :], radius=0.02, color=(1, 0, 0, 0.7))  # only draw for the first env

    def set_viewer_camera(self, eye: np.ndarray, target: np.ndarray):
        self._scene.viewer.set_camera_pose(pos=eye, lookat=target)
    
    #----- Protected methods -----#
    def _parse_cfg(self):
        self._debug = self._cfg.env.debug
        self._control_dt = self._cfg.sim.dt * self._cfg.control.decimation
        self._batch_dofs_links_info = self._cfg.domain_rand.randomize_joint_armature or \
                self._cfg.domain_rand.randomize_joint_friction or \
                self._cfg.domain_rand.randomize_joint_damping
        if self._cfg.sensor.add_depth:
            self.frame_count = 0
    
    def _create_sim(self):
        # create scene
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self._sim_params["dt"],
                substeps=self._sim_params["substeps"]),
            viewer_options=gs.options.ViewerOptions(
                # max_FPS=int(1 / self._control_dt * self._cfg.control.decimation),
                camera_pos=np.array(self._cfg.viewer.pos),
                camera_lookat=np.array(self._cfg.viewer.lookat),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=self._cfg.viewer.rendered_envs_idx,
                shadow=False,
                ),
            rigid_options=gs.options.RigidOptions(
                dt=self._sim_params["dt"],
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=not self._cfg.asset.self_collisions,
                max_collision_pairs=self._cfg.sim.max_collision_pairs,
                IK_max_targets=self._cfg.sim.IK_max_targets,
                batch_dofs_info=self._batch_dofs_links_info,
                batch_links_info=self._batch_dofs_links_info,
            ),
            show_viewer=not self._headless,
        )

        # add terrain
        mesh_type = self._cfg.terrain.mesh_type
        if mesh_type == 'plane':
            self._gs_terrain = self._scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/plane/plane.urdf", 
                    fixed=True)
                )
            env_width = self._cfg.terrain.terrain_length
            env_length = self._cfg.terrain.terrain_width
            border = int(self._cfg.terrain.border_size/self._cfg.terrain.horizontal_scale)
            width_per_env_pixels = int(env_width / self._cfg.terrain.horizontal_scale)
            length_per_env_pixels = int(env_length / self._cfg.terrain.horizontal_scale)
            tot_cols = int(self._cfg.terrain.num_cols * width_per_env_pixels) + 2 * border
            tot_rows = int(self._cfg.terrain.num_rows * length_per_env_pixels) + 2 * border
            self._height_samples = torch.zeros((tot_rows, tot_cols), dtype=torch.int16)
        elif mesh_type == 'heightfield':
            self._terrain = Terrain(self._cfg.terrain)
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            raise NotImplementedError("Trimesh terrain is not validated yet in Genesis, please use heightfield for now.")
            self._terrain = Terrain(self._cfg.terrain)
            self._create_trimesh()
        else:
            raise ValueError(f"Unsupported terrain mesh type: {mesh_type}")
        self._gs_terrain.set_friction(self._cfg.terrain.static_friction)
        # specify the boundary of the heightfield
        self._terrain_x_range = torch.zeros(2, device=self._device)
        self._terrain_y_range = torch.zeros(2, device=self._device)
        if self._cfg.terrain.mesh_type in ['heightfield', 'trimesh']:
            # give a small margin(1.0m)
            self._terrain_x_range[0] = -self._cfg.terrain.border_size + 1.0
            self._terrain_x_range[1] = self._cfg.terrain.border_size + \
                self._cfg.terrain.num_rows * self._cfg.terrain.terrain_length - 1.0
            self._terrain_y_range[0] = -self._cfg.terrain.border_size + 1.0
            self._terrain_y_range[1] = self._cfg.terrain.border_size + \
                self._cfg.terrain.num_cols * self._cfg.terrain.terrain_width - 1.0
        elif self._cfg.terrain.mesh_type == 'plane':  # the plane used has limited size,
            # and the origin of the world is at the center of the plane
            self._terrain_x_range[0] = -self._cfg.terrain.plane_length/2+1
            self._terrain_x_range[1] = self._cfg.terrain.plane_length/2-1
            # the plane is a square
            self._terrain_y_range[0] = -self._cfg.terrain.plane_length/2+1
            self._terrain_y_range[1] = self._cfg.terrain.plane_length/2-1

    def _create_envs(self):
        # Create envs
        asset_path = self._cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        self._robot = self._scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(asset_root, asset_file),
                merge_fixed_links=True,
                links_to_keep=self._cfg.asset.links_to_keep,
                pos=np.array(self._cfg.init_state.pos),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz
                fixed=self._cfg.asset.fix_base_link,
            ),
            # visualize_contact=self._debug,
        )
        
        # add camera if needed
        if self._cfg.sensor.add_depth:
            self._setup_depth_camera()
        
        # build
        self._scene.build(n_envs=self._num_envs)

        self._get_env_origins()

        self._dof_names = self._cfg.asset.dof_names
        self._num_dof = len(self._cfg.asset.dof_names)

        # name to indices
        self._dof_indices = [self._robot.get_joint(
            name).dof_start for name in self._cfg.asset.dof_names]
        print(f"motor dof indices: {self._dof_indices}")
        
        # find link indices, termination links, penalized links, and feet
        def find_link_indices(names):
            link_indices = list()
            for link in self._robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self._robot.link_start)
            return link_indices

        self._termination_contact_indices = find_link_indices(
            self._cfg.asset.terminate_after_contacts_on)
        all_link_names = [link.name for link in self._robot.links]
        print(f"all link names: {all_link_names}")
        print("termination link indices:", self._termination_contact_indices)
        self._penalized_contact_indices = find_link_indices(
            self._cfg.asset.penalize_contacts_on)
        print(f"penalized link indices: {self._penalized_contact_indices}")
        self._feet_names = [
            link.name for link in self._robot.links if self._cfg.asset.foot_name in link.name]
        self._feet_indices = find_link_indices(self._feet_names)
        print(f"feet names: {self._feet_names}, feet link indices: {self._feet_indices}")
        assert len(self._feet_indices) > 0
        self._base_link_index = self._robot.base_link_idx - self._robot.link_start
        print(f"base link index: {self._base_link_index}")
        
        if self._cfg.asset.obtain_link_contact_states:
            self._contact_state_link_indices = find_link_indices(
                self._cfg.asset.contact_state_link_names
            )

        # dof position limits
        self._dof_pos_limits = torch.stack(
            self._robot.get_dofs_limit(self._dof_indices), dim=1)
        # Genesis don't provide api for accessing vel limits, so we set it here
        if hasattr(self._cfg.asset, "dof_vel_limits"):
            self._dof_vel_limits = torch.tensor(self._cfg.asset.dof_vel_limits, device=self._device).unsqueeze(0)
        self._torque_limits = self._robot.get_dofs_force_range(self._dof_indices)[
            1]
        for i in range(self._dof_pos_limits.shape[0]):
            # soft limits
            m = (self._dof_pos_limits[i, 0] + self._dof_pos_limits[i, 1]) / 2
            r = self._dof_pos_limits[i, 1] - self._dof_pos_limits[i, 0]
            self._dof_pos_limits[i, 0] = (
                m - 0.5 * r * self._cfg.rewards.soft_dof_pos_limit
            )
            self._dof_pos_limits[i, 1] = (
                m + 0.5 * r * self._cfg.rewards.soft_dof_pos_limit
            )
            
        self._init_domain_params()
        # randomize friction
        if self._cfg.domain_rand.randomize_friction:
            self._randomize_friction(np.arange(self._num_envs))
        # randomize base mass
        if self._cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self._num_envs))
        # randomize COM displacement
        if self._cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(np.arange(self._num_envs))
        # randomize joint armature
        if self._cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(np.arange(self._num_envs))
        # randomize joint friction
        if self._cfg.domain_rand.randomize_joint_friction:
            self._randomize_joint_friction(np.arange(self._num_envs))
        # randomize joint damping
        if self._cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(np.arange(self._num_envs))
        # randomize pd gain
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(np.arange(self._num_envs))
            
    def _init_buffers(self):
        self._base_init_pos = torch.tensor(
            self._cfg.init_state.pos, device=self._device
        )
        self._base_init_quat = torch.tensor(
            self._cfg.init_state.rot, device=self._device
        )
        self._base_lin_vel = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._base_ang_vel = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._last_base_lin_vel = torch.zeros_like(self._base_lin_vel)
        self._last_base_ang_vel = torch.zeros_like(self._base_ang_vel)
        self._projected_gravity = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self._device, dtype=torch.float).repeat(
            self._num_envs, 1
        )
        self._dof_pos = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._dof_vel = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._last_dof_vel = torch.zeros_like(self._dof_vel)
        self._base_pos = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._base_quat = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float)
        self._base_quat_gs = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float) # quaternion in genesis definition, wxyz
        self._base_euler = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._link_contact_forces = torch.zeros(
            (self._num_envs, self._robot.n_links, 3), device=self._device, dtype=torch.float
        )
        self._feet_pos = torch.zeros(
            (self._num_envs, len(self._feet_indices), 3), device=self._device, dtype=torch.float
        )
        self._feet_vel = torch.zeros(
            (self._num_envs, len(self._feet_indices), 3), device=self._device, dtype=torch.float
        )
        self._last_feet_vel = torch.zeros_like(self._feet_vel)
        # depth images
        if self._cfg.sensor.add_depth:
            self.depth_images = torch.zeros(
                (self._num_envs, 
                 self._cfg.sensor.depth_camera_config.num_history,
                 self._cfg.sensor.depth_camera_config.resolution[1], 
                 self._cfg.sensor.depth_camera_config.resolution[0]), 
                device=self._device, 
                dtype=torch.float
            )
        
        # Terrain information around feet
        if self._cfg.terrain.obtain_terrain_info_around_feet:
            self._normal_vector_around_feet = torch.zeros(
                self._num_envs, len(self._feet_indices) * 3, dtype=torch.float, device=self._device, requires_grad=False)
            self._height_around_feet = torch.zeros(
                self._num_envs, len(self._feet_indices), 9, dtype=torch.float, device=self._device, requires_grad=False)
        
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = torch.zeros(
                self._num_envs, len(self._contact_state_link_indices), dtype=torch.float, device=self._device, requires_grad=False)
        
        self._default_dof_pos = torch.tensor(
            [self._cfg.init_state.default_joint_angles[name]
                for name in self._cfg.asset.dof_names],
            device=self._device,
            dtype=torch.float,
        )
        self._default_dof_pos = self._default_dof_pos.unsqueeze(0)
        # PD control
        stiffness = self._cfg.control.stiffness
        damping = self._cfg.control.damping

        self._p_gains, self._d_gains = [], []
        for dof_name in self._cfg.asset.dof_names:
            for key in stiffness.keys():
                if key in dof_name:
                    self._p_gains.append(stiffness[key])
                    self._d_gains.append(damping[key])
        self._p_gains = torch.tensor(self._p_gains, device=self._device)
        self._d_gains = torch.tensor(self._d_gains, device=self._device)
        if self._batch_dofs_links_info:   
            self._p_gains = self._p_gains[None, :].repeat(self._num_envs, 1)
            self._d_gains = self._d_gains[None, :].repeat(self._num_envs, 1)
        self._robot.set_dofs_kp(self._p_gains, self._dof_indices)
        self._robot.set_dofs_kv(self._d_gains, self._dof_indices)

        self._init_height_points()
        self._measured_heights = torch.zeros(self._num_envs, self._num_height_points, device=self._device, requires_grad=False)
    
    def _init_height_points(self):
        y = torch.tensor(self._cfg.terrain.measured_points_y,
                         device=self._device, requires_grad=False)
        x = torch.tensor(self._cfg.terrain.measured_points_x,
                         device=self._device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self._num_height_points = grid_x.numel()
        self._height_points = torch.zeros(self._num_envs, self._num_height_points,
                             3, device=self._device, requires_grad=False)
        self._height_points[:, :, 0] = grid_x.flatten()
        self._height_points[:, :, 1] = grid_y.flatten()
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self._cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self._custom_origins = True
            self._env_origins = torch.zeros(
                self._num_envs, 3, device=self._device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self._cfg.terrain.max_init_terrain_level
            if not self._cfg.terrain.curriculum:
                max_init_level = self._cfg.terrain.num_rows - 1
            self._terrain_levels = torch.randint(
                0, max_init_level+1, (self._num_envs,), device=self._device)
            self._terrain_types = torch.div(torch.arange(self._num_envs, device=self._device), (
                self._num_envs/self._cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self._max_terrain_level = self._cfg.terrain.num_rows
            self._terrain_origins = torch.from_numpy(
                self._terrain.env_origins).to(self._device).to(torch.float)
            self._env_origins[:] = self._terrain_origins[self._terrain_levels,
                                                       self._terrain_types]
        else:
            self._custom_origins = False
            self._env_origins = torch.zeros(
                self._num_envs, 3, device=self._device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self._num_envs))
            num_rows = np.ceil(self._num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(
                num_rows), torch.arange(num_cols), indexing='ij')
            # plane has limited size, we need to specify spacing base on num_envs, to make sure all robots are within the plane
            # restrict envs to a square of [plane_length/2, plane_length/2]
            spacing = self._cfg.env.env_spacing
            if num_rows * self._cfg.env.env_spacing > self._cfg.terrain.plane_length / 2 or \
                    num_cols * self._cfg.env.env_spacing > self._cfg.terrain.plane_length / 2:
                spacing = min((self._cfg.terrain.plane_length / 2) / (num_rows-1),
                              (self._cfg.terrain.plane_length / 2) / (num_cols-1))
            self._env_origins[:, 0] = spacing * xx.flatten()[:self._num_envs]
            self._env_origins[:, 1] = spacing * yy.flatten()[:self._num_envs]
            self._env_origins[:, 2] = 0.
            self._env_origins[:, 0] -= self._cfg.terrain.plane_length / 4
            self._env_origins[:, 1] -= self._cfg.terrain.plane_length / 4

    def _update_surrounding_heights(self):
        if self._cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self._num_envs, self._num_height_points, device=self._device, requires_grad=False)
        elif self._cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        points = quat_apply_yaw(self._base_quat.repeat(
                1, self._num_height_points), self._height_points) + (self._base_pos[:, :3]).unsqueeze(1)

        # When acquiring heights, the points need to add border_size
        # because in the height_samples, the origin of the terrain is at (border_size, border_size)
        points += self._cfg.terrain.border_size
        points = (points/self._cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self._height_samples.shape[0]-2)
        py = torch.clip(py, 0, self._height_samples.shape[1]-2)

        heights1 = self._height_samples[px, py]
        heights2 = self._height_samples[px+1, py]
        heights3 = self._height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        self._measured_heights = heights.view(self._num_envs, -1) * self._cfg.terrain.vertical_scale
    
    def _calc_terrain_info_around_feet(self):
        """ Finds neighboring points around each foot for terrain height measurement."""
        # Foot positions
        foot_points = self._feet_pos + self._cfg.terrain.border_size
        foot_points = (foot_points/self._cfg.terrain.horizontal_scale).long()
        # px and py for 4 feet, num_envs*len(feet_indices)
        px = foot_points[:, :, 0].view(-1)
        py = foot_points[:, :, 1].view(-1)
        # clip to the range of height samples
        px = torch.clip(px, 0, self._height_samples.shape[0]-2)
        py = torch.clip(py, 0, self._height_samples.shape[1]-2)
        # get heights around the feet, 9 points for each foot
        heights1 = self._height_samples[px-1, py]  # [x-0.1, y]
        heights2 = self._height_samples[px+1, py]  # [x+0.1, y]
        heights3 = self._height_samples[px, py-1]  # [x, y-0.1]
        heights4 = self._height_samples[px, py+1]  # [x, y+0.1]
        heights5 = self._height_samples[px, py]    # [x, y]
        heights6 = self._height_samples[px-1, py-1]  # [x-0.1, y-0.1]
        heights7 = self._height_samples[px+1, py+1]  # [x+0.1, y+0.1]
        heights8 = self._height_samples[px-1, py+1]  # [x-0.1, y+0.1]
        heights9 = self._height_samples[px+1, py-1]  # [x+0.1, y-0.1]
        # Calculate normal vectors around feet
        dx = ((heights2 - heights1) / (self._cfg.terrain.horizontal_scale * 2)).view(self._num_envs, -1)
        dy = ((heights4 - heights3) / (self._cfg.terrain.horizontal_scale * 2)).view(self._num_envs, -1)
        for i in range(len(self._feet_indices)):
            normal_vector = torch.cat((dx[:, i].unsqueeze(1), dy[:, i].unsqueeze(1), 
                -1*torch.ones_like(dx[:, i].unsqueeze(1))), dim=-1).to(self._device)
            normal_vector /= torch.norm(normal_vector, dim=-1, keepdim=True)
            self._normal_vector_around_feet[:, i*3:i*3+3] = normal_vector[:]
        # Calculate height around feet
        for i in range(9):
            self._height_around_feet[:, :, i] = eval(f'heights{i+1}').view(self._num_envs, -1)[:] * self._cfg.terrain.vertical_scale

    def _check_base_pos_out_of_bound(self):
        """ Check if the base position is out of the terrain bounds
        """
        x_out_of_bound = (self._base_pos[:, 0] >= self._terrain_x_range[1]) | (
            self._base_pos[:, 0] <= self._terrain_x_range[0])
        y_out_of_bound = (self._base_pos[:, 1] >= self._terrain_y_range[1]) | (
            self._base_pos[:, 1] <= self._terrain_y_range[0])
        out_of_bound_buf = x_out_of_bound | y_out_of_bound
        env_ids = out_of_bound_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        else:
            # reset base position to initial position
            self._base_pos[env_ids] = self.base_init_pos
            self._base_pos[env_ids] += self._env_origins[env_ids]
            self._robot.set_pos(
                self._base_pos[env_ids], zero_velocity=False, envs_idx=env_ids)

    def _compute_torques(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self._cfg.control.action_scale
        # get two dimensional gains
        if self._p_gains.ndim == 1:
            self._p_gains = self._p_gains.unsqueeze(0).repeat(self._num_envs, 1)
            self._d_gains = self._d_gains.unsqueeze(0).repeat(self._num_envs, 1)
        torques = (
            self._kp_scale * self._p_gains * (actions_scaled +
                                    self._default_dof_pos - self._dof_pos)
            - self._kd_scale * self._d_gains * self._dof_vel
        )
        return torques

    def _init_domain_params(self):
        """ Initializes domain randomization parameters, which are used to randomize the environment."""
        self._friction_values = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._added_base_mass = torch.ones(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._rand_push_vels = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self._base_com_bias = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_armature = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_friction = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_damping = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._kp_scale = torch.ones(
            self._num_envs, self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)
        self._kd_scale = torch.ones(
            self._num_envs, self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)

    def _randomize_friction(self, env_ids=None):
        ''' Randomize friction of all links'''
        min_friction, max_friction = self._cfg.domain_rand.friction_range

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, self._robot.n_links) \
            * (max_friction - min_friction) + min_friction
        self._friction_values[env_ids] = ratios[:,
                                                0].unsqueeze(1).detach().clone()

        self._robot.set_friction_ratio(
            ratios, torch.arange(0, self._robot.n_links), env_ids)

    def _randomize_base_mass(self, env_ids=None):
        ''' Randomize base mass'''
        min_mass, max_mass = self._cfg.domain_rand.added_mass_range
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * \
            (max_mass - min_mass) + min_mass
        self._added_base_mass[env_ids] = added_mass[:].detach().clone()
        self._robot.set_mass_shift(added_mass, self._base_link_index, env_ids)

    def _randomize_com_displacement(self, env_ids):
        ''' Randomize center of mass displacement of the robot'''
        min_displacement_x, max_displacement_x = self._cfg.domain_rand.com_pos_x_range
        min_displacement_y, max_displacement_y = self._cfg.domain_rand.com_pos_y_range
        min_displacement_z, max_displacement_z = self._cfg.domain_rand.com_pos_z_range
        com_displacement = torch.zeros((len(env_ids), 1, 3), dtype=torch.float, device=self._device)

        com_displacement[:, 0, 0] = gs.rand((len(env_ids), 1), dtype=float).squeeze(1) \
            * (max_displacement_x - min_displacement_x) + min_displacement_x
        com_displacement[:, 0, 1] = gs.rand((len(env_ids), 1), dtype=float).squeeze(1) \
            * (max_displacement_y - min_displacement_y) + min_displacement_y
        com_displacement[:, 0, 2] = gs.rand((len(env_ids), 1), dtype=float).squeeze(1) \
            * (max_displacement_z - min_displacement_z) + min_displacement_z
        self._base_com_bias[env_ids] = com_displacement[:,
                                                        0, :].detach().clone()

        self._robot.set_COM_shift(
            com_displacement, self._base_link_index, env_ids)

    def _randomize_joint_armature(self, env_ids):
        min_armature, max_armature = self._cfg.domain_rand.joint_armature_range
        armature = torch.rand((len(env_ids),), dtype=torch.float, device=self._device) \
            * (max_armature - min_armature) + min_armature
        self._joint_armature[env_ids, 0] = armature.detach().clone()
        # [len(env_ids)] -> [len(env_ids), num_actions], all joints within an env have the same armature
        armature = armature.unsqueeze(1).repeat(1, self._num_actions)
        self._robot.set_dofs_armature(
            armature, self._dof_indices, envs_idx=env_ids) 
        # This armature will be Refreshed when envs are reset

    def _randomize_joint_friction(self, env_ids):
        min_friction, max_friction = self._cfg.domain_rand.joint_friction_range
        friction = torch.rand((len(env_ids),), dtype=torch.float, device=self._device) \
            * (max_friction - min_friction) + min_friction
        self._joint_friction[env_ids, 0] = friction.detach().clone()
        friction = friction.unsqueeze(1).repeat(1, self._num_actions)
        self._robot.set_dofs_frictionloss(
            friction, self._dof_indices, envs_idx=env_ids)

    def _randomize_joint_damping(self, env_ids):
        """ Randomize joint damping of the robot
        """
        min_damping, max_damping = self._cfg.domain_rand.joint_damping_range
        damping = torch.rand((len(env_ids),), dtype=torch.float, device=self._device) \
            * (max_damping - min_damping) + min_damping
        self._joint_damping[env_ids, 0] = damping.detach().clone()
        damping = damping.unsqueeze(1).repeat(1, self._num_actions)
        self._robot.set_dofs_damping(
            damping, self._dof_indices, envs_idx=env_ids)

    def _randomize_pd_gain(self, env_ids):
        self._kp_scale[env_ids] = torch_rand_float(
                self._cfg.domain_rand.kp_range[0], self._cfg.domain_rand.kp_range[1], (len(env_ids), self._num_actions), device=self._device)
        self._kd_scale[env_ids] = torch_rand_float(
                self._cfg.domain_rand.kd_range[0], self._cfg.domain_rand.kd_range[1], (len(env_ids), self._num_actions), device=self._device)
    
    def _update_depth_images(self):
        """ Renders the depth camera and retrieves the depth images
        """
        self.depth_images[:] = self.depth_camera.read_image()[:]
        near_clip = self._cfg.sensor.depth_camera_config.near_clip
        far_clip = self._cfg.sensor.depth_camera_config.far_clip
        # clip the depth images to be within near and far clip
        self.depth_images = torch.clip(self.depth_images, near_clip, far_clip)
        # normalize the depth images to be within 0-1
        self.depth_images = (self.depth_images - near_clip) / (far_clip - near_clip) - 0.5
    
    def _draw_debug_depth_images(self):
        if self._num_envs == 1:
            depth = self.depth_images
        else:
            depth = self.depth_images[0]
        if self._cfg.sensor.depth_camera_config.calculate_depth:
            pixel_values = ((depth + 0.5) * 255.0).cpu().numpy().astype(np.uint8)
            image = im.fromarray(pixel_values, mode='L')
            image.save("debug_depth_images/depth_frame%d.jpg" % self.frame_count)
            # cv.imshow("Depth Camera", (255 * normalized_depth.cpu().numpy()).astype(np.uint8))
            # cv.waitKey(1)
            self.frame_count += 1
            
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        self._gs_terrain = self._scene.add_entity(
            gs.morphs.Terrain(
                pos=(-self._cfg.terrain.border_size, - \
                     self._cfg.terrain.border_size, 0.0),
                horizontal_scale=self._cfg.terrain.horizontal_scale,
                vertical_scale=self._cfg.terrain.vertical_scale,
                height_field=self._terrain.height_field_raw,
            ),
        )
        self._height_samples = torch.tensor(self._terrain.heightsamples).view(
            self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)
    
    def _create_trimesh(self):
        """ Adds a trimesh terrain to the simulation, sets parameters based on the cfg.
        """
        # export terrain mesh to {LEGGED_GYM_ROOT_DIR}/resources/terrains/trimesh_terrain.stl
        trimesh_terrain_path = os.path.join(LEGGED_GYM_ROOT_DIR, "resources", "terrains", "trimesh_terrain.stl")
        self._terrain.terrain_mesh.export(trimesh_terrain_path)
        print(f"Exported terrain mesh to {trimesh_terrain_path}")
        
        # add terrain to the scene
        self._gs_terrain = self._scene.add_entity(
            gs.morphs.Mesh(
                file=trimesh_terrain_path,
                pos=(-self._cfg.terrain.border_size,
                     -self._cfg.terrain.border_size, 
                     0.0),
                fixed=True,
                convexify=False,
            ),
        )
        # save height samples for height sampling
        self._height_samples = torch.tensor(self._terrain.heightsamples).view(
            self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)

    def _setup_depth_camera(self):
        ''' Set camera position and direction
        '''
        depth_pattern = gs.sensors.DepthCameraPattern(
            res=self._cfg.sensor.depth_camera_config.resolution,
            fov_horizontal=self._cfg.sensor.depth_camera_config.fov_horizontal,
        )
        sensor_kwargs = dict(
            entity_idx=self._robot.idx,
            pos_offset=self._cfg.sensor.depth_camera_config.pos,
            euler_offset=self._cfg.sensor.depth_camera_config.euler,
            return_world_frame=False,
            draw_debug=self._debug,
            min_range=self._cfg.sensor.depth_camera_config.near_plane,
            max_range=self._cfg.sensor.depth_camera_config.far_plane,
        )
        self.depth_camera = self._scene.add_sensor(gs.sensors.DepthCamera(pattern=depth_pattern, **sensor_kwargs))


    #----- Properties -----#
    @property
    def feet_contact_indices(self):
        """Returns the indices of the feet links in the contact sensors.

        Returns:
            list[int]: Indices of the feet links in the contact sensors.
        """
        return self._feet_indices