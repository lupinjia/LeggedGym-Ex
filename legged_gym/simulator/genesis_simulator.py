from legged_gym import *
from legged_gym.simulator.simulator import Simulator
from PIL import Image as im
import torch
import numpy as np
import os
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import *
import warp as wp
import trimesh
from legged_gym.warp.warp_cam import WarpCam
if SIMULATOR == "genesis":
    import genesis as gs

    #----- Runtime monkey-patch for Genesis GPU device bug -----
    # Genesis's `sanitize_index` (genesis/utils/misc.py) force-converts
    # every input tensor to `gs.device` (CPU). When the simulation runs
    # on CUDA, the CPU index is later used in `scatter_` on CUDA tensors,
    # causing a RuntimeError. We monkey-patch `sanitize_index` at import
    # time so that already-constructed torch.Tensor indices keep their
    # original device. 
    try:
        import genesis.utils.misc as _gs_misc
        _orig_sanitize_index = _gs_misc.sanitize_index

        def _patched_sanitize_index(index, expected_size, max_size, dim, name):
            if isinstance(index, torch.Tensor):
                # Preserve device for already-materialised tensors.
                # Convert dtype only; do NOT move to gs.device.
                index = index.to(dtype=gs.tc_int)
            else:
                # Fallback for numpy arrays, lists, ints, etc.
                index = torch.as_tensor(index, dtype=gs.tc_int, device=gs.device)

            ndim = index.ndim
            if ndim == 0:
                index = index[None]
            elif ndim > 1:
                dim_info = f" `{name}`" if name else ""
                gs.raise_exception(
                    f"Invalid shape: {index.shape}. Expecting 0D or 1D tensor for {dim}-th index{dim_info}."
                )

            if expected_size != -1 and expected_size != len(index):
                dim_info = f" `{name}`" if name else ""
                gs.raise_exception(
                    f"Invalid shape: {index.shape}. Expecting 1D tensor of length {expected_size} for {dim}-th index{dim_info}."
                )
            return index.contiguous()

        _gs_misc.sanitize_index = _patched_sanitize_index
    except Exception as _e:
        import warnings
        warnings.warn(f"Failed to monkey-patch genesis.utils.misc.sanitize_index: {_e}")

    # Additional safety-net: patch torch.scatter_ / masked_scatter_ so that
    # cross-device indices are silently moved to the target tensor's device.
    # This compensates for any Genesis code paths that bypass sanitize_index.
    try:
        _orig_scatter = torch.Tensor.scatter_
        def _patched_scatter(self, dim, index, src_or_value):
            if isinstance(index, torch.Tensor) and index.device != self.device:
                index = index.to(self.device)
            return _orig_scatter(self, dim, index, src_or_value)
        torch.Tensor.scatter_ = _patched_scatter
    except Exception as _e:
        import warnings
        warnings.warn(f"Failed to monkey-patch torch.Tensor.scatter_: {_e}")

    try:
        _orig_masked_scatter = torch.Tensor.masked_scatter_
        def _patched_masked_scatter(self, mask, source):
            if isinstance(mask, torch.Tensor) and mask.device != self.device:
                mask = mask.to(self.device)
            return _orig_masked_scatter(self, mask, source)
        torch.Tensor.masked_scatter_ = _patched_masked_scatter
    except Exception as _e:
        import warnings
        warnings.warn(f"Failed to monkey-patch torch.Tensor.masked_scatter_: {_e}")

""" ********** Genesis Simulator ********** """
class GenesisSimulator(Simulator):
    def __init__(self, cfg, sim_params: dict, device, headless):
        self._sim_params = sim_params
        super().__init__(cfg, sim_params, device, headless)
        # warp init
        if self._cfg.sensor.add_depth:
            wp.init()
            self._create_warp_envs()
            self._create_warp_tensors()
            self._depth_camera_sensor = WarpCam(self._warp_tensor_dict, 
                                self._num_camera_envs, 
                                self._cfg.sensor, 
                                self._mesh_ids, 
                                self._device)
            pixels = self._depth_camera_sensor.update()
            self._depth_images[:,0] = pixels[:,0] # pixels: [num_envs, num_sensors, H, W]

    #----- Public methods -----#
    def step(self, actions):
        self._last_base_lin_vel[:] = self._base_lin_vel[:]
        self._last_base_ang_vel[:] = self._base_ang_vel[:]
        self._last_feet_vel[:] = self._feet_vel[:]
        actions = self._pre_simulator_step(actions)
        for _ in range(self._cfg.control.decimation):
            self._last_dof_vel[:] = self._dof_vel[:]
            self._torques = self._compute_torques(actions)
            self._robot.control_dofs_force(
                self._torques, self._dof_indices)
            self._scene.step()
            # dof_pos and dof_vel use joint sequence of policy
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
        self._base_quat[:] = quat_wxyz_to_xyzw(self._base_quat_gs)
        self._base_euler[:] = get_euler_xyz(self._base_quat)
        self._base_lin_vel[:] = quat_rotate_inverse(self._base_quat, self._robot.get_vel().to(self._device))
        self._base_ang_vel[:] = quat_rotate_inverse(self._base_quat, self._robot.get_ang().to(self._device))
        self._projected_gravity = quat_rotate_inverse(self._base_quat, self._global_gravity)
        self._dof_pos[:] = self._robot.get_dofs_position(self._dof_indices)
        self._dof_vel[:] = self._robot.get_dofs_velocity(self._dof_indices)
        self._link_contact_forces[:] = self._robot.get_links_net_contact_force()
        self._feet_pos[:] = self._robot.get_links_pos()[:, self._feet_indices, :]
        self._feet_vel[:] = self._robot.get_links_vel()[:, self._feet_indices, :]
        self._key_body_pos[:] = self._robot.get_links_pos()[:, self._key_body_indices, :]
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
        
        # reset action queue and delay
        if self._cfg.domain_rand.randomize_ctrl_delay:
            self._action_queue[env_ids] *= 0.
            self._action_queue[env_ids] = 0.
            self._action_delay[env_ids] = torch.randint(self._cfg.domain_rand.ctrl_delay_step_range[0],
                                                       self._cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self._device, requires_grad=False)

        # reset depth image tensors
        # find common ids between env_ids and camera env ids
        if self._cfg.sensor.add_depth:
            camera_env_ids = torch.arange(self._num_camera_envs, device=self._device)
            common_env_ids = torch.tensor(
                [env_id for env_id in env_ids.tolist() if env_id in camera_env_ids.tolist()],
                device=self._device, dtype=torch.long)
            self._depth_images[common_env_ids, :] = 0.
        
        
    def reset_dofs(self, env_ids, dof_pos, dof_vel):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        env_ids = env_ids.to(self._device)

        self._dof_pos[env_ids] = dof_pos[:]
        self._dof_vel[env_ids] = dof_vel[:]
        
        self._robot.set_dofs_position(
            position=self._dof_pos[env_ids],
            dofs_idx_local=self._dof_indices,
            zero_velocity=True,
            envs_idx=env_ids,
        )

    def reset_root_states(self,
                          env_ids,
                          base_pos,
                          base_quat,
                          base_lin_vel_w,
                          base_ang_vel_w):
        env_ids = env_ids.to(self._device)

        # base pos
        self._base_pos[env_ids, :] = base_pos[:]
        self._robot.set_pos(
            self._base_pos[env_ids], zero_velocity=False, envs_idx=env_ids)

        # base quat
        self._base_quat[env_ids, :] = base_quat[:]
        self._base_quat_gs[env_ids] = quat_xyzw_to_wxyz(self._base_quat[env_ids])  # wxyz
        self._robot.set_quat(
            self._base_quat_gs[env_ids], zero_velocity=False, envs_idx=env_ids)
        self._robot.zero_all_dofs_velocity(env_ids)

        # update projected gravity
        self._projected_gravity = quat_rotate_inverse(self._base_quat, self._global_gravity)

        # reset root states - velocity
        base_vel = torch.concat(
            [base_lin_vel_w, base_ang_vel_w], dim=1)
        self._robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=[
                                     0, 1, 2, 3, 4, 5], envs_idx=env_ids)
        self._base_lin_vel[env_ids] = quat_rotate_inverse(self._base_quat[env_ids], self._robot.get_vel().to(self._device)[env_ids])
        self._base_ang_vel[env_ids] = quat_rotate_inverse(self._base_quat[env_ids], self._robot.get_ang().to(self._device)[env_ids])

    def update_sensors(self):
        # Genesis currently exposes depth update via `update_depth_images`
        if self._cfg.sensor.add_depth:
            if self._depth_image_update_counter % self._depth_image_update_decimation == 0:
                self._update_depth_camera()
            self._depth_image_update_counter += 1

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
        dofs_vel = self._robot.get_dofs_velocity().to(self._device)  # (num_envs, num_dof) [0:3] ~ base_link_vel
        push_vel = torch_rand_float(-max_push_vel_xy,
                                     max_push_vel_xy, (self._num_envs, 2), self._device)
        self._rand_push_vels[:, :2] = push_vel.detach().clone()
        dofs_vel[:, :2] += push_vel
        self._robot.set_dofs_velocity(dofs_vel)
        self._last_base_lin_vel[:] = self._base_lin_vel[:]
        self._base_lin_vel[:] = quat_rotate_inverse(
            self._base_quat, self._robot.get_vel().to(self._device))
    
    def push_links(self):
        max_force = self._cfg.domain_rand.max_push_force
        # apply random forces to the links of the robot
        push_force = torch.rand((self._num_envs, self._robot.n_links, 3), 
                                device=self._device) * 2 * max_force - max_force
        all_link_idx = [link.idx - self._robot.link_start for link in self._robot.links]
        self._scene.sim.rigid_solver.apply_links_external_force(
            push_force,
            links_idx=all_link_idx
        )

    def draw_debug_vis(self,
                       ref_key_body_pos=None):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        self._scene.clear_debug_objects()
        if self._cfg.env.debug_draw_key_body_points:
            self._draw_key_body_points(ref_key_body_pos)
        if self._cfg.env.debug_draw_depth_images:
            self._draw_debug_depth_images()
          
    def set_viewer_camera(self, eye: np.ndarray, target: np.ndarray):
        self._scene.viewer.set_camera_pose(pos=eye, lookat=target)
        
    def calc_feet_near_edge(self):
        """ Calculate whether each foot is near the terrain edge, which can be used as a termination condition or for reward shaping
        Returns:
            torch.tensor: whether each foot is near the terrain edge, shape: (num_envs, num_feet)
        """
        if self._cfg.terrain.mesh_type == 'plane':
            return torch.zeros((self._num_envs, len(self._feet_indices)), device=self._device, dtype=torch.bool)
        elif self._cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't calculate feet near edge with terrain mesh type 'none'")
        
        feet_pos_xy = self._feet_pos[:, :, :2] # (num_envs, num_feet, 2)
        feet_points_float = feet_pos_xy + self._cfg.terrain.border_size # add border size to align the origin with heightfield raw
        points = (feet_points_float/self._cfg.terrain.horizontal_scale).long()
        px = points[:,:, 0]
        py = points[:,:, 1]
        px = torch.clip(px, 0, self._edge_mask.shape[0]-1)
        py = torch.clip(py, 0, self._edge_mask.shape[1]-1)
        
        # if any of the four points around the foot is an edge point
        # and the position of the foot is within 5 cm to the edge point, we consider the foot to be near the edge
        edge_threshold = self._cfg.rewards.feet_edge_threshold
        foot_near_edge = torch.zeros((self._num_envs, len(self._feet_indices)), device=self._device, dtype=torch.bool)
        for i in range(-1, 2):
            for j in range(-1, 2):
                edge_points = self._edge_mask[torch.clip(px+i, 0, self._edge_mask.shape[0]-1), torch.clip(py+j, 0, self._edge_mask.shape[1]-1)] == 1
                if edge_points.any():
                    edge_points_pos = torch.zeros((self._num_envs, len(self._feet_indices), 2), device=self._device)
                    edge_points_pos[:,:,0] = (torch.clip(px+i, 0, self._edge_mask.shape[0]-1).float() * self._cfg.terrain.horizontal_scale) - self._cfg.terrain.border_size
                    edge_points_pos[:,:,1] = (torch.clip(py+j, 0, self._edge_mask.shape[1]-1).float() * self._cfg.terrain.horizontal_scale) - self._cfg.terrain.border_size
                    foot_near_edge = foot_near_edge | (edge_points & (torch.norm(feet_pos_xy - edge_points_pos, dim=-1) < edge_threshold))
        
        return foot_near_edge
    
    #----- Protected methods -----#
    def _pre_simulator_step(self, actions):
        # apply action delay by using an action queue
        if self._cfg.domain_rand.randomize_ctrl_delay:
            self._action_queue[:, 1:] = self._action_queue[:, :-1].clone()
            self._action_queue[:, 0] = actions.clone()
            actions = self._action_queue[torch.arange(
                self._num_envs), self._action_delay].clone()
            
        return actions
        
    def _parse_cfg(self):
        self._debug = self._cfg.env.debug
        self._control_dt = self._cfg.sim.dt * self._cfg.control.decimation
        self._batch_dofs_links_info = self._cfg.domain_rand.randomize_joint_armature or \
                self._cfg.domain_rand.randomize_joint_friction or \
                self._cfg.domain_rand.randomize_joint_damping
        if self._cfg.sensor.add_depth:
            self._num_camera_envs = self._cfg.env.num_camera_envs
            # update counter for depth images
            self._depth_image_update_counter = 0
            self._depth_image_update_decimation = self._cfg.sensor.depth_camera_config.decimation
    
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
        if self._cfg.asset.xml_file != "":
            asset_path = self._cfg.asset.xml_file.format(
                LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            asset_root = os.path.dirname(asset_path)
            asset_file = os.path.basename(asset_path)

            self._robot = self._scene.add_entity(
                gs.morphs.MJCF(
                    file=os.path.join(asset_root, asset_file),
                    pos=np.array(self._cfg.init_state.pos),
                    quat=np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz
                )
            )
        else:
            raise NotImplementedError("Please specify xml file path for Genesis simulator!")
        
        # add camera if needed
        if self._cfg.sensor.add_depth:
            self._setup_depth_camera()
        
        # build
        if self._headless:
            # Monkey-patch the visualizer to skip building
            original_build = self._scene.build
            def patched_build(n_envs):
                # Store the original visualizer build method
                original_visualizer_build = self._scene._visualizer.build
                # Temporarily replace it with a no-op
                self._scene._visualizer.build = lambda: None
                # Build the scene
                result = original_build(n_envs)
                # Restore the original method
                self._scene._visualizer.build = original_visualizer_build
                return result
            # Apply the patch
            self._scene.build = patched_build
        # Build the scene
        self._scene.build(n_envs=self._num_envs)

        self._get_env_origins()

        self._dof_names = self._cfg.asset.dof_names
        self._num_dof = len(self._cfg.asset.dof_names)

        # name to indices
        self._dof_indices = [self._robot.get_joint(
            name).dof_start for name in self._cfg.asset.dof_names]
        print(f"motor dof indices: {self._dof_indices}")
        
        # find indices of links specified in the config
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
        self._key_body_indices = find_link_indices(self._cfg.asset.key_bodies)
        print(f"key body link indices: {self._key_body_indices}")
        self._base_link_index = self._robot.base_link_idx - self._robot.link_start
        print(f"base link index: {self._base_link_index}")
        
        if self._cfg.asset.obtain_link_contact_states:
            self._contact_state_link_indices = find_link_indices(
                self._cfg.asset.contact_state_link_names
            )

        # dof position limits
        self._dof_pos_limits = torch.stack(
            self._robot.get_dofs_limit(self._dof_indices), dim=1).to(self._device)
        # Genesis don't provide api for accessing vel limits, so we set it here
        if hasattr(self._cfg.asset, "dof_vel_limits"):
            self._dof_vel_limits = torch.tensor(self._cfg.asset.dof_vel_limits, device=self._device).unsqueeze(0)
        self._torque_limits = self._robot.get_dofs_force_range(self._dof_indices)[
            1].to(self._device)
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
        # randomize friction (only at startup; doing it on reset slows down training)
        if self._cfg.domain_rand.randomize_friction:
            self._randomize_friction(np.arange(self._num_envs))
        # randomize restitution (only at startup; doing it on reset slows down training)
        if self._cfg.domain_rand.randomize_restitution:
            self._randomize_restitution(torch.arange(self._num_envs))
        # randomize base mass (only at startup; doing it on reset slows down training)
        if self._cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self._num_envs))
        # randomize COM displacement (only at startup; doing it on reset slows down training)
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
        self._key_body_pos = torch.zeros(
            (self._num_envs, len(self._key_body_indices), 3), device=self._device, dtype=torch.float
        )
        self._last_feet_vel = torch.zeros_like(self._feet_vel)
        
        # Camera image tensor
        if self._cfg.sensor.add_depth:
            pointcloud_dims = 3 * (self._cfg.sensor.depth_camera_config.return_pointcloud == True)
            if pointcloud_dims > 0: # pointcloud returned by depth camera
                self._depth_images = torch.zeros(
                    (self._num_camera_envs, 
                    self._cfg.sensor.depth_camera_config.num_history,
                    *self._cfg.sensor.depth_camera_config.resolution,
                    pointcloud_dims), 
                    device=self._device, 
                    dtype=torch.float
                )
            else:
                self._depth_images = torch.zeros(
                    (self._num_camera_envs, 
                    self._cfg.sensor.depth_camera_config.num_history,
                    *self._cfg.sensor.depth_camera_config.resolution), 
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
        
        # default_dof_pos use joint sequence of policy
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

        # p_gains use joint sequence of policy
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
        
        # control delay
        if self._cfg.domain_rand.randomize_ctrl_delay:
            self._action_queue = torch.zeros(
                self._num_envs, self._cfg.domain_rand.ctrl_delay_step_range[1]+1, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
            self._action_delay = torch.randint(self._cfg.domain_rand.ctrl_delay_step_range[0],
                                              self._cfg.domain_rand.ctrl_delay_step_range[1]+1, (self._num_envs,), device=self._device, requires_grad=False)

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
            self._terrain_types = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
            if hasattr(self._cfg.env, "num_camera_envs"): # split envs with cameras to all terrain types, ensuring the diversity of data collected by cameras
                self._terrain_types[:self._cfg.env.num_camera_envs] = torch.div(
                    torch.arange(self._cfg.env.num_camera_envs, device=self._device), (self._cfg.env.num_camera_envs/self._cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
                self._terrain_types[self._cfg.env.num_camera_envs:] = torch.div(
                    torch.arange(self._num_envs - self._cfg.env.num_camera_envs, device=self._device), ((self._num_envs - self._cfg.env.num_camera_envs)/self._cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            else:
                self._terrain_types = torch.div(torch.arange(self._num_envs, device=self._device), (self._num_envs/self._cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
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
        env_ids = out_of_bound_buf.nonzero(as_tuple=False).flatten().to(self._device)
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
        env_ids = torch.as_tensor(env_ids, device=self._device)
        min_friction, max_friction = self._cfg.domain_rand.friction_range

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, self._robot.n_links) \
            * (max_friction - min_friction) + min_friction
        self._friction_values[env_ids] = ratios[:,
                                                0].unsqueeze(1).detach().clone()

        self._robot.set_friction_ratio(
            ratios, torch.arange(0, self._robot.n_links, device=self._device), env_ids)

    def _randomize_restitution(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self._device)
        return super()._randomize_restitution(env_ids)

    def _randomize_base_mass(self, env_ids=None):
        ''' Randomize base mass'''
        env_ids = torch.as_tensor(env_ids, device=self._device)
        min_mass, max_mass = self._cfg.domain_rand.added_mass_range
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * \
            (max_mass - min_mass) + min_mass
        self._added_base_mass[env_ids] = added_mass[:].detach().clone()
        self._robot.set_mass_shift(added_mass, self._base_link_index, env_ids)

    def _randomize_com_displacement(self, env_ids):
        ''' Randomize center of mass displacement of the robot'''
        env_ids = torch.as_tensor(env_ids, device=self._device)
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
        env_ids = torch.as_tensor(env_ids, device=self._device)
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
        env_ids = torch.as_tensor(env_ids, device=self._device)
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
        env_ids = torch.as_tensor(env_ids, device=self._device)
        min_damping, max_damping = self._cfg.domain_rand.joint_damping_range
        damping = torch.rand((len(env_ids),), dtype=torch.float, device=self._device) \
            * (max_damping - min_damping) + min_damping
        self._joint_damping[env_ids, 0] = damping.detach().clone()
        damping = damping.unsqueeze(1).repeat(1, self._num_actions)
        self._robot.set_dofs_damping(
            damping, self._dof_indices, envs_idx=env_ids)

    def _randomize_pd_gain(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self._device)
        self._kp_scale[env_ids] = torch_rand_float(
                self._cfg.domain_rand.kp_range[0], self._cfg.domain_rand.kp_range[1], (len(env_ids), self._num_actions), device=self._device)
        self._kd_scale[env_ids] = torch_rand_float(
                self._cfg.domain_rand.kd_range[0], self._cfg.domain_rand.kd_range[1], (len(env_ids), self._num_actions), device=self._device)
    
    def _update_depth_camera(self):
        """ Renders the depth camera and retrieves the depth images
        """
        near_clip = self._cfg.sensor.depth_camera_config.near_clip
        far_clip = self._cfg.sensor.depth_camera_config.far_clip
        pixels = self._depth_camera_sensor.update().clone()
        if self._depth_images.shape[1] > 1: # stack history of depth images
            self._depth_images[:, 1:] = self._depth_images[:, :-1].detach().clone()
        # store values for denoised depth images
        self._depth_images[:, 0] = pixels[:,0,:,:] # pixels: [num_envs, num_sensors, H, W]
        # clip values
        self._depth_images[:, 0] = torch.clip(self._depth_images[:, 0], near_clip, far_clip)
        # normalize the depth images to be within [-0.5, 0.5]
        self._depth_images[:, 0] = (self._depth_images[:, 0] - near_clip) / (far_clip - near_clip) - 0.5
    
    def _create_warp_envs(self):
      # extract terrain mesh
      terrain_mesh = self._gs_terrain.geoms[0].get_trimesh()
      
      #save terrain mesh
      transform = np.zeros((3,))
      transform[0] = -self._cfg.terrain.border_size 
      transform[1] = -self._cfg.terrain.border_size
      transform[2] = 0.0
      translation = trimesh.transformations.translation_matrix(transform)
      terrain_mesh.apply_transform(translation)
      
      vertices = terrain_mesh.vertices
      triangles = terrain_mesh.faces
      vertex_tensor = torch.tensor(vertices, 
                                    dtype=torch.float32, 
                                    device=self._device,
                                    requires_grad=False)
      #if none type in vertex_tensor
      if vertex_tensor.any() is None:
          print("vertex_tensor is None")
      vertex_vec3_array = wp.from_torch(vertex_tensor,dtype=wp.vec3)        
      faces_wp_int32_array = wp.from_numpy(triangles.flatten(), dtype=wp.int32,device=self._device)
      
      # mesh coordinate convention may be the same as isaacgym
      self._wp_meshes =  wp.Mesh(points=vertex_vec3_array,indices=faces_wp_int32_array)
      
      Warning("Currently, only static terrain mesh is added to Warp.")
      self._mesh_ids = self.mesh_ids_array = wp.array([self._wp_meshes.id], dtype=wp.uint64)
    
    def _create_warp_tensors(self):
        self._warp_tensor_dict={}
        pointcloud_dims = 3 * (self._cfg.sensor.depth_camera_config.return_pointcloud == True)
        if pointcloud_dims > 0:
            self._depth_image_tensor_warp = torch.zeros((self._num_camera_envs, 
                                                        self._cfg.sensor.depth_camera_config.num_sensors,
                                                        *self._cfg.sensor.depth_camera_config.resolution,
                                                        pointcloud_dims),    # xyz
                                                       dtype=torch.float32, device=self._device)
        else:
            self._depth_image_tensor_warp = torch.zeros((self._num_camera_envs, 
                                                        self._cfg.sensor.depth_camera_config.num_sensors,
                                                        *self._cfg.sensor.depth_camera_config.resolution),
                                                    dtype=torch.float32, device=self._device)
        self._sensor_pos_tensor = torch.zeros_like(self._base_pos[:self._num_camera_envs, :3])
        self._sensor_quat_tensor = torch.zeros_like(self._base_quat[:self._num_camera_envs, :4])

        # sensor pose
        pos_offset = [self._cfg.sensor.depth_camera_config.pos[0], 
                      self._cfg.sensor.depth_camera_config.pos[1], 
                      self._cfg.sensor.depth_camera_config.pos[2]]
        rpy_offset = [self._cfg.sensor.depth_camera_config.euler[0], 
                      self._cfg.sensor.depth_camera_config.euler[1], 
                      self._cfg.sensor.depth_camera_config.euler[2]]
        self._sensor_offset_pos = torch.tensor(pos_offset, device=self._device).repeat((self._num_camera_envs, 1))
        rpy_offset = torch.tensor(rpy_offset, device=self._device).repeat((self._num_camera_envs, 1))
        
        # apply domain randomization to sensor position and sensor rotation
        if self._cfg.domain_rand.randomize_camera_pos:
          self._camera_pos_offset = torch.zeros(
                self._num_camera_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
          self._camera_pos_offset[:self._num_camera_envs, 0] = torch_rand_float(
                -self._cfg.domain_rand.camera_com_displacement_range[0],
                self._cfg.domain_rand.camera_com_displacement_range[0],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          self._camera_pos_offset[:self._num_camera_envs, 1] = torch_rand_float(
                -self._cfg.domain_rand.camera_com_displacement_range[1],
                self._cfg.domain_rand.camera_com_displacement_range[1],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          self._camera_pos_offset[:self._num_camera_envs, 2] = torch_rand_float(
                -self._cfg.domain_rand.camera_com_displacement_range[2],
                self._cfg.domain_rand.camera_com_displacement_range[2],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          self._sensor_offset_pos += self._camera_pos_offset[:self._num_camera_envs]
          
        if self._cfg.domain_rand.randomize_camera_euler:
          self._camera_euler_offset = torch.zeros(
                self._num_camera_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
          self._camera_euler_offset[:self._num_camera_envs, 0] = torch_rand_float(
                -self._cfg.domain_rand.camera_euler_offset_range[0],
                self._cfg.domain_rand.camera_euler_offset_range[0],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          self._camera_euler_offset[:self._num_camera_envs, 1] = torch_rand_float(
                -self._cfg.domain_rand.camera_euler_offset_range[1],
                self._cfg.domain_rand.camera_euler_offset_range[1],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          self._camera_euler_offset[:self._num_camera_envs, 2] = torch_rand_float(
                -self._cfg.domain_rand.camera_euler_offset_range[2],
                self._cfg.domain_rand.camera_euler_offset_range[2],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          rpy_offset += self._camera_euler_offset[:self._num_camera_envs]
          
        self._sensor_offset_quat = quat_from_euler_xyz(rpy_offset[:, 0], 
                                                       rpy_offset[:, 1], 
                                                       rpy_offset[:, 2])
        
        self._warp_tensor_dict["depth_image_tensor"] = self._depth_image_tensor_warp
        self._warp_tensor_dict['device'] = self._device
        self._warp_tensor_dict['num_envs'] = self._num_camera_envs
        self._warp_tensor_dict['num_sensors'] = self._cfg.sensor.depth_camera_config.num_sensors
        self._warp_tensor_dict['sensor_pos_tensor'] = self._sensor_pos_tensor
        self._warp_tensor_dict['sensor_quat_tensor'] = self._sensor_quat_tensor
        self._warp_tensor_dict['mesh_ids'] = self._mesh_ids
    
    def _draw_debug_depth_images(self):
        if self._cfg.sensor.depth_camera_config.calculate_depth:
            depth = self._depth_images[0, 0, :, :]
            # print(f"depth values: {depth}")
            # far_clip = self._cfg.sensor.depth_camera_config.far_clip
            # near_clip = self._cfg.sensor.depth_camera_config.near_clip
            pixel_values = ((depth + 0.5) * 255.0).cpu().numpy().astype(np.uint8)
            # image = im.fromarray(pixel_values, mode='L')
            # image.save("debug_depth_images/depth_frame%d.jpg" % self.frame_count)
            import cv2 as cv
            cv.imshow("Depth Camera", pixel_values)
            cv.waitKey(1)
            # pixel_values = (depth + 0.5) * (far_clip - near_clip) + near_clip
            # pixel_values = pixel_values.cpu().numpy().astype(np.float32)
            # print(f"depth pixel values: {pixel_values}")
        elif self._cfg.sensor.depth_camera_config.return_pointcloud:
            pointcloud = self._depth_images[0, 0, :, :, :]  # get pointcloud of first env, first step
            pointcloud_np = pointcloud.view(-1, 3)
            self._scene.draw_debug_spheres(pointcloud_np, radius=0.01, color=(0, 1, 1, 1))
            # print(f"pointcloud values: {pointcloud_np}")
    
    def _draw_key_body_points(self, ref_key_body_pos=None):
        """ Draws key body points for debugging
        """
        if ref_key_body_pos is not None:
            self._scene.draw_debug_spheres(ref_key_body_pos.view(-1, 3), radius=0.03, color=(1, 0, 0, 1))
        else:
            pass
            
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
        self._edge_mask = torch.tensor(self._terrain.edge_mask).view(self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)
    
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

    #----- Properties -----#
    @property
    def feet_contact_indices(self):
        """Returns the indices of the feet links in the contact sensors.

        Returns:
            list[int]: Indices of the feet links in the contact sensors.
        """
        return self._feet_indices
    
    @property
    def feet_quat(self):
        feet_quat = self._robot.get_links_quat()[:, self._feet_indices, :].to(self._device)
        return feet_quat[..., [3, 0, 1, 2]] # from wxyz to xyzw