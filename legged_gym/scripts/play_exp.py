from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *

import numpy as np
import torch
from legged_gym.scripts.joystick import Joystick
from legged_gym.utils.exp_data_logger import ExpLogger
import argparse

def _normalize_gpu_arg(gpu):
    gpu = str(gpu).strip().lower()
    if gpu.isdigit():
        return f"cuda:{gpu}"
    if gpu == "cuda":
        return gpu
    if gpu.startswith("cuda:"):
        index = gpu.split(":", 1)[1]
        if index.isdigit():
            return f"cuda:{index}"
    raise ValueError(
        f"Unsupported GPU specifier '{gpu}'. Use values like 'cuda', 'cuda:0', or '1'."
    )


def configure_runtime_device(args):
    """Normalize GPU selection and, when needed, mask visibility to the requested physical GPU.

    Genesis and some CUDA codepaths may still resolve work onto the process-local `cuda:0`.
    When a specific physical GPU is requested, we remap visibility so that local `cuda:0`
    corresponds to the requested GPU.
    """
    if getattr(args, "cpu", False):
        if hasattr(args, "gpu"):
            args.gpu = "cpu"
        if hasattr(args, "device"):
            args.device = "cpu"
        if hasattr(args, "requested_gpu"):
            args.requested_gpu = "cpu"
        return args

    requested_gpu = getattr(args, "requested_gpu", None)
    if requested_gpu is None:
        requested_gpu = getattr(args, "gpu", None)
    if requested_gpu is None:
        requested_gpu = getattr(args, "device", "cuda:0")

    requested_gpu = _normalize_gpu_arg(requested_gpu)
    runtime_gpu = requested_gpu

    if requested_gpu.startswith("cuda:"):
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            visible_gpu_ids = [gpu_id.strip() for gpu_id in visible_devices.split(",") if gpu_id.strip()]
            local_index = int(requested_gpu.split(":", 1)[1])
            # If CUDA_VISIBLE_DEVICES is already set before Python starts, treat cuda:N
            # as a process-local device index.
            if 0 <= local_index < len(visible_gpu_ids):
                runtime_gpu = f"cuda:{local_index}"
            else:
                physical_index = str(local_index)
                if physical_index not in visible_gpu_ids:
                    raise ValueError(
                        f"Requested GPU '{requested_gpu}' is not available under "
                        f"CUDA_VISIBLE_DEVICES={visible_devices}."
                    )
                runtime_gpu = f"cuda:{visible_gpu_ids.index(physical_index)}"
        else:
            physical_index = requested_gpu.split(":", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = physical_index
            runtime_gpu = "cuda:0"

    elif requested_gpu == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        runtime_gpu = "cuda:0"

    args.requested_gpu = requested_gpu
    args.gpu = runtime_gpu
    if hasattr(args, "device"):
        args.device = runtime_gpu
    return args


def init_genesis(args, gs):
    """Initialize Genesis after device selection has been normalized."""
    configure_runtime_device(args)
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    if not args.cpu and args.gpu.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.gpu))

def get_args():
    """Parse command line arguments

    Returns:
        args: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',           type=str, default='go2', help="task name")
    parser.add_argument('--headless',       action='store_true', default=False, help="enable visualization by default")
    parser.add_argument('--cpu',            action='store_true', default=False, help="use CPU instead of CUDA")
    parser.add_argument('--gpu',            type=str, default='cuda:0', help="which GPU to use (default: cuda:0)")
    parser.add_argument('--num_envs',       type=int, default=None, help="number of parallel environments")
    parser.add_argument('--max_iterations', type=int, default=None, help="max number of training iterations")
    parser.add_argument('--resume',         action='store_true', default=False, help="resume training from specified checkpoint")
    parser.add_argument('--sync_wandb',     action='store_true', default=False, help="synchronize training log with wandb")
    parser.add_argument('--export_onnx',    action='store_true', default=False, help="export policy as onnx (besides jit)")
    parser.add_argument('--debug',          action='store_true', default=False, help="enable debug mode")
    parser.add_argument('--load_run',       type=str, default=None, help="run to load, default: last run")
    parser.add_argument('--ckpt',           type=int, default=-1, help="checkpoint to load, -1 means latest")
    parser.add_argument('--use_joystick',   action='store_true', default=False, help="use joystick to provide commands")
    parser.add_argument('--joystick_type',  type=str, default='xbox', help="type of joystick: xbox, switch")
    parser.add_argument('--follow_robot',   action='store_true', default=False, help="whether the camera follows the robot during play")
    parser.add_argument('--record_frames',   action='store_true', default=False, help="whether to record the camera")

    return configure_runtime_device(parser.parse_args())

def override_configs(env_cfg, args):
    """Override some environment configuration parameters for testing

    Args:
        env_cfg: environment configuration
        args: command line arguments
    """
    task_name = args.task
    # override some parameters for testing
    # number of environments
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    if "cts" in task_name:  # cts specific
        env_cfg.env.num_teacher = 1
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    # adjust parameters according to terrain type
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 2
        env_cfg.terrain.border_size = 5.0
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected   = True
        
        # random uniform terrain
        env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.random_uniform_terrain", 
                                          "min_height" : -0.05, "max_height": 0.05, 
                                          "step":0.005, "downsampled_scale" : 0.2}
        # # slope
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_sloped_terrain",
        #                                   "slope": -0.4, "platform_size": 3.0}
        # stairs
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_stairs_terrain",
        #                                 "step_width": 0.31, "step_height": -0.1, "platform_size": 3.0}
        # # discrete obstacles
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.discrete_obstacles_terrain",
        #                                   "max_height": 0.06,
        #                                   "min_size": 1.0,
        #                                   "max_size": 2.0,
        #                                   "num_rects": 20,
        #                                   "platform_size": 3.0}
        # wave terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.wave_terrain", 
        #                                   "amplitude": 0.2, "num_waves": 2}
        # stepping stones
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.stepping_stones_terrain",
        #                                   "stone_size": 1.0, "max_height": 0.1,
        #                                   "stone_distance": 0.3, "platform_size": 3.0}
        # gap terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.gap_terrain", 
        #                                   "gap_size": 0.2, "platform_size": 3.0}
        # pit terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pit_terrain", 
        #                                   "depth": 0.2, "platform_size": 3.0}
    else:
        for i in range(2):
            env_cfg.viewer.pos[i] = env_cfg.viewer.pos[i] - env_cfg.terrain.plane_length / 4
            env_cfg.viewer.lookat[i] = env_cfg.viewer.lookat[i] - env_cfg.terrain.plane_length / 4    
        
            
    if args.use_joystick:
        env_cfg.commands.heading_command = False
    
    # env_cfg.commands.ranges.lin_vel_x = [-1.0, 1.0]
    # env_cfg.commands.ranges.lin_vel_y = [-1.0, 1.0]
    # env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]

    env_cfg.commands.ranges.lin_vel_x   = [-1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y   = [-1.0, 1.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]

    env_cfg.commands.ranges.heading = [0.0, 0.0]

    # Turn off/on domain randomization elements
    env_cfg.noise.add_noise = True
    # Disable some of the domain randomization (our payload will handle that now)
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_pd_gain = False           # Maybe keep this on?
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = True

    env_cfg.asset.fix_base_link = False
    # env_cfg.env.debug_viz = False
    # env_cfg.env.debug = False
    # env_cfg.env.debug_draw_terrain_height_points = False

    if args.record_frames or args.follow_robot:
        print("Adding Camera!")
        env_cfg.viewer.add_camera = True  # use a extra camera for moving

    

def print_debug_info(env, robot_index):
    """Print debug information while interacting

    Args:
        env: environment object
        robot_index (int): index of the robot to print info for
    """
    # print debug info
    # print("base lin vel: ", env.simulator.base_lin_vel[robot_index, :].cpu().numpy())
    # print("base yaw angle: ", env.simulator.base_euler[robot_index, 2].item())
    # print("base height: ", env.simulator.base_pos[robot_index, 2].cpu().numpy())
    # print("foot_height: ", env.simulator.feet_pos[robot_index, :, 2].cpu().numpy())
    # print(f"ankle pitch: {env.simulator.dof_pos[robot_index, [3,7]].cpu().numpy()}")
    pass

def interaction_loop(train_cfg, env, policy, args):
    """Run interaction loop between environment and policy

    Args:
        env: environment object
        policy : a policy that takes observations and outputs actions
        args: command line arguments
    """
    
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = 300 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards

    # logger = ExpLogger(train_cfg.runner.exp_data_path)
    exp_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'exp_logs', train_cfg.runner.experiment_name, train_cfg.runner.load_run)
    logger = ExpLogger(exp_log_root, ref_key='q_actual', length_limit=100)
    
    # Get initial observations according to task type
    task_name = args.task
    if "ts" in task_name or "cat" in task_name:  # teacher-student
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    elif "ee" in task_name:  # explicit estimator
        estimator_features, _, _ = env.get_observations()
    elif "dreamwaq" in task_name:  # dreamwaq
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = env.get_observations()
    elif "pact" in task_name:
        obs_buf, obs_history, privileged_obs_buf, explicit_labels = env.get_observations()
    else: # vanilla
        obs = env.get_observations()
    
    # Setup joystick if needed
    if args.use_joystick:
        joystick = Joystick(joystick_type=args.joystick_type)
    
    # env.commands[:, 0] = 0.5
    # env.commands[:, 1] = 0
    # env.commands[:, 2] = 0
    # env.commands[:, 3] = 0
    
    if args.record_frames:
        env.simulator._floating_camera.start_recording()

    # interaction loop
    for i in range(int(10.00*env.max_episode_length)):

        # env.commands[:, 0] = 0.5
        # env.commands[:, 1] = 0
        # env.commands[:, 2] = 0
        
        # update commands from joystick
        if args.use_joystick:
            joystick.update()
            env.commands[:, 0] = -joystick.ly
            env.commands[:, 1] = -joystick.lx
            env.commands[:, 2] = -joystick.rx
        
        # set the viewer camera to follow the first environment by default
        # TODO - fix recording/general camera follow conflict
        if args.follow_robot:
            pos = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.pos, dtype=np.float32)
            lookat = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.lookat, dtype=np.float32)
            # env.set_viewer_camera(pos, lookat)
            env.set_camera(pos, lookat)
            env.simulator._floating_camera.render()
        
        # Step the environment according to task type
        if "ts" in task_name or "cat" in task_name:
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
        elif "ee" in task_name:
            actions = policy(estimator_features.detach())
            estimator_features, estimator_labels, _, rews, dones, infos = env.step(actions.detach())
        elif "waq" in task_name:
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos = env.step(actions.detach())
        elif "pact" in task_name:
            # print("obs_buf - ", obs_buf.cpu().numpy())
            # print("obs_history - ", obs_history.cpu().numpy())
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, explicit_labels, rews, dones, infos, grfs = env.step(actions.detach())
        else:
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
        
        # print debug info
        print_debug_info(env, robot_index)
        
        # # Update logger info
        # if i < stop_state_log:
        #     logger.log_states(
        #         {
        #             'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
        #             'dof_pos': env.simulator.dof_pos[robot_index, joint_index].item(),
        #             'dof_vel': env.simulator.dof_vel[robot_index, joint_index].item(),
        #             'dof_torque': env.simulator.torques[robot_index, joint_index].item(),
        #             'command_x': env.commands[robot_index, 0].item(),
        #             'command_y': env.commands[robot_index, 1].item(),
        #             'command_yaw': env.commands[robot_index, 2].item(),
        #             'base_vel_x': env.simulator.base_lin_vel[robot_index, 0].item(),
        #             'base_vel_y': env.simulator.base_lin_vel[robot_index, 1].item(),
        #             'base_vel_z': env.simulator.base_lin_vel[robot_index, 2].item(),
        #             'base_vel_yaw': env.simulator.base_ang_vel[robot_index, 2].item(),
        #             'contact_forces_z': env.simulator.link_contact_forces[robot_index, 
        #                                                                   env.simulator.feet_indices, 2].cpu().numpy()
        #         }
        #     )
        # elif i==stop_state_log:
        #     logger.plot_states()
        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i==stop_rew_log:
        #     logger.print_rewards()

        logger.log_states(
            {
                'base_cmd':env.commands.detach().cpu().numpy().tolist(),
                'base_pose':env.simulator.base_pos.detach().cpu().numpy().tolist(),
                'base_rpy':env.simulator.base_euler.detach().cpu().numpy().tolist(),
                'q_actual':env.simulator.dof_pos.detach().cpu().numpy().tolist(),
                'base_lin_vel':env.simulator.base_lin_vel.detach().cpu().numpy().tolist(),
                'base_ang_vel':env.simulator.base_ang_vel.detach().cpu().numpy().tolist(),
                'dof_vel':env.simulator.dof_vel.detach().cpu().numpy().tolist(),
                'proj_grav':env.simulator.projected_gravity.detach().cpu().numpy().tolist(),
                'feet_pos':env.simulator.feet_pos.detach().cpu().numpy().tolist(),
                'failure':list(map(int, env.get_failure_idx().detach().cpu().numpy().tolist())),
            })

def export_policy(alg_runner, path: str, args, env_cfg, train_cfg):
    """export the policy as jit script according to different task types

    Args:
        alg_runner: algorithm runner
        path (str): path to which the policy is exported
        args: command line arguments
        env_cfg: environment configuration
        train_cfg: training configuration
    """
    task_name = args.task
    if "ts" in task_name or "cat" in task_name:
        exporter = PolicyExporterTS(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "ee" in task_name:
        exporter = PolicyExporterEE(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "dreamwaq" in task_name:
        exporter = PolicyExporterWaQ(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "pact" in task_name:
        exporter = PolicyExporterPACT(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, train_cfg)
    else:
        exporter = PolicyExporter(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    
    print('Exported policy as jit script to: ', path)
    if args.export_onnx:
        print('Exported policy as onnx to: ', path)
    

def play(args):
    """Main function to run the play script

    Args:
        args (_type_): command line arguments
    """
    if "genesis" in SIMULATOR:
        init_genesis(args, gs)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    override_configs(env_cfg, args)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++ or python)
    if train_cfg.runner.load_run == -1:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        try:
            runs = os.listdir(log_root)
            #TODO sort by date to handle change of month
            runs.sort()
            if 'exported' in runs: runs.remove('exported')
            train_cfg.runner.load_run = runs[-1]
        except:
            raise ValueError("No runs in this directory: " + root)
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 
                            train_cfg.runner.load_run, 'exported')
    # export_policy(ppo_runner, path, args, env_cfg, train_cfg)

    interaction_loop(train_cfg, env, policy, args)

    if args.record_frames:
        try:
            filename_mp4 = f"{train_cfg.runner.experiment_name}_discrete_normal_viz.mp4"
        except:
            from datetime import datetime
            filename_mp4 = f"{datetime.now().timestamp()}"
        
        env.simulator._floating_camera.stop_recording(save_to_filename=filename_mp4, fps=30)
        print("Saved recording to " + filename_mp4)
    
    
if __name__ == '__main__':
    args = get_args()
    play(args)
