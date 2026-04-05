from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *

import numpy as np
import torch
from legged_gym.scripts.joystick import Joystick
    
def override_configs(env_cfg, args):
    """Override some environment configuration parameters for testing

    Args:
        env_cfg: environment configuration
        args: command line arguments
    """
    task_name = args.task
    # override some parameters for testing
    # number of environments
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 32)
    if "cts" in task_name:  # cts specific
        env_cfg.env.num_teacher = 1
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    # adjust parameters according to terrain type
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 2
        env_cfg.terrain.border_size = 5.0
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = True
        env_cfg.env.debug_draw_terrain_height_points = False
        
        
        # random uniform terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.random_uniform_terrain", 
        #                                   "min_height" : -0.05, "max_height": 0.05, 
        #                                   "step":0.005, "downsampled_scale" : 0.2}
        # slope
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_sloped_terrain",
        #                                   "slope": -0.4, "platform_size": 3.0}
        # stairs
        env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_stairs_terrain",
                                        "step_width": 0.31, "step_height": -0.1, "platform_size": 3.0}
        # discrete obstacles
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.discrete_obstacles_terrain",
        #                                   "max_height": 0.1,
        #                                   "min_size": 1.0,
        #                                   "max_size": 2.0,
        #                                   "num_rects": 20,
        #                                   "platform_size": 3.0}
        # wave terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.wave_terrain", 
        #                                   "amplitude": 0.1, "num_waves": 2}
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
        
        
    env_cfg.env.debug = True
    
    if args.use_joystick:
        env_cfg.commands.heading_command = False
    
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

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
    
    
    print('Exported policy as jit script to: ', path)
    if args.export_onnx:
        print('Exported policy as onnx to: ', path)
    
def get_args():
    """Parse command line arguments

    Returns:
        args: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, nargs='+', default=['go2'], help="task name(s)")
    parser.add_argument('--headless',       action='store_true', default=False, help="enable visualization by default")
    parser.add_argument('--cpu',            action='store_true', default=False, help="use CPU instead of CUDA")
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

    return parser.parse_args()

import copy
from rsl_rl.modules.actor_critic_dreamwaq_lora import ActorCriticDreamWaQLoRA
from rsl_rl.utils.LoRA import LoRALinear, _merge_seq

class ModulePair(nn.Module):
    def __init__(self, actor: nn.Module, vae: nn.Module):
        super().__init__()
        self.actor = actor
        self.vae = vae

    def merge(self, merge: bool = True):
        _merge_seq(self.actor, merge)
        _merge_seq(self.vae.encoder, merge)
        _merge_seq(self.vae.latent_mu, merge)
        _merge_seq(self.vae.vel_mu, merge)
        _merge_seq(self.vae.latent_var, merge)
        _merge_seq(self.vae.vel_var, merge)
        _merge_seq(self.vae.decoder, merge)

    def forward(self, obs, obs_history):
        vae_out = self.vae.inference(obs_history)
        x = torch.cat([obs, vae_out], dim=-1)
        return self.actor(x)
    
    def share_mem(self, others: list[type(self)]):
        t = list(others)
        if len(t) == 0:
            return
        models = [self] + t

        def should_share(name: str):
            return "lora" not in name.lower()

        def tie_module_params(modules):
            ref_module = modules[0]

            for name, param in ref_module.named_parameters(recurse=True):
                if not should_share(name):
                    continue

                # Navigate to the parent module + attribute name
                parts = name.split(".")
                attr_name = parts[-1]

                def get_parent(module, parts):
                    for p in parts[:-1]:
                        module = getattr(module, p)
                    return module

                ref_parent = get_parent(ref_module, parts)
                ref_param = getattr(ref_parent, attr_name)

                # Share with all other models
                for m in modules[1:]:
                    target_parent = get_parent(m, parts)

                    # Replace parameter with shared reference
                    setattr(target_parent, attr_name, ref_param)

        tie_module_params([m.actor for m in models])
        tie_module_params([m.vae for m in models])

class PolicyExporterWaQLora(torch.nn.Module):
    def __init__(self, actor_critics: ActorCriticDreamWaQLoRA):
        self.cop = []
        self.loras = torch.nn.ModuleList()
        for actor_critic in actor_critics:
            self.cop.append(
                ModulePair(
                    copy.deepcopy(actor_critic.actor), 
                    copy.deepcopy(actor_critic.vae)
                )
            )
        self.cop[0].share_mem(self.cop[1:])
        self.index: int = -1
        self.current: ModulePair = None
        self.merged: bool = False
    
    @torch.jit.export
    def swap(self, new_index: int = -1):
        if new_index == -1:
            if not self.merged:
                return
            if self.merged:
                self.current.merge(False)
                self.merged = False
                return
        self.current.merge(False)
        self.index = inp
        for i, model in enumerate(self.loras):
            if i == self.index:
                self.current = model
        self.current.merge(True)
        self.merged = True

    def forward(self, obs, obs_history):
        return self.current(obs, obs_history)

    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


def printer(results):
    for item in results:
        print(item.ppo_runner.__dict__)

def export(results):
    exporter = PolicyExporterWaQLora([r.ppo_runner.alg.actor_critic for r in results])
    exporter.export(results[0]["export_path"], results[0]["env_cfg"], False, results[0]["train_cfg"])


def play(args):
    """Main function to run the play script

    Args:
        args: command line arguments
    """
    results = []  # <-- list of dictionaries

    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning',
        )

    for task_name in args.task:
        print(f"\n=== Running task: {task_name} ===")

        env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
        override_configs(env_cfg, args)

        env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env,
            name=task_name,
            args=args,
            train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=env.device)

        # resolve latest run if needed
        if train_cfg.runner.load_run == -1:
            log_root = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                'logs',
                train_cfg.runner.experiment_name
            )
            try:
                runs = os.listdir(log_root)
                runs.sort()
                if 'exported' in runs:
                    runs.remove('exported')
                train_cfg.runner.load_run = runs[-1]
            except:
                raise ValueError("No runs in this directory: " + log_root)

        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            'logs',
            train_cfg.runner.experiment_name,
            train_cfg.runner.load_run,
            'exported'
        )

        results.append({
            "task": task_name,
            "env": env,
            "env_cfg": env_cfg,
            "train_cfg": train_cfg,
            "ppo_runner": ppo_runner,
            "policy": policy,
            "export_path": path,
        })
    export(results)
    return results
    
if __name__ == '__main__':
    args = get_args()
    play(args)
