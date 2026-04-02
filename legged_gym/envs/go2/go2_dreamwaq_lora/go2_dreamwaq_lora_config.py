from legged_gym import *
from legged_gym.envs.base.legged_robot_dreamwaq_config import LeggedRobotDreamwaqCfg, LeggedRobotDreamwaqCfgPPO
from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq import Go2Dreamwaq
from legged_gym.envs.base.common_cfgs import Go2RoughCommonCfg

class Go2DreamwaqLoraCfg( LeggedRobotDreamwaqCfg ):
    class env( LeggedRobotDreamwaqCfg.env ):
        num_envs = 3000
        num_actions = 12
        num_observations = 45  # num_obs
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = 16
        num_explicit_dims = 24  # base linear velocity
        num_decoder_output = num_observations
        c_frame_stack = 5
        single_critic_obs_len = num_observations + 31 + 81 + 17 + 3
        num_privileged_obs = c_frame_stack * single_critic_obs_len
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
    
    class terrain( Go2RoughCommonCfg.terrain ):
        pass
    class init_state( Go2RoughCommonCfg.init_state ):
        pass
    class control( Go2RoughCommonCfg.control ):
        pass
    class asset( Go2RoughCommonCfg.asset ):
        pass
    class rewards( Go2RoughCommonCfg.rewards ):
        class scales( Go2RoughCommonCfg.rewards.scales ):
            pass

    class commands( LeggedRobotDreamwaqCfg.commands ):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotDreamwaqCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
    class domain_rand(LeggedRobotDreamwaqCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.7]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.
        randomize_com_displacement = True
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = False
        joint_armature_range = [0.015, 0.025]  # [N*m*s/rad]
        randomize_joint_friction = False
        joint_friction_range = [0.01, 0.02]
        randomize_joint_damping = False
        joint_damping_range = [0.25, 0.3]

class Go2DreamwaqLoraCfgPPO( LeggedRobotDreamwaqCfgPPO ):
    runner_class_name = "DreamWaQLoRaRunner" 
    class policy( LeggedRobotDreamwaqCfgPPO.policy ):
        base_model="/home/pablo/Documents/HCR_Genesis_LR_CL/logs/go2_flat/Apr02_01-57-54_dreamwaq_flat_genesis/model_3000.pt"
        critic_hidden_dims = [1024, 256, 128]
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [256, 128]
        actor_ranks: list[int] = [4, 4, 4, 4]
        encoder_ranks: list[int] = [4, 4, 4]
        decoder_ranks: list[int] = [4, 4, 4]
        latent_mu_rank: int = 4
        vel_mu_rank: int = 4
        latent_var_ranks: list[int] = [4] #can be in int too but list for consistency
        vel_var_ranks: list[int] = [4] #can be in int too but list for consistency
    class algorithm( LeggedRobotDreamwaqCfgPPO.algorithm ):
        encoder_lr = 2.e-4
        num_encoder_epochs = 1
        vae_kld_weight = 2.0
    class runner( LeggedRobotDreamwaqCfgPPO.runner ):
        policy_class_name = "ActorCriticDreamWaQLoRA"
        run_name = 'dreamwaq_lora'
        if SIMULATOR == "genesis":
            run_name += "_genesis"
        elif SIMULATOR == "isaacgym":
            run_name += "_isaacgym"
        elif SIMULATOR == "isaaclab":
            run_name += "_isaaclab"
        experiment_name = 'go2_lora'
        save_interval = 500
        max_iterations = 3000