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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal


from rsl_rl.modules.vae import VAE
from rsl_rl.modules.actor_critic_dreamwaq import ActorCriticDreamWaQ
from rsl_rl.utils import LoRA

'''
Actor-Critic for Hybrid Implicit-Explicit architecture using VAE
'''

class ActorCriticDreamWaQLoRA(ActorCriticDreamWaQ):
    is_recurrent = False

    def __init__(
        self,
        *args,
        base_model: str = None,
        actor_ranks: list[int] = [4, 4, 4, 4],
        encoder_ranks: list[int] = [4, 4, 4],
        decoder_ranks: list[int] = [4, 4, 4],
        latent_mu_rank: int = 4,
        vel_mu_rank: int = 4,
        latent_var_ranks: list[int] = [4],
        vel_var_ranks: list[int] = [4],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        print(kwargs)

        # Apply LoRA
        self.actor = LoRA._from_sequential(self.actor, actor_ranks)
        self.vae.encoder = LoRA._from_sequential(self.vae.encoder, encoder_ranks)

        self.vae.latent_mu = LoRA.LoRALinear._from_linear(
            self.vae.latent_mu, latent_mu_rank
        )
        self.vae.vel_mu = LoRA.LoRALinear._from_linear(
            self.vae.vel_mu, vel_mu_rank
        )

        self.vae.latent_var = LoRA._from_sequential(
            self.vae.latent_var, latent_var_ranks
        )
        self.vae.vel_var = LoRA._from_sequential(
            self.vae.vel_var, vel_var_ranks
        )

        self.vae.decoder = LoRA._from_sequential(
            self.vae.decoder, decoder_ranks
        )

        # Optional base model load
        #if base_model is not None:
        assert base_model is not None
        print(f"Loading baseline: {base_model}")
        loaded_dict = torch.load(base_model)
        self.load_state_dict(loaded_dict["model_state_dict"])

        # Debug prints
        print(f"Encoder MLP (LORA): {self.vae.encoder}")
        print(f"Decoder MLP (LORA): {self.vae.decoder}")
        print(f"Actor MLP (LORA): {self.actor}")

    def load_state_dict(self, *args, **kwargs):
        super().load_state_dict(*args, **kwargs, strict=False)


if __name__ == "__main__":
    from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq_config import Go2DreamwaqCfg
    from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq_config import Go2DreamwaqCfgPPO

    cfg = Go2DreamwaqCfg()
    cfgppo = Go2DreamwaqCfgPPO()
    ActorCriticDreamWaQLoRA(
                num_actor_obs = cfg.env.num_observations,
                num_actions=cfg.env.num_actions,
                num_privileged_obs=cfg.env.num_privileged_obs, 
                num_history_input=cfg.env.num_history_obs,
                num_latent_dims=cfg.env.num_latent_dims,
                num_explicit_dims=cfg.env.num_explicit_dims,
                num_decoder_output=cfg.env.num_decoder_output,
                #actor_hidden_dims=cfgppo.policy.,
                critic_hidden_dims=cfgppo.policy.critic_hidden_dims,
                encoder_hidden_dims=cfgppo.policy.encoder_hidden_dims,
                decoder_hidden_dims=cfgppo.policy.decoder_hidden_dims,
                activation='elu',
                init_noise_std=1.0,
    )