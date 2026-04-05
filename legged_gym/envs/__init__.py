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

from legged_gym import *
from .base.legged_robot import LeggedRobot
# go2
from legged_gym.envs.go2.go2 import GO2
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO
# go2 walk these ways
from legged_gym.envs.go2.go2_wtw.go2_wtw import GO2WTW
from legged_gym.envs.go2.go2_wtw.go2_wtw_config import GO2WTWCfg, GO2WTWCfgPPO
# go2_ts(teacher-student)
from legged_gym.envs.go2.go2_ts.go2_ts import Go2TS
from legged_gym.envs.go2.go2_ts.go2_ts_config import Go2TSCfg, Go2TSCfgPPO
# go2_ee(explicit estimator)
from legged_gym.envs.go2.go2_ee.go2_ee import Go2EE
from legged_gym.envs.go2.go2_ee.go2_ee_config import Go2EECfg, Go2EECfgPPO
# go2_cts(concurrent teacher-student)
from legged_gym.envs.go2.go2_cts.go2_cts import Go2CTS
from legged_gym.envs.go2.go2_cts.go2_cts_config import Go2CTSCfg, Go2CTSCfgPPO
# go2_dreamwaq
from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq import Go2Dreamwaq
from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq_config import Go2DreamwaqCfg, Go2DreamwaqCfgPPO
#made a syslink so the task manager doesnt get confused
from legged_gym.envs.go2.go2_dreamwaq_flat.go2_dreamwaq_flat import Go2Dreamwaq as Go2DreamwaqFlat
from legged_gym.envs.go2.go2_dreamwaq_flat.go2_dreamwaq_flat_config import Go2DreamwaqFlatCfg, Go2DreamwaqFlatCfgPPO

from legged_gym.envs.go2.go2_dreamwaq_lora.go2_dreamwaq_lora import Go2DreamwaqLora
from legged_gym.envs.go2.go2_dreamwaq_lora.go2_dreamwaq_lora_config import Go2DreamwaqLoraCfg, Go2DreamwaqLoraCfgPPO

from legged_gym.envs.go2.go2_dreamwaq_lora_env.go2_dreamwaq_lora_env import Go2DreamwaqLora as Go2DreamwaqLoraEnv
from legged_gym.envs.go2.go2_dreamwaq_lora_env.go2_dreamwaq_lora_env_config import Go2DreamwaqLoraCfg as Go2DreamwaqLoraEnvCfg, Go2DreamwaqLoraCfgPPO as Go2DreamwaqLoraEnvCfgPPO

# go2_cat(constraint-as-termination)
from legged_gym.envs.go2.go2_cat.go2_cat import Go2CaT
from legged_gym.envs.go2.go2_cat.go2_cat_config import Go2CaTCfg, Go2CaTCfgPPO
# go2_ts_depth
from legged_gym.envs.go2.go2_ts_depth.go2_ts_depth import Go2TSDepth
from legged_gym.envs.go2.go2_ts_depth.go2_ts_depth_config import Go2TSDepthCfg, Go2TSDepthCfgPPO
# go2_nav
from legged_gym.envs.go2.go2_nav.go2_nav import GO2Nav
from legged_gym.envs.go2.go2_nav.go2_nav_config import GO2NavCfg, GO2NavCfgPPO

# tron1_pf
from legged_gym.envs.tron1_pf.tron1_pf import TRON1PF
from legged_gym.envs.tron1_pf.tron1_pf_config import TRON1PFCfg, TRON1PFCfgPPO
# tron_pf_ee
from legged_gym.envs.tron1_pf.tron1_pf_ee.tron1_pf_ee import TRON1PF_EE
from legged_gym.envs.tron1_pf.tron1_pf_ee.tron1_pf_ee_config import TRON1PF_EECfg, TRON1PF_EECfgPPO
# tron1_sf
from legged_gym.envs.tron1_sf.tron1_sf import TRON1SF
from legged_gym.envs.tron1_sf.tron1_sf_config import TRON1SFCfg, TRON1SFCfgPPO
# bipedal_walker
from legged_gym.envs.bipedal_walker.bipedal_walker_config import BipedalWalkerCfg, BipedalWalkerCfgPPO
from legged_gym.envs.bipedal_walker.bipedal_walker import BipedalWalker
# # go2_sysid
# from legged_gym.envs.go2.go2_sysid.go2_sysid import GO2SysID
# from legged_gym.envs.go2.go2_sysid.go2_sysid_config import GO2SysIDCfg


from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", GO2, GO2Cfg(), GO2CfgPPO())
task_registry.register( "go2_wtw", GO2WTW, GO2WTWCfg(), GO2WTWCfgPPO())
task_registry.register( "go2_ts", Go2TS, Go2TSCfg(), Go2TSCfgPPO())
task_registry.register( "go2_ee", Go2EE, Go2EECfg(), Go2EECfgPPO())
task_registry.register( "go2_cts", Go2CTS, Go2CTSCfg(), Go2CTSCfgPPO())
task_registry.register( "go2_dreamwaq", Go2Dreamwaq, Go2DreamwaqCfg(), Go2DreamwaqCfgPPO())
task_registry.register( "go2_dreamwaq_flat", Go2DreamwaqFlat, Go2DreamwaqFlatCfg(), Go2DreamwaqFlatCfgPPO())
task_registry.register( "go2_dreamwaq_lora", Go2DreamwaqLora, Go2DreamwaqLoraCfg(), Go2DreamwaqLoraCfgPPO())
task_registry.register( "go2_dreamwaq_lora_env", Go2DreamwaqLoraEnv, Go2DreamwaqLoraEnvCfg(), Go2DreamwaqLoraEnvCfgPPO())
task_registry.register( "go2_cat", Go2CaT, Go2CaTCfg(), Go2CaTCfgPPO())
# task_registry.register( "go2_ts_depth", Go2TSDepth, Go2TSDepthCfg(), Go2TSDepthCfgPPO())
task_registry.register( "go2_nav", GO2Nav, GO2NavCfg(), GO2NavCfgPPO())
task_registry.register( "tron1_pf", TRON1PF, TRON1PFCfg(), TRON1PFCfgPPO())
task_registry.register( "tron1_pf_ee", TRON1PF_EE, TRON1PF_EECfg(), TRON1PF_EECfgPPO())
task_registry.register( "tron1_sf", TRON1SF, TRON1SFCfg(), TRON1SFCfgPPO())
# task_registry.register( "go2_sysid", GO2SysID, GO2SysIDCfg(), GO2CfgPPO())
# task_registry.register( "bipedal_walker", BipedalWalker, BipedalWalkerCfg(), BipedalWalkerCfgPPO())