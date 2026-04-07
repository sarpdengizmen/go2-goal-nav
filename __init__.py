# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 flat point-goal navigation task — self-contained package.

To share this task, copy this entire ``goal_nav/`` folder into:
    <isaaclab_tasks>/manager_based/locomotion/velocity/config/go2/

Then add ``from . import goal_nav`` to the parent ``go2/__init__.py``.

Dependencies: stock isaaclab + isaaclab_assets only.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Goal-Standalone-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UnitreeGo2FlatGoalStandaloneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatGoalStandalonePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Goal-Standalone-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UnitreeGo2FlatGoalStandaloneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatGoalStandalonePPORunnerCfg",
    },
)
