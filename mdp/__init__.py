# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP components for the Go2 goal-navigation task.

Re-exports everything from ``isaaclab.envs.mdp`` (standard actions, observations,
events, terminations, rewards) and adds the goal-nav-specific commands and rewards.
This module is the only MDP dependency of ``goal_nav/env_cfg.py``.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .commands import PointGoalCommand, PointGoalCommandCfg  # noqa: F401
from .rewards import (  # noqa: F401
    distance_to_goal,
    goal_proximity_exp,
    heading_penalty_to_goal,
    heading_reward_to_goal,
    stand_still_joint_deviation_l1,
)
