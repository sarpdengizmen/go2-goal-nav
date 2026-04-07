# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-navigation reward functions for the Go2 point-goal task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def distance_to_goal(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return the straight-line distance from the robot to the goal point (positive scalar).

    Use with a **negative** reward weight so the agent is penalised for being far away.
    Reads ``dist_to_goal`` from the command term (true distance, unaffected by command zeroing).
    """
    term = env.command_manager.get_term(command_name)
    return term.dist_to_goal



def goal_proximity_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward proximity to the goal using a Gaussian exponential kernel.

    Returns ``exp(-dist² / std²)``: 1.0 at the goal center, decaying smoothly to 0 far away.
    Use with a **positive** weight. Recommended ``std`` ≈ 1.0–2.0 m for typical navigation distances.
    """
    term = env.command_manager.get_term(command_name)
    return torch.exp(-(term.dist_to_goal**2) / std**2)


def heading_penalty_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the robot for not facing the goal (linear in heading error).

    Heading error = ``|atan2(dy_b, dx_b)|`` in radians, range ``[0, π]``.
    Returns zero when within *distance_threshold* metres of the goal.
    Use with a **negative** weight.
    """
    cmd = env.command_manager.get_command(command_name)[:, :2]
    term = env.command_manager.get_term(command_name)
    heading_error = torch.abs(torch.atan2(cmd[:, 1], cmd[:, 0]))
    return heading_error * (term.dist_to_goal > distance_threshold)


def heading_reward_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for facing the goal proportional to cosine of the heading error.

    Returns ``cos(heading_error) = dx_b / dist``, a continuous value in ``[-1.0, 1.0]``:
    +1.0 when perfectly facing the goal, 0.0 when perpendicular, -1.0 when facing directly away.
    Suppressed within *distance_threshold* metres of the goal. Use with a **positive** weight.
    """
    cmd = env.command_manager.get_command(command_name)[:, :2]
    term = env.command_manager.get_term(command_name)
    dist = torch.clamp(term.dist_to_goal, min=1e-6)
    reward = cmd[:, 0] / dist
    return reward * (term.dist_to_goal > distance_threshold)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation from default pose when within *command_threshold* metres of the goal."""
    term = env.command_manager.get_term(command_name)
    return mdp.joint_deviation_l1(env, asset_cfg) * (term.dist_to_goal < command_threshold)
