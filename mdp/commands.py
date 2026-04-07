# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Point-goal command generator for the Go2 goal-navigation task."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PointGoalCommand(CommandTerm):
    """Commands a fixed XY target point in world frame, expressed as body-frame displacement.

    On each episode reset a goal is sampled at a random distance and angle from the robot.
    The command output is the 2-D vector from the robot to the goal expressed in the robot
    body frame: ``[dx_b, dy_b]``, updated every step as the robot moves and rotates.
    The norm of this vector equals the straight-line distance to the goal.

    If ``cfg.resample_on_reach`` is True, a new goal is sampled after the robot has dwelled
    within ``cfg.goal_reach_threshold`` for ``cfg.resample_dwell_s`` seconds. Intended for
    play / evaluation demos — leave False during training.
    """

    def __init__(self, cfg: PointGoalCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        # world-frame XY goal positions, one per env
        self.goal_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
        # body-frame displacement to goal (the actual command tensor)
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        # true straight-line distance to goal, used by reward functions
        self.dist_to_goal = torch.zeros(self.num_envs, device=self.device)
        # distance from the previous timestep
        self.prev_dist_to_goal = torch.zeros(self.num_envs, device=self.device)
        self._time_at_goal = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Body-frame displacement to the goal. Shape: (num_envs, 2)."""
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        angles = torch.empty(n, device=self.device).uniform_(0.0, 2.0 * math.pi)
        dists = torch.empty(n, device=self.device).uniform_(*self.cfg.goal_distance_range)
        robot_xy = self.robot.data.root_pos_w[env_ids, :2]
        self.goal_pos_w[env_ids, 0] = robot_xy[:, 0] + dists * torch.cos(angles)
        self.goal_pos_w[env_ids, 1] = robot_xy[:, 1] + dists * torch.sin(angles)
        # initialise both distance buffers to the sampled distance to avoid a spurious
        # progress spike on the first step after reset
        self.dist_to_goal[env_ids] = dists
        self.prev_dist_to_goal[env_ids] = dists
        self._time_at_goal[env_ids] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        positions = torch.cat(
            [self.goal_pos_w, torch.full((self.num_envs, 1), 0.5, device=self.device)], dim=-1
        )
        self.goal_visualizer.visualize(translations=positions)

    def _update_metrics(self):
        pass

    def _update_command(self):
        # snapshot distance before updating
        self.prev_dist_to_goal[:] = self.dist_to_goal

        # world-frame displacement to goal
        delta_w = self.goal_pos_w - self.robot.data.root_pos_w[:, :2]
        delta_w_3d = torch.cat([delta_w, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        # rotate into body frame
        delta_b_3d = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w), delta_w_3d
        )
        self.dist_to_goal[:] = delta_w.norm(dim=1)

        self._command[:] = delta_b_3d[:, :2]
        if self.cfg.goal_reach_threshold > 0.0:
            at_goal = self.dist_to_goal < self.cfg.goal_reach_threshold
            if self.cfg.resample_on_reach:
                # accumulate dwell time; reset timer for envs that left the goal radius
                self._time_at_goal[at_goal] += self._env.step_dt
                self._time_at_goal[~at_goal] = 0.0
                self._command[at_goal] = 0.0
                # resample once dwell time exceeds threshold
                ready_ids = (self._time_at_goal >= self.cfg.resample_dwell_s).nonzero(as_tuple=False).flatten()
                if len(ready_ids) > 0:
                    self._resample_command(ready_ids)
                    delta_w = self.goal_pos_w - self.robot.data.root_pos_w[:, :2]
                    delta_w_3d = torch.cat([delta_w, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
                    delta_b_3d = math_utils.quat_apply_inverse(
                        math_utils.yaw_quat(self.robot.data.root_quat_w), delta_w_3d
                    )
                    self.dist_to_goal[:] = delta_w.norm(dim=1)
                    self._command[:] = delta_b_3d[:, :2]
            else:
                self._command[at_goal] = 0.0


@configclass
class PointGoalCommandCfg(CommandTermCfg):
    """Configuration for :class:`PointGoalCommand`."""

    class_type: type = PointGoalCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene (e.g. ``"robot"``)."""

    goal_distance_range: tuple[float, float] = (2.0, 8.0)
    """Min/max straight-line distance from the robot to the sampled goal [m]."""

    goal_reach_threshold: float = 0.5
    """Distance below which the command is zeroed (or a new goal is sampled if resample_on_reach=True) [m].
    Set to 0.0 to disable."""

    resample_on_reach: bool = False
    """If True, sample a new goal after the robot dwells within goal_reach_threshold for
    resample_dwell_s seconds. Intended for play/eval mode to create a continuous multi-goal
    navigation demo. Leave False during training so goal zeroing behaviour is unchanged."""

    resample_dwell_s: float = 1.0
    """Seconds the robot must remain within goal_reach_threshold before a new goal is sampled.
    Only used when resample_on_reach=True."""

    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    """Set very large so the goal never resamples mid-episode."""

    goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_position",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )
    """Green sphere marker at the goal position."""
