# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-contained flat terrain + point-goal configuration for Unitree Go2.

Every component of the MDP is defined explicitly in this single file.
Nothing is inherited from intermediate config layers — only ManagerBasedRLEnvCfg.

External dependencies:
  - isaaclab          (core framework — standard installation)
  - isaaclab_assets   (robot USD assets — standard installation)
  The custom MDP components (PointGoalCommand, goal-nav rewards) are bundled
  in the sibling ``mdp/`` package and require no additional installation.

Task: navigate to a randomly placed goal point 2-8 m away.
Policy inputs: angular velocity, body-frame goal displacement, joint state, last action.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# All MDP functions — stock isaaclab.envs.mdp plus goal-nav custom functions.
# This is the only non-standard import; the mdp/ package lives next to this file.
from . import mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


_GOAL_PROXIMITY_THRESHOLD = 0.2  # metres — heading reward suppressed inside this radius


##
# Scene
##


@configclass
class _SceneCfg(InteractiveSceneCfg):
    """Flat plane scene with Go2 robot and contact sensing."""

    # Flat ground — no height-field generator, no terrain curriculum
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Unitree Go2 quadruped
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Contact sensor on all robot bodies (used for termination check and feet air-time)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Dome light
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# Commands
##


@configclass
class _CommandsCfg:
    """Point-goal command: body-frame (dx, dy) displacement to the target."""

    point_goal = mdp.PointGoalCommandCfg(
        asset_name="robot",
        goal_distance_range=(2.0, 8.0),
        resampling_time_range=(1.0e6, 1.0e6),   # one fixed goal per episode
        debug_vis=True,
    )


##
# Actions
##


@configclass
class _ActionsCfg:
    """Joint position targets with a conservative scale for the Go2."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


##
# Observations
##


@configclass
class _ObservationsCfg:
    """Observations: angular velocity, goal direction, joint state, last action.

    Excluded intentionally:
    - base_lin_vel: not directly observable from onboard IMU in real deployment
    - projected_gravity: redundant on flat terrain and not directly measured
    - height_scan: not needed on flat terrain
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # 3-D angular velocity of the base in the body frame
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # 2-D body-frame displacement to goal (output of PointGoalCommand: [dx_b, dy_b])
        goal_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "point_goal"})
        # Joint positions relative to default pose
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # Joint velocities relative to default
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # Previous action sent to the robot
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Rewards
##


@configclass
class _RewardsCfg:
    """Reward terms: goal-navigation rewards + locomotion quality penalties."""

    # ── Navigation ─────────────────────────────────────────────────────────

    # Dense linear distance penalty — constant gradient all the way to center
    distance_to_goal = RewTerm(
        func=mdp.distance_to_goal,
        weight=-0.7,
        params={"command_name": "point_goal"},
    )
    # Coarse proximity bonus — gradient peaks at ~0.7m, guides long-range approach
    goal_proximity = RewTerm(
        func=mdp.goal_proximity_exp,
        weight=5.0,
        params={"command_name": "point_goal", "std": 1.0},
    )
    # Fine proximity bonus — gradient peaks at ~0.21m, braking funnel for precise stop
    goal_proximity_fine = RewTerm(
        func=mdp.goal_proximity_exp,
        weight=8.0,
        params={"command_name": "point_goal", "std": 0.3},
    )
    # Heading alignment reward — cosine similarity; +1 facing goal, -1 facing away.
    # Prevents backwards locomotion. Suppressed inside goal radius.
    heading_reward_to_goal = RewTerm(
        func=mdp.heading_reward_to_goal,
        weight=1.0,
        params={"command_name": "point_goal", "distance_threshold": _GOAL_PROXIMITY_THRESHOLD},
    )
    # Absolute heading error penalty — constant gradient regardless of error magnitude.
    # Forces active correction even for small (5-10°) drifts that cosine ignores.
    heading_penalty_to_goal = RewTerm(
        func=mdp.heading_penalty_to_goal,
        weight=-0.2,
        params={"command_name": "point_goal", "distance_threshold": _GOAL_PROXIMITY_THRESHOLD},
    )
    # Falling penalty — must exceed discounted future distance penalty to prevent suicide.
    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-1000.0,
        params={"term_keys": "base_contact"},
    )

    # ── Locomotion quality ─────────────────────────────────────────────────

    # Prevent base from bouncing vertically
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # Penalise roll / pitch angular rates
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    # Encourage flat body orientation
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    # Joint effort and jerk penalties
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # Penalise sudden action changes — primary lever for gait smoothness
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    # Reward periodic foot lift (encourages a proper gait)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "point_goal",
            "threshold": 0.25,
        },
    )
    # Penalise large deviations from the default joint configuration (always active)
    joint_deviation_l1 = RewTerm(func=mdp.joint_deviation_l1, weight=-0.75)
    # Strongly penalise joint deviation when near the goal — holds a structured pose at arrival
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-1.0,
        params={"command_name": "point_goal", "command_threshold": 0.2},
    )


##
# Terminations
##


@configclass
class _TerminationsCfg:
    """Episode end conditions."""

    # Natural episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Penalised termination: base link contacts the ground
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )


##
# Events (domain randomisation + resets)
##


@configclass
class _EventsCfg:
    """Domain randomisation events applied at startup, reset, and during the episode."""

    # ── Startup ────────────────────────────────────────────────────────────

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # ── Reset ──────────────────────────────────────────────────────────────

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # ── Interval ───────────────────────────────────────────────────────────

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


##
# Environment configurations
##


@configclass
class UnitreeGo2FlatGoalStandaloneEnvCfg(ManagerBasedRLEnvCfg):
    """Go2 flat-terrain point-goal training configuration.

    All MDP components are defined explicitly in this single file.
    Inherits only from ManagerBasedRLEnvCfg — no intermediate config layers.
    """

    scene: _SceneCfg = _SceneCfg(num_envs=4096, env_spacing=2.5)
    commands: _CommandsCfg = _CommandsCfg()
    actions: _ActionsCfg = _ActionsCfg()
    observations: _ObservationsCfg = _ObservationsCfg()
    rewards: _RewardsCfg = _RewardsCfg()
    terminations: _TerminationsCfg = _TerminationsCfg()
    events: _EventsCfg = _EventsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class UnitreeGo2FlatGoalStandaloneEnvCfg_PLAY(UnitreeGo2FlatGoalStandaloneEnvCfg):
    """Play / inference variant: small scene, clean observations, no domain randomisation."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.sim.physx.gpu_max_rigid_contact_count = 2**20
        self.sim.physx.gpu_max_rigid_patch_count = 2**15

        self.observations.policy.enable_corruption = False

        self.events.physics_material = None
        self.events.actuator_gains = None
        self.events.push_robot = None

        self.events.reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "yaw": (0.0, 0.0),
        }

        # Resample a new goal after 1 second at the current one —
        # creates a continuous multi-goal navigation demo without episode resets.
        self.commands.point_goal.resample_on_reach = True
        self.commands.point_goal.goal_reach_threshold = 0.5
        self.commands.point_goal.resample_dwell_s = 1.0
