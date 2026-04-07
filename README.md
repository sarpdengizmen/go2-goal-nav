# Go2 Flat Point-Goal Navigation Task

Self-contained RL task for the Unitree Go2 quadruped: navigate to a randomly placed goal point on flat terrain.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Isaac Sim | 4.5.0 | GPU required (PhysX) |
| Python | 3.10 | Bundled with Isaac Sim 4.5 |
| IsaacLab | 2.x | Core framework |
| `isaaclab_assets` | bundled with IsaacLab | Provides `UNITREE_GO2_CFG` |
| `isaaclab_rl` | bundled with IsaacLab | RSL-RL wrapper |
| RSL-RL | bundled with IsaacLab | PPO implementation |

No additional pip installs required beyond a standard IsaacLab setup.

---

## Installation

1. Drop this `goal_nav/` folder into:
   ```
   <IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/
   ```

2. Add one line to `go2/__init__.py` (at the top, with the other imports):
   ```python
   from . import goal_nav  # noqa: F401
   ```

3. Verify the installation by running one training iteration:
   ```bash
   python scripts/reinforcement_learning/rsl_rl/train.py \
       --task Isaac-Velocity-Flat-Goal-Standalone-Unitree-Go2-v0 \
       --headless --num_envs 16 --max_iterations 1
   ```
   If Isaac Sim initialises and the first iteration completes without error, the installation is correct.

---

## Task Description

| Property | Value |
|---|---|
| Robot | Unitree Go2 (12 DoF quadruped) |
| Terrain | Flat plane, no curriculum |
| Goal | Random XY point, 2–8 m from start, fixed for the episode |
| Episode length | 8 seconds (400 steps at 50 Hz) |
| Environments (training) | 4096 parallel |
| Environments (play) | 1 |

**Policy inputs (26-D observation vector):**
| Term | Dim | Description |
|---|---|---|
| `base_ang_vel` | 3 | Angular velocity in body frame |
| `goal_command` | 2 | Body-frame displacement `[dx_b, dy_b]` to goal |
| `joint_pos` | 12 | Joint positions relative to default pose |
| `joint_vel` | 12 | Joint velocities relative to default |
| `actions` | 12 | Last joint position targets sent |

Intentionally excluded: `base_lin_vel` (not IMU-observable on hardware), `projected_gravity` (redundant on flat terrain), `height_scan` (not needed on flat terrain).

**Actions:** 12-D joint position targets, scale 0.25 rad around default pose, 50 Hz.

---

## Reward Stack

| Term | Weight | Role |
|---|---|---|
| `distance_to_goal` | −0.7 | Dense distance penalty — constant gradient to goal |
| `goal_proximity` | +5.0 | Coarse Gaussian (std=1 m) — long-range approach |
| `goal_proximity_fine` | +8.0 | Fine Gaussian (std=0.3 m) — precision braking funnel |
| `heading_reward_to_goal` | +1.0 | Cosine similarity — faces goal, suppressed within 0.2 m |
| `heading_penalty_to_goal` | −0.2 | Absolute heading error — corrects small drifts |
| `termination_penalty` | −1000.0 | Fall penalty (intentionally dominant) |
| `lin_vel_z_l2` | −2.0 | Vertical bounce suppression |
| `ang_vel_xy_l2` | −0.5 | Roll/pitch rate penalty |
| `flat_orientation_l2` | −2.5 | Body tilt penalty |
| `dof_torques_l2` | −2e-4 | Joint effort penalty |
| `dof_acc_l2` | −2.5e-7 | Joint acceleration penalty |
| `action_rate_l2` | −0.15 | Action smoothness |
| `feet_air_time` | +0.25 | Foot-lift reward (threshold 0.25 s) |
| `joint_deviation_l1` | −0.75 | Default-pose deviation (always active) |
| `stand_still` | −1.0 | Posture penalty when within 0.2 m of goal |


---

## Domain Randomisation

Applied at startup (each environment gets a fixed random value for the episode):

| Parameter | Range | Purpose |
|---|---|---|
| Ground static friction | 0.6 – 1.2 | Surface variability |
| Ground dynamic friction | 0.4 – 0.9 | Surface variability |
| Base mass offset | −1.0 – +3.0 kg | Payload / battery variance |
| Actuator stiffness | ×0.8 – ×1.2 | Motor manufacturing tolerance |
| Actuator damping | ×0.8 – ×1.2 | Motor manufacturing tolerance |

Random push applied during episode: ±0.5 m/s in X/Y, interval 6–10 s.

Domain randomisation is disabled in the play variant (`_PLAY` config).

---

## Training

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Goal-Standalone-Unitree-Go2-v0 \
    --headless \
    --num_envs 4096
```

Logs are written to `logs/rsl_rl/unitree_go2_flat_goal_standalone/<timestamp>/`.

Default training config: 1500 iterations, network `[128, 128, 128]`, PPO with adaptive LR.

## Play / Visualisation

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Goal-Standalone-Unitree-Go2-Play-v0 \
    --num_envs 1
```

The play variant uses the same trained checkpoint as the training task (checkpoint lookup strips `-Play` from the task name automatically). Goals are resampled 1 second after arrival, creating a continuous navigation demo.

---

## Folder Structure

```
goal_nav/
├── README.md               ← this file
├── __init__.py             ← registers both task IDs with gymnasium
├── env_cfg.py              ← full MDP configuration (scene, rewards, obs, events)
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py   ← PPO hyperparameters + experiment name
└── mdp/
    ├── __init__.py         ← re-exports isaaclab.envs.mdp + custom additions
    ├── commands.py         ← PointGoalCommand: fixed world-frame goal, body-frame output
    └── rewards.py          ← goal-nav reward functions (distance, proximity, heading)
```