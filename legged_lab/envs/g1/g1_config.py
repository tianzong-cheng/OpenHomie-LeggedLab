# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import G1_CFG
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)


@configclass
class G1RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    base_height_exp = RewTerm(func=mdp.track_height_exp, weight=2.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.025)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*", ".*_shoulder_pitch.*", ".*_elbow.*"]
            )
        },
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle.*"])},
    )
    # squat_knee
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
    # feet_clearance
    # feet_lateral_distance
    # knee_lateral_distance
    # feet_ground_parallel
    # feet_parallel
    # smoothness
    # joint_power
    # feet_stumble
    joint_torques = RewTerm(mdp.joint_torques_l2, weight=-2.5e-6)
    joint_vel = RewTerm(mdp.joint_vel_l2, weight=-1.0e-4)
    # joint_vel_limit
    # torque_limit
    # no_fly
    # joint_tracking_error
    # feet_slip
    # feet_contact_force
    # contact_momentum
    # action_vanish
    # stand_still
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class G1HomieEnvCfg(BaseEnvCfg):

    reward = G1RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.robot.terminate_contacts_body_names = [".*torso.*"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]

        self.lower_dof = [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]
        self.upper_dof = [2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        self.num_lower_dof = len(self.lower_dof)
        self.num_upper_dof = len(self.upper_dof)
        self.num_dof = self.num_lower_dof + self.num_upper_dof
        # Upper body action resample interval
        self.upper_resample_interval_s = 1.0

        self.robot.actor_obs_history_length = 6
        self.robot.critic_obs_history_length = 6


@configclass
class G1HomieAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_homie"
    wandb_project: str = "g1_homie"
