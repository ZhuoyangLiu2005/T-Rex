# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to move the end-effector by a relative pose.

This script demonstrates how to move both robot arms to a position and orientation
relative to their current pose.
"""

import numpy as np
import tyro
from dexcomm import RateLimiter
from dexmotion.motion_manager import MotionManager

from dexcontrol.robot import Robot


def move_real_robot(
    bot: Robot, qs_sample: list[list[float]], control_hz: float = 250
) -> None:
    """Move the real robot to a specific pose.

    Args:
        qs_sample: List of joint arrays to move
    """
    rate_limiter = RateLimiter(control_hz)
    for q in qs_sample:
        # Extract left and right arm joint arrays
        left_arm_joints = q[:7]
        right_arm_joints = q[7:14]

        # Set joint positions for both arms
        bot.left_arm.set_joint_pos(left_arm_joints)
        bot.right_arm.set_joint_pos(right_arm_joints)
        rate_limiter.sleep()


def main() -> None:
    """Run the move to relative pose example.

    Creates an instance of MoveToRelativePoseTask and executes the motion planning
    and control sequence to move both end-effectors to a relative pose.
    """
    # Initialize robot and get current joint positions
    bot = Robot()
    control_hz = 250
    if bot.has_component("torso"):
        initial_joint_pos = bot.get_joint_pos_dict(
            component=["left_arm", "right_arm", "torso", "head"]
        )
    else:
        initial_joint_pos = bot.get_joint_pos_dict(
            component=["left_arm", "right_arm", "head"]
        )

    # Create task instance with initial joint configuration
    mm = MotionManager(initial_joint_configuration_dict=initial_joint_pos)

    qs_sample = mm.right_arm.set_ee_pose(
        pos=np.array([0.0, 0.0, 0.1]),
        relative=True,
        target_frame="R_arm_j5",
    )
    input("Press Enter to move right end-effector by 0.1 meters in z direction")
    move_real_robot(bot, qs_sample, control_hz)

    qs_sample = mm.left_arm.set_ee_pose(pos=np.array([0.0, 0.1, 0.0]), relative=True)
    input("Press Enter to move left end-effector by 0.1 meters in y direction")
    move_real_robot(bot, qs_sample, control_hz)

    qs_sample = mm.left_arm.move_ee_xyz(
        np.array([0.1, 0.0, 0.0]), target_frame="L_arm_j3"
    )
    input("Press Enter to move left end-effector by 0.1 meters in x direction")
    move_real_robot(bot, qs_sample, control_hz)

    qs_sample = mm.right_arm.move_ee_rpy(np.array([np.pi / 4, 0.0, 0.0]))
    input("Press Enter to rotate right end-effector by 45 degrees around x-axis")
    move_real_robot(bot, qs_sample, control_hz)


if __name__ == "__main__":
    tyro.cli(main)
