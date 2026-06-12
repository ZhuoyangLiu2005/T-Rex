# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to control robot head movements.

This script demonstrates basic head control by moving to zero position and then
moving each joint individually.
"""

import numpy as np
import tyro
from loguru import logger

from dexcontrol.core.head import Head
from dexcontrol.robot import Robot


def move_joint_sequence(
    head: Head,
    joint_idx: int,
    step_size: float,
    wait_time: float = 1.0,
) -> None:
    """Executes movement sequence for a single joint.

    Moves joint positive, negative, then back to zero.

    Args:
        head: Robot head instance to control
        joint_idx: Index of joint to move
        step_size: Size of joint movement in radians
        wait_time: Time to wait between movements in seconds
    """
    joint_names = ["yaw", "pitch", "roll"]
    # Create the three positions (all initialized to zeros)
    positive_pos = np.zeros(3)
    negative_pos = np.zeros(3)
    zero_pos = np.zeros(3)

    # Set the target joint values
    positive_pos[joint_idx] = step_size
    negative_pos[joint_idx] = -step_size

    # Move positive
    logger.info(f"Moving head {joint_names[joint_idx]} positive ({step_size} rad)")
    head.set_joint_pos(positive_pos, wait_time=wait_time)

    # Move negative
    logger.info(f"Moving head {joint_names[joint_idx]} negative ({-step_size} rad)")
    head.set_joint_pos(negative_pos, wait_time=wait_time)

    # Return to zero
    logger.info(f"Moving head {joint_names[joint_idx]} to zero")
    head.set_joint_pos(zero_pos, wait_time=wait_time)


def main(
    step_size: float = 0.5,
) -> None:
    """Move robot head to zero position and then move each joint individually.

    Args:
        step_size: Size of the joint movement in radians
        wait_time: Time to wait between movements in seconds
    """
    # Initialize robot
    bot = Robot()
    head = bot.head

    try:
        # Initial head position (slightly tilted down)
        initial_pos = np.array([-np.pi / 6, 0.0, 0.0])
        logger.info(f"Moving head to initial position: {np.rad2deg(initial_pos)} deg")
        head.set_joint_pos(initial_pos, wait_time=1.0)

        # Define joint names and movement sequence
        joint_names = ["yaw (left/right)", "pitch (up/down)", "roll"]

        # Move each joint through sequence
        for joint_idx in range(3):
            logger.info(
                f"Starting movement sequence for joint {joint_idx} ({joint_names[joint_idx]})"
            )
            move_joint_sequence(head, joint_idx, step_size)

        logger.info("Movement sequence completed")

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    finally:
        head.set_joint_pos(np.zeros(3))  # Reset head position
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
