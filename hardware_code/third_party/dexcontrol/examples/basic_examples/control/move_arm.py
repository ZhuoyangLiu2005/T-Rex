# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to control robot arm movements.

This script demonstrates basic arm control by moving to zero position and then
moving each joint individually. The arm moves through a sequence of positions
while maintaining safety and control.
"""

from typing import Literal

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main(
    arm_side: Literal["right", "left"] = "right",
    step_size: float = 0.2,
) -> None:
    """Executes a sequence of arm movements to demonstrate joint control.

    The arm first moves to a zero position, then sequentially moves each joint
    by the specified step size before returning it to zero.

    Args:
        arm_side: Which arm to move ("right" or "left").
        step_size: Magnitude of joint movement in radians.

    Raises:
        ValueError: If arm_side is not "right" or "left".
    """
    # Initialize robot and control parameters
    logger.warning("Warning: Be ready to press e-stop if needed!")
    logger.warning("Please make sure the arms have enough space to move.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    bot = Robot()

    # Validate input and select appropriate arm
    if arm_side not in ["right", "left"]:
        raise ValueError('arm_side must be "right" or "left"')

    arm = bot.left_arm if arm_side == "left" else bot.right_arm
    logger.info(f"Initializing movement sequence for {arm_side} arm")

    try:
        # Move to initial zero position
        logger.info("Moving to zero position")
        arm.set_joint_pos(np.zeros(7), wait_time=4.0)

        # Sequentially move each joint
        for joint_idx in range(7):
            logger.info(f"Moving joint {joint_idx}")

            target_pos = np.zeros(7)
            target_pos[joint_idx] = step_size
            arm.set_joint_pos(target_pos, wait_time=3.0)

            target_pos[joint_idx] = 0
            arm.set_joint_pos(target_pos, wait_time=3.0)

        logger.info("Movement sequence completed successfully")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
    finally:
        logger.info("Shutting down robot")
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
