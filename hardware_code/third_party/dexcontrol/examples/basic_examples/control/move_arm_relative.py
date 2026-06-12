# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to demonstrate relative arm control.

This script shows how to implement relative joint control by incrementally
moving each joint from its current position using the relative=True parameter.
The arm performs a sequence of relative movements rather than moving to absolute positions.
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
    """Executes a sequence of relative arm movements to demonstrate joint control.

    The arm first moves to a zero position, then sequentially moves each joint
    relatively by the specified step size using the relative=True parameter.

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
    logger.info(f"Initializing relative movement sequence for {arm_side} arm")

    try:
        # Move to initial zero position (absolute movement)
        logger.info("Moving to zero position")
        arm.set_joint_pos(np.zeros(7), wait_time=4.0)

        # Sequentially move each joint relatively
        for joint_idx in range(7):
            logger.info(f"Moving joint {joint_idx} relatively")

            # Create delta position arrays for relative movement
            delta_pos = np.zeros(7)
            delta_pos[joint_idx] = step_size

            # Move joint positively (relative movement)
            logger.info(f"Moving joint {joint_idx} by +{step_size} radians")
            arm.set_joint_pos(delta_pos, relative=True, wait_time=2.0)

            # Move joint negatively (relative movement)
            delta_neg = np.zeros(7)
            delta_neg[joint_idx] = (
                -2 * step_size
            )  # -2x to go back and further by step_size
            logger.info(f"Moving joint {joint_idx} by -{2 * step_size} radians")
            arm.set_joint_pos(delta_neg, relative=True, wait_time=2.0)

            # Return to original position (relative movement)
            logger.info(f"Returning joint {joint_idx} to original position")
            arm.set_joint_pos(delta_pos, relative=True, wait_time=2.0)

        logger.info("Relative movement sequence completed successfully")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
    finally:
        logger.info("Shutting down robot")
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
