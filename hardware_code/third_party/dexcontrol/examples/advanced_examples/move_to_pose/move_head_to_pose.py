#!/usr/bin/env python3
# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script for moving the robot's head to a predefined pose.

This script demonstrates how to move the robot's head to a predefined pose
from the pose library, with an option to compensate for torso pitch.
It includes safety prompts and proper shutdown procedures.
"""

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main(
    pose: str = "home",
    comp_pitch: bool = False,
) -> None:
    """Moves the robot's head to a predefined pose.

    Args:
        pose: Name of the predefined pose to move to.
        comp_pitch: Whether to compensate for torso pitch angle.
    """
    # Safety confirmation
    logger.warning(
        "Warning: Be ready to press e-stop if needed! "
        "This example does not check for self-collisions."
    )
    logger.warning("Please ensure the head and torso have sufficient space to move.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    # Initialize robot
    bot = Robot()
    head = bot.head

    try:
        logger.info(f"Moving head to {pose} position")
        if comp_pitch:
            # Adjust pose for torso pitch and move
            torso_pitch = bot.torso.pitch_angle
            logger.debug(f"Current torso pitch: {np.rad2deg(torso_pitch):.2f} degrees")

            adjusted_pose = bot.compensate_torso_pitch(
                head.get_predefined_pose(pose),
                "head",
            )
            head.set_joint_pos(adjusted_pose, wait_time=6.0)
        else:
            # Move directly to predefined pose
            head.go_to_pose(pose, wait_time=6.0)
    finally:
        logger.info("Shutting down robot")
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
