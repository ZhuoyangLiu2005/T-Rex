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

This script demonstrates basic arm control by moving to predefined positions.
"""

from typing import Literal

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main(
    side: Literal["right", "left", "both"] = "both",
    target: Literal["zero", "L_shape", "folded"] = "L_shape",
) -> None:
    """Moves robot arm(s) to a preset position and opens the hand(s).

    Args:
        side: Which arm(s) to move ("right", "left", or "both").
        target: Target position ("zero", "L_shape", or "folded").

    Raises:
        ValueError: If side is invalid.
    """
    logger.warning(
        "Warning: Be ready to press e-stop if needed! "
        "This example does not check for self-collisions."
    )
    logger.warning("Please ensure the arms have sufficient space to move.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    if side not in ["right", "left", "both"]:
        raise ValueError('side must be "right", "left", or "both"')

    bot = Robot()
    control_sides = ["left", "right"] if side == "both" else [side]

    try:
        for current_side in control_sides:
            arm = bot.left_arm if current_side == "left" else bot.right_arm

            logger.info(f"Moving {current_side} arm")

            # Get target position and adjust for torso pitch
            target_pos = arm.get_predefined_pose(target)
            body_part = "left_arm" if current_side == "left" else "right_arm"
            target_pos = bot.compensate_torso_pitch(target_pos, body_part)

            arm.set_joint_pos(target_pos, wait_time=3.0, exit_on_reach=True)
            logger.info(f"Moved {current_side} arm to {target} position")

    finally:
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
