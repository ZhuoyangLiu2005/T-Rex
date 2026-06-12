# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to demonstrate relative torso control.

This script shows how to implement relative torso joint control using the
relative=True parameter. It moves a specified torso joint by a relative angle
while ensuring the arms are in a safe folded position.
"""

from typing import Final

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models


@supported_models("vega_1", "vega_1p")
def main(
    joint_idx: int = 2,
) -> None:
    """Executes a relative movement of a specified torso joint.

    The script first ensures arms are folded and hands closed for safety, then
    moves the torso to a medium crouch position before executing the relative
    joint movement.

    Args:
        joint_idx: Index of the torso joint to move (0-2).

    Raises:
        ValueError: If joint_idx is not in range [0,2].
    """
    if not 0 <= joint_idx <= 2:
        raise ValueError("joint_idx must be between 0 and 2")

    # Safety confirmation
    logger.warning("Warning: Be ready to press e-stop if needed!")
    logger.warning("Please ensure adequate clearance around robot before proceeding.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    # Initialize robot and move to safe starting position
    bot = Robot()

    # Ensure arms are folded
    if not (
        bot.left_arm.is_pose_reached("folded")
        and bot.right_arm.is_pose_reached("folded")
    ):
        bot.set_joint_pos(
            {
                "left_arm": bot.left_arm.get_predefined_pose("folded"),
                "right_arm": bot.right_arm.get_predefined_pose("folded"),
            },
            wait_time=10.0,
        )

    # Close hands and set initial torso position
    if bot.have_hand("left"):
        bot.left_hand.close_hand()
    if bot.have_hand("right"):
        bot.right_hand.close_hand()
    bot.torso.go_to_pose("crouch45_medium", wait_time=5.0)

    # Execute relative joint movement
    RELATIVE_ANGLE: Final[float] = np.deg2rad(25)
    delta_pos = np.zeros(3)
    delta_pos[joint_idx] = RELATIVE_ANGLE
    bot.torso.set_joint_pos(delta_pos, wait_time=5.0, relative=True)

    logger.info(
        f"Final torso pitch angle: {np.rad2deg(bot.torso.pitch_angle):.2f} degrees"
    )

    bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
