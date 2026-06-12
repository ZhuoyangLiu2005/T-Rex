# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to get robot joint positions.

This script demonstrates how to retrieve joint position information from different
parts of the robot including arms, hands, head, torso and chassis. It initializes
a robot instance and queries the joint positions of each component before cleanly
shutting down.
"""

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main() -> None:
    """Gets and logs joint positions from all robot components.

    Creates a robot instance, retrieves joint positions from each component
    (arms, hands, head, torso, chassis), logs the positions, and performs
    a clean shutdown.
    """
    # Initialize robot with default configuration
    with Robot() as bot:
        # Get joint positions from all available components
        joint_positions = {
            "Left arm": (bot.left_arm.get_joint_pos(), bot.left_arm.get_timestamp_ns()),
            "Right arm": (
                bot.right_arm.get_joint_pos(),
                bot.right_arm.get_timestamp_ns(),
            ),
        }
        if bot.has_component("head"):
            joint_positions["Head"] = (
                bot.head.get_joint_pos(),
                bot.head.get_timestamp_ns(),
            )
        if bot.has_component("torso"):
            joint_positions["Torso"] = (
                bot.torso.get_joint_pos(),
                bot.torso.get_timestamp_ns(),
            )
        if bot.has_component("chassis"):
            joint_positions["Chassis_Steer"] = (
                bot.chassis.chassis_steer.get_joint_pos(),
                bot.chassis.chassis_steer.get_timestamp_ns(),
            )
            joint_positions["Chassis_Drive"] = (
                bot.chassis.chassis_drive.get_joint_pos(),
                bot.chassis.chassis_drive.get_timestamp_ns(),
            )
        if bot.have_hand("left"):
            joint_positions["Left hand"] = (
                bot.left_hand.get_joint_pos(),
                bot.left_hand.get_timestamp_ns(),
            )
        if bot.have_hand("right"):
            joint_positions["Right hand"] = (
                bot.right_hand.get_joint_pos(),
                bot.right_hand.get_timestamp_ns(),
            )

        # Log joint positions
        for component, (position, timestamp) in joint_positions.items():
            logger.info(
                f"{component} joint position: {np.round(position, 3)}, timestamp: {timestamp}"
            )


if __name__ == "__main__":
    tyro.cli(main)
