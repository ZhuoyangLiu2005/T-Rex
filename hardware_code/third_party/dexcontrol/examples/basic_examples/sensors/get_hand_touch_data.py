# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to display touch sensor data from the robot's hands.

This script demonstrates how to read and display touch sensor data from
the robot hands. Only V2 hands have touch sensors available.
"""

import time

import tyro
from dexbot_utils import HandType
from loguru import logger

from dexcontrol.robot import Robot


def print_hand_sensor_data(robot: Robot, arm_side: str) -> None:
    """Prints the current touch sensor data from the specified hand.

    Args:
        robot: Robot instance to read sensor data from.
        arm_side: Which arm to check ('left' or 'right').

    Returns:
        None
    """
    # Get hand types from robot

    hand_attr = f"{arm_side}_hand"
    hand = getattr(robot, hand_attr)

    finger_forces = hand.get_finger_tip_force()

    logger.info(f"\n{arm_side.upper()} HAND TOUCH SENSOR DATA:")
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Little"]

    for name, force in zip(finger_names, finger_forces):
        logger.info(f"  {name:6}: {force:8.4f} N")


def main(arm_side: str = "left") -> None:
    """Continuously displays touch sensor data from the specified hand.

    Args:
        arm_side: Which arm to monitor ('left' or 'right'). Defaults to 'left'.
    """
    if arm_side not in ["left", "right"]:
        logger.error(f"Invalid arm_side: {arm_side}. Must be 'left' or 'right'")
        return

    logger.info(f"Starting hand touch sensor monitoring for {arm_side} hand...")
    logger.info("Note: Touch sensor data is only available on HandF5D6_V2 hands")

    robot = Robot()

    hand_types = robot.query_hand_type()
    if hand_types.get(arm_side) != HandType.HandF5D6_V2:
        raise ValueError(
            f"Touch sensor data only available for V2 hands. Current hand type: {hand_types.get(arm_side)}"
        )

    try:
        while True:
            # The internal data update rate is 50Hz, for demo purpose we only use 20Hz
            print_hand_sensor_data(robot, arm_side)
            time.sleep(0.05)
    except KeyboardInterrupt:
        logger.info("Stopping hand sensor monitoring...")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
