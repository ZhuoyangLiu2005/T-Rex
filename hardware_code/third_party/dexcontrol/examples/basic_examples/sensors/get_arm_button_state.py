# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to display button states from the robot's wrench sensors.

This script demonstrates how to read and display the states of the blue and green
buttons on the wrench sensors attached to the robot arms.
"""

import time

import tyro

from dexcontrol.robot import Robot


def print_button_state(robot: Robot, arm_side: str) -> None:
    """Prints the current state of buttons on the specified arm's wrench sensor.

    Args:
        robot: Robot instance to read button states from.
        arm_side: Which arm to check ('left' or 'right').

    Returns:
        None
    """
    arm = robot.left_arm if arm_side == "left" else robot.right_arm

    if arm.wrench_sensor is None:
        print(f"\n{arm_side.upper()} ARM: No wrench sensor detected")
        return

    button_state = arm.wrench_sensor.get_button_state()

    print(f"\n{arm_side.upper()} ARM:")
    print(f"Blue button: {button_state['blue_button']}")
    print(f"Green button: {button_state['green_button']}")
    print("\n" + "-" * 50)


def main(arm_side: str = "left") -> None:
    """Continuously displays button states from the specified arm's wrench sensor.

    Args:
        arm_side: Which arm to monitor ('left' or 'right'). Defaults to 'left'.
    """
    robot = Robot()

    try:
        while True:
            print_button_state(robot, arm_side)
            time.sleep(0.05)  # 20Hz update rate
    except KeyboardInterrupt:
        robot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
