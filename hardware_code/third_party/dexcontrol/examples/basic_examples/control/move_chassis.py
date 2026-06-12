# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script demonstrating basic robot chassis control.

This script demonstrates controlling a robot's chassis movements by executing a sequence
of basic motions including forward/backward translation, sideways strafing, and rotation.
The script uses the Robot class to access and control the chassis subsystem.

Example:
    To run with default parameters:
        $ python move_chassis.py

    To specify custom speed and duration:
        $ python move_chassis.py --speed 0.3 --duration 3.0
"""

import tyro
from loguru import logger

from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models


@supported_models("vega_1", "vega_1p")
def main(
    speed: float = 0.2,
    duration: float = 4.0,
) -> None:
    """Executes a sequence of chassis movements to demonstrate basic motion control.

    The sequence consists of:
    1. Forward translation
    2. Backward translation
    3. Left strafe
    4. Right strafe
    5. Counter-clockwise rotation
    6. Clockwise rotation

    Args:
        speed: Linear velocity for translations (m/s) or angular velocity for
            rotations (rad/s). Defaults to 0.2.
        duration: Time to maintain each movement in seconds. Defaults to 4.0.
    """
    logger.warning("Warning: Be ready to press e-stop if needed!")
    logger.warning("Please ensure adequate clearance around robot before proceeding.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    with Robot() as bot:
        chassis = bot.chassis

        # Forward and backward movement
        logger.info("Moving forward...")
        chassis.move_straight(speed, wait_time=duration)
        logger.info("Moving backward...")
        chassis.move_straight(-speed, wait_time=duration)

        # Sideways movement
        logger.info("Strafing left...")
        chassis.move_sideways(speed, wait_time=duration)
        logger.info("Strafing right...")
        chassis.move_sideways(-speed, wait_time=duration)

        # Rotational movement
        logger.info("Rotating counter-clockwise...")
        chassis.turn(speed, wait_time=duration)
        logger.info("Rotating clockwise...")
        chassis.turn(-speed, wait_time=duration)

        logger.info("Movement sequence completed.")


if __name__ == "__main__":
    tyro.cli(main)
