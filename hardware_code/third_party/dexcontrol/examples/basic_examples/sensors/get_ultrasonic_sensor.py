# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to read ultrasonic sensor data.

This script demonstrates how to retrieve distance measurements from the robot's
ultrasonic sensor. It initializes a robot instance, reads the sensor value, and
performs a clean shutdown.
"""

import tyro
from loguru import logger

from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models


@supported_models("vega_1")
def main() -> None:
    """Gets and logs ultrasonic sensor reading.

    Creates a robot instance, retrieves the current distance measurement from
    the ultrasonic sensor, logs the value, and performs a clean shutdown.

    The ultrasonic sensor returns distance measurements in meters.
    """
    configs = get_robot_config()
    configs.enable_sensor("ultrasonic")
    with Robot(configs=configs) as bot:
        # Get and log ultrasonic sensor reading
        distance = bot.sensors.ultrasonic.get_obs()
        logger.info(f"Ultrasonic sensor distance (m): {distance}")


if __name__ == "__main__":
    tyro.cli(main)
