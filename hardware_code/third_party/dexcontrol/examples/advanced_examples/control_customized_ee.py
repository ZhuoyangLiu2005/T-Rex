# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script demonstrating customized end-effector control via pass-through.

This script shows how to send raw RS485 messages to a customized end effector
and receive responses. This is useful for controlling custom end effectors
that are not natively supported by the robot.

NOTE: This only works when the end effector type is UNKNOWN. If a Hand (F5D6)
or DexGripper is detected, EE pass-through is disabled.

Example use case: Robotiq gripper activation via Modbus RTU.
"""

import time
from typing import Literal

import tyro
from loguru import logger

from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models


@supported_models("vega_1", "vega_1p", "vega_1u")
def main(
    side: Literal["left", "right", "both"] = "left",
    message_hex: str = "09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1",
    wait_response: bool = True,
    timeout: float = 1.0,
) -> None:
    """Send a pass-through message to the customized end effector.

    Args:
        side: Which arm's end effector to communicate with.
        message_hex: Hex string of the message to send (space-separated bytes).
            Default is Robotiq activation command.
        wait_response: Whether to wait for and display the response.
        timeout: Maximum time to wait for response in seconds.
    """
    with Robot() as bot:
        # Convert hex string to bytes
        message = bytes.fromhex(message_hex.replace(" ", ""))
        logger.info(f"Sending message: {message.hex(' ')}")

        if side in ("left", "both"):
            if bot.left_arm.enable_ee_pass_through:
                logger.info("Sending to left arm EE...")
                bot.left_arm.send_ee_pass_through_message(message)

                if wait_response:
                    start = time.time()
                    while time.time() - start < timeout:
                        response = bot.left_arm.get_ee_pass_through_response()
                        if response is not None:
                            logger.info(f"Left arm EE response: {response}")
                            break
                        time.sleep(0.01)
                    else:
                        logger.warning("Left arm: No response received within timeout")
            else:
                logger.warning(
                    "Left arm: EE pass-through not enabled (known EE type detected)"
                )

        if side in ("right", "both"):
            if bot.right_arm.enable_ee_pass_through:
                logger.info("Sending to right arm EE...")
                bot.right_arm.send_ee_pass_through_message(message)

                if wait_response:
                    start = time.time()
                    while time.time() - start < timeout:
                        response = bot.right_arm.get_ee_pass_through_response()
                        if response is not None:
                            logger.info(f"Right arm EE response: {response}")
                            break
                        time.sleep(0.01)
                    else:
                        logger.warning("Right arm: No response received within timeout")
            else:
                logger.warning(
                    "Right arm: EE pass-through not enabled (known EE type detected)"
                )


if __name__ == "__main__":
    tyro.cli(main)
