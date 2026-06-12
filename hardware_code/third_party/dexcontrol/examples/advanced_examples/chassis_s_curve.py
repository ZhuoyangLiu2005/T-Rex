# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script demonstrating S-curve chassis movement control.

This script shows how to control a robot chassis to follow an S-curve trajectory by
modulating steering angles in a sinusoidal pattern while maintaining constant velocity.
"""

import time

import numpy as np
import tyro
from loguru import logger

from dexcontrol.core.chassis import Chassis
from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models


def execute_s_curve(
    chassis: Chassis,
    amplitude: float,
    frequency: float,
    velocity: float,
    duration: float,
    control_hz: float = 100.0,
) -> None:
    """Executes an S-curve movement pattern with the chassis.

    Controls the robot to follow a sinusoidal steering pattern while moving at constant
    velocity, creating an S-shaped trajectory.

    Args:
        chassis: Robot chassis controller instance.
        amplitude: Maximum steering angle in radians.
        frequency: Oscillation frequency of the steering angle in Hz.
        velocity: Forward velocity of the chassis in m/s.
        duration: Total movement duration in seconds.
        control_hz: Control loop frequency in Hz.
    """
    wheel_velocities = np.full(2, velocity, dtype=np.float32)
    start_time = time.time()
    dt = 1 / control_hz

    while time.time() - start_time < duration:
        t = time.time() - start_time
        # Calculate steering angles for both wheels using sine wave
        steering_angle = amplitude * np.sin(2 * np.pi * frequency * t)
        steering_angles = np.full(2, steering_angle, dtype=np.float32)

        # Update chassis control
        chassis.set_motion_state(
            steering_angles,
            wheel_velocities,
            wait_time=dt,
            wait_kwargs={"control_hz": control_hz},
        )
        logger.debug(
            f"t={t:.2f}s: steering={steering_angle:.3f}rad, velocity={velocity:.2f}m/s"
        )


@supported_models("vega_1", "vega_1p")
def main(
    amplitude: float = np.pi / 5,
    frequency: float = 0.5,
    velocity: float = 0.5,
    duration: float = 3.0,
    control_hz: float = 100.0,
) -> None:
    """Runs the S-curve movement demonstration.

    Args:
        amplitude: Maximum steering angle in radians.
        frequency: Oscillation frequency in Hz.
        velocity: Forward velocity in m/s.
        duration: Movement duration in seconds.
        control_hz: Control loop frequency in Hz.
    """
    logger.warning("Warning: Ensure adequate space around chassis before proceeding!")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    bot = Robot()
    chassis = bot.chassis

    try:
        logger.info("Initiating S-curve movement...")
        execute_s_curve(
            chassis=chassis,
            amplitude=amplitude,
            frequency=frequency,
            velocity=velocity,
            duration=duration,
            control_hz=control_hz,
        )

        logger.info("Movement completed successfully")
        logger.info(f"Final steering angles: {chassis.steering_angle}")
        logger.info(f"Final wheel velocities: {chassis.wheel_velocity}")

    except KeyboardInterrupt:
        logger.warning("Movement interrupted by user")
    finally:
        chassis.stop()
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
