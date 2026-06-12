# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script for basic joint level admittance control
without system identification.

This script shows a simple joint level admittance control that does not
require dedicated system identification. For optimal performance, it is recommended to:
1. Perform proper system identification
2. Run this code on a Jetson or PC connected to the robot via ethernet, as admittance
   control performance is sensitive to network latency
"""

import numpy as np
import tyro
from dexcomm.utils import RateLimiter
from loguru import logger

from dexcontrol.core.arm import Arm
from dexcontrol.robot import Robot


class JointAdmittanceController:
    """Joint-level admittance controller for robot arm force control.

    This controller implements a simple joint-level admittance control scheme that
    allows    the robot to respond to external forces at each joint while optionally
    maintaining desired joint positions.
    """

    def __init__(
        self,
        kd_gain: float = 1.0,
        dt: float = 0.01,
        zero_force_control: bool = False,
    ) -> None:
        """Initialize the joint admittance controller.

        Args:
            kd_gain: Gain multiplier for stiffness and damping matrices.
            dt: Control loop time step in seconds.
            zero_force_control: If True, reduces stiffness for zero-force mode.
        """
        self.is_zero_force_control = zero_force_control
        # 7x7 matrices for 7-DOF arm joint control
        self.K = np.diag([2.0, 0.25, 0.25, 2.0, 0.25, 0.25, 0.25]) * kd_gain
        self.D = np.diag([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) * kd_gain
        self.dt = dt
        if zero_force_control:
            self.K = self.K / 75

    def get_admittance_joint_pos(
        self,
        joint_pos_cur: np.ndarray,
        joint_forces_ext: np.ndarray,
        joint_pos_des: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the admittance-controlled joint positions.

        Args:
            joint_pos_cur: Current joint positions as 7-element array [rad].
            joint_forces_ext: External joint forces as 7-element array [Nm].
            joint_pos_des: Desired joint positions as 7-element array [rad]. If None,
                          only external forces are considered.

        Returns:
            Corrected joint positions as 7-element array [rad].
        """
        if joint_pos_des is None:
            virtual_torques = np.zeros(7)
        else:
            joint_pos_error = joint_pos_des - joint_pos_cur
            virtual_torques = self.K @ joint_pos_error

        # Add external forces to virtual torques
        total_torques = virtual_torques + joint_forces_ext

        # Compute joint velocity correction using admittance dynamics
        joint_vel_correction = np.linalg.inv(self.D) @ total_torques

        # Integrate to get position correction
        joint_pos_correction = joint_vel_correction * self.dt

        # Apply correction to current joint positions
        corrected_joint_pos = joint_pos_cur + joint_pos_correction

        return corrected_joint_pos


def __get_joint_force_from_current(
    current_cur: np.ndarray,
    current_initial: np.ndarray,
    force_thresholds: np.ndarray,
) -> np.ndarray:
    """Get joint force readings from current offset compensation.

    Args:
        current_cur: Current joint current readings as 7-element array [A].
        current_initial: Initial joint current readings for offset compensation [A].
        force_thresholds: Thresholds for force activation as 7-element array [A].
            Current readings below these thresholds are ignored for each joint.

    Returns:
        Joint forces as 7-element array [Nm] corresponding to the 7 arm joints.
    """
    # Compute current offset (current - initial)
    current_offset = current_cur - current_initial

    # Apply thresholds and convert to force
    joint_forces = np.zeros(7)

    for i in range(7):
        offset = current_offset[i]
        threshold = force_thresholds[i]

        if abs(offset) < threshold:
            # Below threshold - no force
            joint_forces[i] = 0.0
        else:
            # Linear mapping above threshold (invert sign for compliance)
            joint_forces[i] = -offset

    return joint_forces


def _get_initial_joint_positions(bot: Robot) -> dict[str, np.ndarray]:
    """Get initial joint positions for both arms.

    Args:
        bot: Robot instance.

    Returns:
        Dictionary mapping arm names to initial joint positions.
    """
    return {
        "left": np.array(bot.left_arm.get_joint_pos()),
        "right": np.array(bot.right_arm.get_joint_pos()),
    }


def _get_initial_joint_currents(bot: Robot) -> dict[str, np.ndarray]:
    """Get initial joint current readings for offset compensation.

    Args:
        bot: Robot instance.

    Returns:
        Dictionary mapping arm names to initial joint current readings.
    """
    return {
        "left": np.array(bot.left_arm.get_joint_current()),
        "right": np.array(bot.right_arm.get_joint_current()),
    }


def _update_joint_positions_with_admittance(
    current_joint_pos: dict[str, np.ndarray],
    current_joint_forces: dict[str, np.ndarray],
    admittance_controller: JointAdmittanceController,
    init_joint_pos: dict[str, np.ndarray],
    zero_force: bool,
) -> dict[str, np.ndarray]:
    """Update joint positions using pre-calculated joint forces.

    Args:
        current_joint_pos: Dictionary of current joint positions.
        current_joint_forces: Dictionary of current joint forces.
        admittance_controller: Joint admittance controller instance.
        init_joint_pos: Dictionary of initial joint positions.
        zero_force: Whether to use zero-force mode.

    Returns:
        Dictionary of updated joint positions.
    """
    updated_joint_pos = {}

    for arm in ("left", "right"):
        # Compute new joint positions using admittance control
        new_joint_pos = admittance_controller.get_admittance_joint_pos(
            joint_pos_cur=current_joint_pos[arm],
            joint_forces_ext=current_joint_forces[arm],
            joint_pos_des=None if zero_force else init_joint_pos[arm],
        )

        updated_joint_pos[arm] = new_joint_pos

    return updated_joint_pos


def _send_joint_commands(
    target_joint_pos: dict[str, np.ndarray],
    arms: dict[str, Arm],
) -> None:
    """Send joint position commands to robot arms.

    Args:
        target_joint_pos: Dictionary of target joint positions.
        arms: Dictionary of arm objects.
    """
    for arm in ("left", "right"):
        arms[arm].set_joint_pos(target_joint_pos[arm].tolist())


def main(
    zero_force: bool = True,
    kd_gain: float = 0.1,
    control_hz: float = 200.0,
    force_thresholds: list[float] = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
) -> None:
    """Main function for joint-level admittance control demo.

    This function demonstrates joint-level admittance control with two modes:
    1. Zero force control (default): The robot arm moves only in response to external
       forces detected at each joint, without trying to reach any target positions.
    2. Initial position admittance control: The robot arm tries to return to initial
       joint positions while also responding to external forces.

    Args:
        zero_force: If True, enables zero force control mode. If False, maintain
            desired joint positions while responding to external forces.
        kd_gain: Gain for the admittance controller. The larger the gain, the stiffer
            the robot arm will be under external forces.
        control_hz: Control loop frequency in Hz.
        force_thresholds: List of thresholds for force activation [A] for each joint.
            Current readings below these thresholds are ignored for each joint.
            Default is [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25] for all 7 joints.
    """
    if kd_gain < 0.3:
        logger.warning("kd_gain is too small. Setting to 0.3.")
        kd_gain = 0.3

    # Convert force thresholds to numpy array
    force_thresholds_array = np.array(force_thresholds)
    if len(force_thresholds_array) != 7:
        raise ValueError(
            "force_thresholds must have exactly 7 elements for the 7 arm joints"
        )

    # Initialize robot and components
    bot = Robot()
    arms = {"left": bot.left_arm, "right": bot.right_arm}

    # Get initial joint positions and currents
    init_joint_pos = _get_initial_joint_positions(bot)
    init_joint_currents = _get_initial_joint_currents(bot)

    # Initialize joint admittance controller
    dt = 1.0 / control_hz
    admittance_controller = JointAdmittanceController(
        zero_force_control=zero_force,
        kd_gain=kd_gain,
        dt=dt,
    )

    # Setup rate limiter
    rate_limiter = RateLimiter(rate_hz=control_hz)

    logger.info(f"Joint admittance control started at {control_hz} Hz.")
    logger.info(f"Zero force mode: {zero_force}")
    logger.info(f"Force thresholds: {force_thresholds_array}A")

    # Initialize force calculation variables
    current_joint_forces = {
        "left": np.zeros(7),
        "right": np.zeros(7),
    }

    try:
        while True:
            # Get current joint positions and currents
            current_joint_pos = {
                "left": np.array(bot.left_arm.get_joint_pos()),
                "right": np.array(bot.right_arm.get_joint_pos()),
            }

            current_joint_currents = {
                "left": np.array(bot.left_arm.get_joint_current()),
                "right": np.array(bot.right_arm.get_joint_current()),
            }

            # Update forces at control frequency
            for arm in ("left", "right"):
                # Calculate new forces from current offset compensation
                new_forces = __get_joint_force_from_current(
                    current_joint_currents[arm],
                    init_joint_currents[arm],
                    force_thresholds_array,
                )
                current_joint_forces[arm] = new_forces

            # Update joint positions with admittance control using current forces
            target_joint_pos = _update_joint_positions_with_admittance(
                current_joint_pos,
                current_joint_forces,
                admittance_controller,
                init_joint_pos,
                zero_force,
            )

            # Send joint commands
            _send_joint_commands(target_joint_pos, arms)

            rate_limiter.sleep()

    except KeyboardInterrupt:
        logger.info("Joint admittance control stopped by user.")
    finally:
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
