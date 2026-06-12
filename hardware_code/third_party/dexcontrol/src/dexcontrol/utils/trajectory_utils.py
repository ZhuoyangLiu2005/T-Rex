# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Trajectory utility functions for smooth motion generation."""

import numpy as np


def generate_linear_trajectory(
    current_pos: np.ndarray,
    target_pos: np.ndarray,
    max_vel: float | np.ndarray = 0.5,
    control_hz: float = 100,
) -> tuple[np.ndarray, int]:
    """Generate a linear trajectory between current and target positions.

    Args:
        current_pos: Current position array.
        target_pos: Target position array.
        max_vel: Maximum velocity in units per second. Can be:
            - float: Same velocity limit for all dimensions
            - numpy array: Per-dimension velocity limits (same length as current_pos)
        control_hz: Control frequency in Hz.

    Returns:
        Tuple containing:
        - trajectory: Array of waypoints from current to target position.
        - num_steps: Number of steps in the trajectory.
    """
    # Calculate time needed for each dimension
    pos_diff = np.abs(target_pos - current_pos)

    if isinstance(max_vel, np.ndarray):
        # Per-dimension velocity limits - find the dimension that takes longest
        time_needed = pos_diff / max_vel
        max_time = np.max(time_needed)
    else:
        # Single velocity limit for all dimensions
        max_diff = np.max(pos_diff)
        max_time = max_diff / max_vel

    num_steps = int(max_time * control_hz)

    # Ensure at least one step
    num_steps = max(1, num_steps)

    # Generate trajectory with endpoints (exclude the starting point in the return)
    trajectory = np.linspace(current_pos, target_pos, num_steps + 1, endpoint=True)[1:]

    return trajectory, num_steps
