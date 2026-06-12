# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Shared constants and helpers for arm tracking benchmarks."""

import numpy as np
from loguru import logger

NUM_JOINTS = 7
ZERO_POS = np.array([0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0])
ZERO_TOLERANCE = 0.05  # rad


def verify_zero_position(arm) -> None:
    """Read current joint positions and verify they match ZERO_POS.

    Args:
        arm: Arm component to read joint positions from.

    Raises:
        RuntimeError: If any joint deviates from ZERO_POS by more than
            ZERO_TOLERANCE.
    """
    actual = arm.get_joint_pos()
    errors = np.abs(actual - ZERO_POS)

    # Log actual positions so the user can see them
    pos_str = ", ".join(f"{v:.4f}" for v in actual)
    err_str = ", ".join(f"{v:.4f}" for v in errors)
    logger.info(f"Actual joint positions: [{pos_str}]")
    logger.info(f"Position errors:        [{err_str}]")

    bad = np.where(errors > ZERO_TOLERANCE)[0]
    if len(bad) > 0:
        details = ", ".join(
            f"joint {j}: actual={actual[j]:.4f}, err={errors[j]:.4f} rad" for j in bad
        )
        raise RuntimeError(
            f"Zero-position check failed (tolerance={ZERO_TOLERANCE} rad). "
            f"Failed joints: {details}"
        )
    logger.info("Zero position verified (all joints within tolerance)")
