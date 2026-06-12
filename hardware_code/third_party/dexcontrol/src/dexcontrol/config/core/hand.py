# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

from dataclasses import dataclass, field


@dataclass
class HandConfig:
    _target_: str = "dexcontrol.core.hand.HandF5D6"
    state_sub_topic: str = "state/hand/right"
    control_pub_topic: str = "control/hand/right"
    touch_sensor_sub_topic: str = "state/hand/right/touch"  # Only for V2 hand
    dof: int = 6
    joint_name: list[str] = field(
        default_factory=lambda: [
            "R_th_j1",
            "R_ff_j1",
            "R_mf_j1",
            "R_rf_j1",
            "R_lf_j1",
            "R_th_j0",
        ]
    )

    pose_pool: dict[str, list[float]] = field(
        default_factory=lambda: {
            "open": [0.1834, 0.2891, 0.2801, 0.284, 0.2811, -0.0158],
            "close": [-0.2668, -1.0946, -1.0844, -1.0154, -1.0118, 1.6],
        }
    )
