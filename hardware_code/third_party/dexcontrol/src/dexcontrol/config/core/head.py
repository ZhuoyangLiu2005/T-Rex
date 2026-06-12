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
class HeadConfig:
    _target_: str = "dexcontrol.core.head.Head"
    state_sub_topic: str = "state/head"
    control_pub_topic: str = "control/head"
    set_mode_query: str = "mode/head"
    joint_name: list[str] = field(
        default_factory=lambda: ["head_j1", "head_j2", "head_j3"]
    )
    dof: int = 3
    joint_limit: list[list[float]] = field(
        default_factory=lambda: [
            [-1.2217, 1.2217],
            [-2.7925, 2.7925],
            [-1.396, 1.396],
        ]
    )
    joint_vel_limit: list[float] = field(default_factory=lambda: [2.5, 2.5, 2.5])
    pose_pool: dict[str, list[float]] = field(
        default_factory=lambda: {
            "home": [0.0, 0.0, 0.0],
            "tucked": [0.0, 0.0, -1.37],
        }
    )
