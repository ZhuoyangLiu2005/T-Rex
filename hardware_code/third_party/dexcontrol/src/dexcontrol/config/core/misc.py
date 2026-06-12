# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Miscellaneous robot component configurations."""

import os
from dataclasses import dataclass

from dexcontrol.utils.constants import (
    DISABLE_ESTOP_CHECKING_ENV_VAR,
    DISABLE_HEARTBEAT_ENV_VAR,
)


@dataclass
class BatteryConfig:
    _target_: str = "dexcontrol.core.misc.Battery"
    state_sub_topic: str = "state/bms"


@dataclass
class EStopConfig:
    _target_: str = "dexcontrol.core.misc.EStop"
    state_sub_topic: str = "state/estop"
    estop_query_name: str = "system/estop"
    enabled: bool = True  # Can be disabled via DEXCONTROL_DISABLE_ESTOP_CHECKING=1

    def __post_init__(self):
        """Check environment variable to disable estop checking."""
        if os.getenv(DISABLE_ESTOP_CHECKING_ENV_VAR, "0").lower() in (
            "1",
            "true",
            "yes",
        ):
            self.enabled = False


@dataclass
class HeartbeatConfig:
    _target_: str = "dexcontrol.core.misc.Heartbeat"
    heartbeat_topic: str = "/heartbeat"
    timeout_seconds: float = 1.0
    enabled: bool = True  # Can be disabled via DEXCONTROL_DISABLE_HEARTBEAT=1

    def __post_init__(self):
        """Check environment variable to disable heartbeat monitoring."""
        if os.getenv(DISABLE_HEARTBEAT_ENV_VAR, "0").lower() in ("1", "true", "yes"):
            self.enabled = False
