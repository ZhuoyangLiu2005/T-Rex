# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""LIDAR sensor implementations using Zenoh subscribers.

This module provides LIDAR sensor classes that use the specialized LIDAR
subscriber for scan data.
"""

from typing import Any

import numpy as np
from dexcomm import Node
from dexcomm.codecs import LidarScan2DCodec


class RPLidarSensor:
    """LIDAR sensor using Zenoh subscriber.

    This sensor provides LIDAR scan data using the LidarSubscriber
    for efficient data handling with lazy decoding.
    """

    def __init__(
        self,
        name,
        configs,
    ) -> None:
        """Initialize the LIDAR sensor.

        Args:
            configs: Configuration for the LIDAR sensor.
        """
        self._name = name
        self._node = Node(name=self._name)
        # Create the LIDAR subscriber
        self._subscriber = self._node.create_subscriber(
            callback=None,
            decoder=LidarScan2DCodec.decode,
            topic=configs.topic
        )


    def shutdown(self) -> None:
        """Shutdown the LIDAR sensor."""
        self._subscriber.shutdown()

    def is_active(self) -> bool:
        """Check if the LIDAR sensor is actively receiving data.

        Returns:
            True if receiving data, False otherwise.
        """
        return self._subscriber.is_active(0.5)

    def wait_for_active(self, timeout: float = 5.0) -> bool:
        """Wait for the LIDAR sensor to start receiving data.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if sensor becomes active, False if timeout is reached.
        """
        msg = self._subscriber.wait_for_message(timeout)
        return msg is not None

    def get_obs(self) -> dict[str, Any] | None:
        """Get the latest LIDAR scan data.

        Returns:
            Latest scan data dictionary if available, None otherwise.
            Dictionary contains:
                - ranges: Array of range measurements in meters
                - angles: Array of corresponding angles in radians
                - qualities: Array of quality values (0-255) if available, None otherwise
                - timestamp: Timestamp in nanoseconds (int)
        """
        data = self._subscriber.get_latest()
        return data

    def get_ranges(self) -> np.ndarray | None:
        """Get the latest range measurements.

        Returns:
            Array of range measurements in meters if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        return data['ranges'] if data else None

    def get_angles(self) -> np.ndarray | None:
        """Get the latest angle measurements.

        Returns:
            Array of angle measurements in radians if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        return data['angles'] if data else None

    def get_qualities(self) -> np.ndarray | None:
        """Get the latest quality measurements.

        Returns:
            Array of quality values (0-255) if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        return data['intensities'] if data else None

    def get_point_count(self) -> int:
        """Get the number of points in the latest scan.

        Returns:
            Number of points in the scan, 0 if no data available.
        """
        ranges = self.get_ranges()
        if ranges is not None:
            return len(ranges)
        return 0

    @property
    def name(self) -> str:
        """Get the LIDAR name.

        Returns:
            LIDAR name string.
        """
        return self._name
