# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""3D LIDAR sensor implementation with DexComm.

This module provides a 3D LIDAR sensor class that uses the Lidar3DCodec
for efficient point cloud data handling.
"""

from typing import Any

import numpy as np
from dexcomm import Node
from dexcomm.codecs import Lidar3DCodec


class Lidar3DSensor:
    """3D LIDAR sensor using DexComm subscriber.

    This sensor provides 3D point cloud data using the Lidar3DCodec for
    efficient data handling with lazy decoding.

    The point cloud data includes:
        - x, y, z coordinates (separate arrays)
        - intensity values
        - ring information
        - per-point timestamps
        - organized cloud metadata (height, width, is_dense)
    """

    def __init__(
        self,
        name: str,
        configs,
    ) -> None:
        """Initialize the 3D LIDAR sensor.

        Args:
            name: Name of the sensor.
            configs: Configuration for the 3D LIDAR sensor.
        """
        self._name = name
        self._node = Node(name=self._name)
        # Create the 3D LIDAR subscriber
        self._subscriber = self._node.create_subscriber(
            callback=None,
            decoder=Lidar3DCodec.decode,
            topic=configs.topic,
        )

    def shutdown(self) -> None:
        """Shutdown the 3D LIDAR sensor."""
        self._subscriber.shutdown()

    def is_active(self) -> bool:
        """Check if the 3D LIDAR sensor is actively receiving data.

        Returns:
            True if receiving data, False otherwise.
        """
        return self._subscriber.is_active(0.5)

    def wait_for_active(self, timeout: float = 5.0) -> bool:
        """Wait for the 3D LIDAR sensor to start receiving data.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if sensor becomes active, False if timeout is reached.
        """
        msg = self._subscriber.wait_for_message(timeout)
        return msg is not None

    def get_obs(self) -> dict[str, Any] | None:
        """Get the latest 3D LIDAR point cloud data.

        Returns:
            Latest point cloud data dictionary if available, None otherwise.
            Dictionary contains:
                - x: Array of x coordinates (meters)
                - y: Array of y coordinates (meters)
                - z: Array of z coordinates (meters)
                - intensity: Array of intensity values
                - ring: Array of ring/channel IDs
                - point_timestamps_ns: Per-point timestamps in nanoseconds
                - timestamp_ns: Scan timestamp in nanoseconds (int)
                - sequence: Sequence number
                - height: Point cloud height (organized clouds)
                - width: Point cloud width (organized clouds)
                - is_dense: Whether cloud has invalid points
                - point_count: Total number of points
        """
        data = self._subscriber.get_latest()
        return data

    def get_points(self) -> np.ndarray | None:
        """Get the latest point cloud as Nx3 array.

        Returns:
            Array of shape (N, 3) with xyz coordinates if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        if data is None:
            return None

        # Stack x, y, z into Nx3 array
        x = data.get("x")
        y = data.get("y")
        z = data.get("z")

        if x is None or y is None or z is None:
            return None

        return np.column_stack([x, y, z])

    def get_points_with_intensity(self) -> np.ndarray | None:
        """Get the latest point cloud with intensity as Nx4 array.

        Returns:
            Array of shape (N, 4) with xyzi if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        if data is None:
            return None

        # Stack x, y, z, intensity into Nx4 array
        x = data.get("x")
        y = data.get("y")
        z = data.get("z")
        intensity = data.get("intensity")

        if x is None or y is None or z is None or intensity is None:
            return None

        return np.column_stack([x, y, z, intensity])

    def get_xyz(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Get the latest x, y, z coordinate arrays separately.

        Returns:
            Tuple of (x, y, z) arrays if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        if data is None:
            return None

        x = data.get("x")
        y = data.get("y")
        z = data.get("z")

        if x is None or y is None or z is None:
            return None

        return (x, y, z)

    def get_intensity(self) -> np.ndarray | None:
        """Get the latest intensity measurements.

        Returns:
            Array of intensity values if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        return data["intensity"] if data else None

    def get_ring(self) -> np.ndarray | None:
        """Get the latest ring/channel information.

        Returns:
            Array of ring IDs if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        return data["ring"] if data else None

    def get_point_timestamps(self) -> np.ndarray | None:
        """Get per-point timestamps.

        Returns:
            Array of timestamps in nanoseconds if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        return data["point_timestamps_ns"] if data else None

    def get_timestamp(self) -> int | None:
        """Get the scan timestamp.

        Returns:
            Timestamp in nanoseconds if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        return data["timestamp_ns"] if data else None

    def get_point_count(self) -> int:
        """Get the number of points in the latest scan.

        Returns:
            Number of points in the scan, 0 if no data available.
        """
        data = self._subscriber.get_latest()
        if data is None:
            return 0
        return data.get("point_count", 0)

    def get_cloud_shape(self) -> tuple[int, int] | None:
        """Get the organized point cloud shape (height, width).

        Returns:
            Tuple of (height, width) if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        if data is None:
            return None

        height = data.get("height")
        width = data.get("width")

        if height is None or width is None:
            return None

        return (height, width)

    def is_dense(self) -> bool:
        """Check if the point cloud is dense (no invalid points).

        Returns:
            True if dense, False if contains invalid points or no data.
        """
        data = self._subscriber.get_latest()
        if data is None:
            return False
        return data.get("is_dense", False)

    @property
    def name(self) -> str:
        """Get the 3D LIDAR name.

        Returns:
            LIDAR name string.
        """
        return self._name
