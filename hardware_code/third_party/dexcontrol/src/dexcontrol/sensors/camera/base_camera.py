# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Base camera sensor with unified Zenoh and RTC transport support.

This module provides the foundation for camera sensors in dexcontrol, supporting
both Zenoh (JPEG/PNG compressed, reliable) and RTC (real-time video, hardware
accelerated) transports for receiving data from dexsensor.

Architecture:
    - StreamSubscriber: Wrapper providing unified API for Zenoh/RTC transports
    - BaseCameraSensor: Abstract base class for all camera sensors

Key Features:
    - Zenoh transport uses dexcomm Node API with automatic codec decoding
    - RTC transport uses dexcomm VideoSubscriber with hardware acceleration
    - Zero-copy data access where possible
    - Thread-safe buffering and state management
    - Production-grade error handling and logging
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from queue import Empty, Queue
from typing import Any, Protocol, runtime_checkable

import numpy as np
from dexcomm import Node
from dexcomm.codecs import DepthImageCodec, RGBImageCodec
from loguru import logger

from dexcontrol.exceptions import ConfigurationError, ServiceUnavailableError

# RTC imports (optional for systems without hardware video support)
try:
    from dexcomm.rtc import RtcConfig, VideoCodec, VideoSubscriber

    RTC_AVAILABLE = True
except ImportError:
    RTC_AVAILABLE = False
    VideoSubscriber = None
    VideoCodec = None
    RtcConfig = None


class TransportType(str, Enum):
    """Supported transport types for camera streams."""

    ZENOH = "zenoh"
    RTC = "rtc"


class StreamType(str, Enum):
    """Supported stream data types."""

    RGB = "rgb"
    DEPTH = "depth"


@runtime_checkable
class SubscriberProtocol(Protocol):
    """Protocol for subscriber-like objects with unified API."""

    def get_latest(self) -> np.ndarray | dict | None:
        """Get the latest data from the subscriber."""
        ...

    def wait_for_message(self, timeout: float) -> Any:
        """Wait for a message with timeout."""
        ...

    def shutdown(self) -> None:
        """Shutdown the subscriber."""
        ...


class StreamSubscriber:
    """Unified wrapper for camera stream subscribers.

    Provides a consistent interface for both Zenoh (reliable, compressed) and
    RTC (real-time, hardware accelerated) transports. Handles codec selection,
    data buffering, and transport-specific initialization automatically.

    Attributes:
        stream_name: Unique identifier for this stream (e.g., 'rgb', 'left_rgb').
        transport: Transport type (zenoh or rtc).
        stream_type: Data type (rgb or depth).

    Thread Safety:
        All public methods are thread-safe. Internal state is protected by locks
        where necessary.
    """

    def __init__(
        self,
        stream_name: str,
        transport: TransportType,
        stream_type: StreamType,
        node: Node | None = None,
        topic: str | None = None,
        rtc_channel: str | None = None,
        width: int | None = None,
        height: int | None = None,
        codec: str = "h264",
        buffer_size: int = 1,
    ) -> None:
        """Initialize a stream subscriber.

        Args:
            stream_name: Name of the stream (e.g., 'rgb', 'left_rgb', 'depth').
            transport: Transport type to use.
            stream_type: Type of data this stream carries.
            node: DexComm Node instance (required for Zenoh).
            topic: Zenoh topic (required for Zenoh transport).
            rtc_channel: RTC signaling channel (required for RTC transport).
            width: Frame width (optional - VideoSubscriber auto-queries if None).
            height: Frame height (optional - VideoSubscriber auto-queries if None).
            codec: Video codec for RTC ('vp8', 'h264', 'h265'). Default: 'h264'.
            buffer_size: Number of frames to buffer. Default: 1.

        Raises:
            ValueError: If required parameters are missing or invalid.
            ConfigurationError: If RTC is requested but not available.
        """
        self.stream_name = stream_name
        self.transport = transport
        self.stream_type = stream_type

        self._subscriber: SubscriberProtocol | None = None
        self._frame_queue: Queue[np.ndarray] | None = None
        self._latest_frame: np.ndarray | None = None
        self._active = False

        # Validate and create subscriber based on transport type
        if transport == TransportType.ZENOH:
            self._create_zenoh_subscriber(node, topic, buffer_size)
        elif transport == TransportType.RTC:
            self._create_rtc_subscriber(
                rtc_channel, width, height, codec, buffer_size
            )
        else:
            raise ValueError(f"Unknown transport type: {transport}")

    def _create_zenoh_subscriber(
        self,
        node: Node | None,
        topic: str | None,
        buffer_size: int,
    ) -> None:
        """Create a Zenoh subscriber with appropriate codec.

        Args:
            node: DexComm Node instance.
            topic: Zenoh topic to subscribe to.
            buffer_size: Number of messages to buffer.

        Raises:
            ValueError: If node or topic is None.
        """
        if node is None or topic is None:
            raise ValueError(
                f"Zenoh transport requires 'node' and 'topic' "
                f"for stream '{self.stream_name}'"
            )

        # Select codec based on stream type
        decoder = (
            DepthImageCodec.decode
            if self.stream_type == StreamType.DEPTH
            else RGBImageCodec.decode
        )

        # Create subscriber using Node API
        self._subscriber = node.create_subscriber(
            topic=topic,
            decoder=decoder,
            buffer_size=buffer_size,
        )

        logger.info(
            f"Created Zenoh subscriber for '{self.stream_name}' "
            f"[type={self.stream_type.value}, topic={topic}]"
        )

    def _create_rtc_subscriber(
        self,
        rtc_channel: str | None,
        width: int | None,
        height: int | None,
        codec: str,
        buffer_size: int,
    ) -> None:
        """Create an RTC video subscriber with hardware acceleration.

        VideoSubscriber automatically queries metadata from the publisher if
        width/height are not provided, eliminating the need for manual specification.

        Args:
            rtc_channel: RTC signaling channel name.
            width: Frame width in pixels (auto-queried from publisher if None).
            height: Frame height in pixels (auto-queried from publisher if None).
            codec: Video codec identifier.
            buffer_size: Number of frames to buffer.

        Raises:
            ValueError: If required parameters are missing.
            ConfigurationError: If RTC support is not available.
        """
        if not RTC_AVAILABLE:
            raise ConfigurationError(
                f"RTC transport requested for '{self.stream_name}' but "
                "dexcomm.rtc is not available. "
                "Rebuild dexcomm with 'rtc' feature enabled."
            )

        if self.stream_type == StreamType.DEPTH:
            raise ValueError(
                f"RTC transport is not supported for depth streams "
                f"(stream: '{self.stream_name}')"
            )

        if rtc_channel is None:
            raise ValueError(
                f"RTC transport requires 'rtc_channel' for stream '{self.stream_name}'"
            )

        # Log auto-detection if width/height not provided
        if width is None or height is None:
            logger.info(
                f"Width/height not specified for '{self.stream_name}', "
                f"VideoSubscriber will auto-query from publisher"
            )

        # Map codec string to VideoCodec enum
        codec_map = {
            "vp8": VideoCodec.VP8,
            "h264": VideoCodec.H264,
            "h265": VideoCodec.H265,
        }
        video_codec = codec_map.get(codec.lower())
        if video_codec is None:
            raise ValueError(
                f"Unknown codec '{codec}'. Supported: {list(codec_map.keys())}"
            )

        # Create frame queue for RTC callback
        self._frame_queue = Queue(maxsize=buffer_size)

        def on_frame(frame: np.ndarray) -> None:
            """Callback for incoming RTC frames (runs in RTC thread)."""
            try:
                # Non-blocking put - discard oldest frame if queue is full
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except Empty:
                        pass
                self._frame_queue.put_nowait(frame)
                self._latest_frame = frame
                self._active = True
            except Exception as e:
                logger.debug(
                    f"Error queuing frame for '{self.stream_name}': {e}"
                )

        # Create RTC config for local network (low latency)
        config = RtcConfig.local()

        # Create VideoSubscriber - all decoding done in Rust with hardware acceleration
        # Note: width/height are optional - VideoSubscriber auto-queries from publisher
        self._subscriber = VideoSubscriber(
            rtc_channel,
            video_codec,
            on_frame,
            config,
            width,  # Optional - auto-queried if None
            height,  # Optional - auto-queried if None
            bgr=False,  # Return RGB format
            metadata_timeout_secs=5,  # Timeout for auto-query
        )

        # Log success (dimensions may have been auto-detected by VideoSubscriber)
        resolution_str = (
            f"{width}x{height}" if width and height else "auto-detected"
        )
        logger.info(
            f"Created RTC subscriber for '{self.stream_name}' "
            f"[codec={codec.upper()}, resolution={resolution_str}, "
            f"channel={rtc_channel}]"
        )

    def get_latest(self) -> np.ndarray | dict | None:
        """Get the most recent frame from the stream.

        Returns:
            For Zenoh: dict with 'data' (np.ndarray) and 'timestamp' (int) keys.
            For RTC: np.ndarray directly (RGB format, uint8).
            None if no data is available.

        Thread Safety:
            This method is thread-safe.
        """
        if self._subscriber is None:
            return None

        try:
            if self.transport == TransportType.RTC:
                # For RTC, return latest from queue
                return self._latest_frame.copy() if self._latest_frame is not None else None
            else:
                # For Zenoh, use subscriber's get_latest
                return self._subscriber.get_latest()
        except Exception as e:
            logger.debug(f"Error getting latest from '{self.stream_name}': {e}")
            return None

    def is_active(self, window_seconds: float = 1.0) -> bool:
        """Check if the stream is actively receiving data.

        Args:
            window_seconds: Time window to check for activity (unused for current impl).

        Returns:
            True if data was received recently, False otherwise.

        Thread Safety:
            This method is thread-safe.
        """
        if self._subscriber is None:
            return False

        try:
            if self.transport == TransportType.RTC:
                return self._active
            else:
                # For Zenoh, check if we have any data
                return self._subscriber.get_latest() is not None
        except Exception as e:
            logger.debug(f"Error checking activity for '{self.stream_name}': {e}")
            return False

    def wait_for_message(self, timeout: float = 5.0) -> Any:
        """Wait for a message to arrive.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Message data if received within timeout, None otherwise.

        Thread Safety:
            This method is thread-safe.
        """
        if self._subscriber is None:
            return None

        try:
            if self.transport == TransportType.RTC:
                # For RTC, wait for frame in queue
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self._latest_frame is not None:
                        return self._latest_frame.copy()
                    time.sleep(0.01)
                return None
            else:
                # For Zenoh, use subscriber's wait method
                return self._subscriber.wait_for_message(timeout)
        except Exception as e:
            logger.debug(
                f"Error waiting for message on '{self.stream_name}': {e}"
            )
            return None

    def shutdown(self) -> None:
        """Shutdown the subscriber and release all resources.

        Thread Safety:
            This method is thread-safe and can be called multiple times.
        """
        if self._subscriber is not None:
            try:
                self._subscriber.shutdown()
                logger.debug(f"Shutdown subscriber for '{self.stream_name}'")
            except Exception as e:
                logger.error(
                    f"Error shutting down '{self.stream_name}': {e}"
                )
            finally:
                self._subscriber = None

        # Clear RTC-specific state
        if self._frame_queue is not None:
            try:
                while not self._frame_queue.empty():
                    self._frame_queue.get_nowait()
            except Exception:
                pass
            self._frame_queue = None

        self._latest_frame = None
        self._active = False


class BaseCameraSensor(ABC):
    """Abstract base class for camera sensors.

    Provides common functionality for managing camera streams with both Zenoh
    and RTC transports. Handles Node creation, stream lifecycle, and provides
    utilities for checking stream status.

    Subclasses must implement:
        - get_obs(): Method to retrieve observations in sensor-specific format

    Attributes:
        name: Unique identifier for this camera sensor.
        available_streams: List of stream names that are configured.
        active_streams: List of stream names currently receiving data.

    Thread Safety:
        All public methods are thread-safe unless otherwise noted.
    """

    def __init__(self, name: str) -> None:
        """Initialize the base camera sensor.

        Args:
            name: Unique identifier for this camera sensor.

        Raises:
            ServiceUnavailableError: If Node creation fails.
        """
        self._name = name
        self._node: Node | None = None
        self._streams: dict[str, StreamSubscriber] = {}

        # Create DexComm Node for Zenoh subscribers
        try:
            self._node = Node(name=f"{name}_camera_node")
            logger.info(f"Created DexComm Node for camera '{name}'")
        except Exception as e:
            logger.error(f"Failed to create DexComm Node for '{name}': {e}")
            raise ServiceUnavailableError(f"Node creation failed: {e}") from e

    def _create_stream(
        self,
        stream_name: str,
        config: dict[str, Any],
        stream_type: StreamType = StreamType.RGB,
    ) -> StreamSubscriber | None:
        """Create a stream subscriber from configuration.

        Args:
            stream_name: Unique identifier for this stream.
            config: Stream configuration dictionary with keys:
                - enable (bool): Whether stream is enabled.
                - transport (str): 'zenoh' or 'rtc'.
                - topic (str): Zenoh topic (for zenoh transport).
                - rtc_channel (str): RTC signaling channel (for rtc transport).
                - width (int): Frame width (for rtc transport).
                - height (int): Frame height (for rtc transport).
                - codec (str): Video codec (for rtc transport, default: 'h264').
            stream_type: Type of data this stream carries.

        Returns:
            StreamSubscriber instance if successful, None if disabled or failed.
        """
        if not config.get("enable", False):
            logger.debug(f"Stream '{stream_name}' is disabled")
            return None

        # Parse transport type
        transport_str = config.get("transport", "zenoh")
        try:
            transport = TransportType(transport_str)
        except ValueError:
            logger.error(
                f"Invalid transport '{transport_str}' for stream '{stream_name}'"
            )
            return None

        # Enforce depth streams must use Zenoh
        if stream_type == StreamType.DEPTH and transport != TransportType.ZENOH:
            logger.warning(
                f"Depth stream '{stream_name}' must use Zenoh transport, "
                f"overriding config"
            )
            transport = TransportType.ZENOH

        try:
            return StreamSubscriber(
                stream_name=stream_name,
                transport=transport,
                stream_type=stream_type,
                node=self._node,
                topic=config.get("topic"),
                rtc_channel=config.get("rtc_channel"),
                width=config.get("width"),
                height=config.get("height"),
                codec=config.get("codec", "h264"),
                buffer_size=config.get("buffer_size", 1),
            )
        except Exception as e:
            logger.error(f"Failed to create stream '{stream_name}': {e}")
            return None

    @abstractmethod
    def get_obs(self, **kwargs) -> Any:
        """Get observations from the camera.

        Subclasses must implement this method to return observations in their
        specific format (e.g., single array for USB camera, dict for ZED camera).

        Args:
            **kwargs: Sensor-specific parameters.

        Returns:
            Observation data in sensor-specific format.
        """
        pass

    def shutdown(self) -> None:
        """Shutdown the camera sensor and release all resources.

        This method is idempotent and can be called multiple times safely.

        Thread Safety:
            This method is thread-safe.
        """
        logger.info(f"Shutting down camera '{self._name}'")

        # Shutdown all streams
        for stream_name, stream in self._streams.items():
            if stream is not None:
                try:
                    stream.shutdown()
                except Exception as e:
                    logger.error(
                        f"Error shutting down stream '{stream_name}': {e}"
                    )

        # Shutdown node
        if self._node is not None:
            try:
                self._node.shutdown()
                logger.debug(f"Shutdown DexComm Node for '{self._name}'")
            except Exception as e:
                logger.error(f"Error shutting down DexComm Node: {e}")
            finally:
                self._node = None

    def is_active(self) -> bool:
        """Check if any camera stream is actively receiving data.

        Returns:
            True if at least one stream is active, False otherwise.

        Thread Safety:
            This method is thread-safe.
        """
        return any(
            stream.is_active()
            for stream in self._streams.values()
            if stream is not None
        )

    def is_stream_active(self, stream_name: str) -> bool:
        """Check if a specific stream is actively receiving data.

        Args:
            stream_name: Name of the stream to check.

        Returns:
            True if the stream is active, False otherwise.

        Thread Safety:
            This method is thread-safe.
        """
        stream = self._streams.get(stream_name)
        return stream.is_active() if stream is not None else False

    def wait_for_active(
        self, timeout: float = 5.0, require_all: bool = False
    ) -> bool:
        """Wait for camera streams to become active.

        Args:
            timeout: Maximum time to wait in seconds.
            require_all: If True, waits for all enabled streams to become active.
                        If False, waits for at least one stream.

        Returns:
            True if the condition is met within timeout, False otherwise.

        Thread Safety:
            This method is thread-safe.
        """
        active_streams = [s for s in self._streams.values() if s is not None]
        if not active_streams:
            logger.warning(f"'{self._name}': No streams enabled")
            return True

        if require_all:
            # Wait for each stream sequentially
            for stream in active_streams:
                if stream.wait_for_message(timeout) is None:
                    logger.warning(
                        f"'{self._name}': Timeout waiting for stream "
                        f"'{stream.stream_name}'"
                    )
                    return False
            logger.info(f"'{self._name}': All streams are active")
            return True
        else:
            # Wait for at least one stream
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.is_active():
                    logger.info(f"'{self._name}': At least one stream is active")
                    return True
                time.sleep(0.1)
            logger.warning(f"'{self._name}': Timeout waiting for any stream")
            return False

    @property
    def name(self) -> str:
        """Get the camera sensor name."""
        return self._name

    @property
    def available_streams(self) -> list[str]:
        """Get list of available (configured) stream names."""
        return [
            name for name, stream in self._streams.items() if stream is not None
        ]

    @property
    def active_streams(self) -> list[str]:
        """Get list of currently active stream names."""
        return [
            name
            for name, stream in self._streams.items()
            if stream is not None and stream.is_active()
        ]

    def __enter__(self) -> "BaseCameraSensor":
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - ensures cleanup."""
        self.shutdown()

    def __del__(self) -> None:
        """Destructor - ensures cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass  # Suppress exceptions during cleanup
