# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""WebRTC subscriber implementation with DexComm-compatible API.

Clean, modular implementation of WebRTC video streaming that provides
the exact same interface as DexComm subscribers.
"""

import asyncio
import json
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger

from dexcontrol.exceptions import ConfigurationError
from dexcontrol.utils.comm_helper import query_json_service

# WebRTC dependencies
try:
    import websockets

    WEBRTC_AVAILABLE = True
except ImportError:
    websockets = None
    WEBRTC_AVAILABLE = False

# Performance optimization
try:
    import uvloop

    UVLOOP_AVAILABLE = True
except ImportError:
    uvloop = None
    UVLOOP_AVAILABLE = False


class RTCSubscriber:
    """WebRTC video subscriber with DexComm-compatible API.

    Receives video streams via WebRTC and provides the same interface
    as DexComm's Subscriber class for seamless integration.
    """

    def __init__(
        self,
        signaling_url: str | None = None,
        info_topic: str | None = None,
        callback: Callable[[np.ndarray], None] | None = None,
        buffer_size: int = 1,
        name: str | None = None,
    ):
        """Initialize WebRTC subscriber.

        Args:
            signaling_url: Direct WebRTC signaling URL (preferred)
            info_topic: Topic to query for WebRTC connection info (fallback)
            callback: Optional callback for incoming frames
            buffer_size: Number of frames to buffer
            name: Optional name for logging
        """
        self.signaling_url = signaling_url
        self.topic = info_topic
        self.callback = callback
        self.buffer_size = buffer_size
        self.name = name or f"rtc_{(info_topic or 'direct').replace('/', '_')}"

        # Data storage
        self._buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._data_lock = threading.Lock()

        # Statistics
        self._frame_count = 0
        self._last_receive_time_ns: int | None = None

        # Connection state
        self._active = False
        self._connected = False
        self._rtc_info: dict | None = None

        # Threading
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # WebRTC objects
        self._pc: Any | None = None
        self._websocket: Any | None = None
        self._async_stop: Any | None = None

        # Initialize connection
        self._initialize()

    def _initialize(self) -> None:
        """Initialize WebRTC connection."""
        if not WEBRTC_AVAILABLE:
            raise ConfigurationError(
                f"WebRTC dependencies not available for '{self.name}'. "
                "Install: pip install aiortc websockets"
            )

        # Use direct signaling URL if provided, otherwise query for it
        if self.signaling_url:
            self._rtc_info = {"signaling_url": self.signaling_url}
            logger.debug(
                f"{self.name}: Using direct signaling URL: {self.signaling_url}"
            )
        else:
            # Query for connection info
            self._rtc_info = self._query_connection_info()
            if not self._rtc_info:
                logger.warning(f"{self.name}: Failed to get connection info")
                return

        # Start WebRTC connection
        self._start_connection()

    def _query_connection_info(self) -> dict[str, Any] | None:
        """Query for WebRTC connection information."""
        logger.debug(f"{self.name}: Querying {self.topic}")

        info = query_json_service(topic=self.topic, timeout=2.0)

        if info and "signaling_url" in info:
            logger.info(f"{self.name}: Got connection info")
            return info

        return None

    def _start_connection(self) -> None:
        """Start WebRTC connection in background thread."""
        url = self._rtc_info.get("signaling_url")
        if not url:
            logger.error(f"{self.name}: No signaling URL")
            return

        self._url = url
        self._thread = threading.Thread(
            target=self._run_async_loop, daemon=True, name=f"{self.name}_thread"
        )
        self._thread.start()

    def _run_async_loop(self) -> None:
        """Run async event loop in thread."""
        try:
            if UVLOOP_AVAILABLE and uvloop:
                loop = uvloop.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._connect_and_receive())
                loop.close()
            else:
                asyncio.run(self._connect_and_receive())
        except Exception as e:
            logger.error(f"{self.name}: Event loop error: {e}")
        finally:
            self._cleanup_state()

    async def _connect_and_receive(self) -> None:
        """Main async function for WebRTC connection."""
        self._async_stop = asyncio.Event()
        monitor = asyncio.create_task(self._monitor_stop())

        try:
            # Create peer connection with certificate workaround
            # Fix for pyOpenSSL compatibility issue
            import ssl

            from aiortc import RTCConfiguration, RTCPeerConnection

            ssl._create_default_https_context = ssl._create_unverified_context

            # Create configuration without ICE servers to avoid cert issues
            config = RTCConfiguration(iceServers=[])
            self._pc = RTCPeerConnection(configuration=config)

            # Setup track handler
            @self._pc.on("track")
            async def on_track(track):
                logger.info(f"{self.name}: Received {track.kind} track")
                if track.kind == "video":
                    # Start receiving frames as a background task
                    asyncio.create_task(self._receive_frames(track))

            # Connect to signaling server
            await self._establish_connection()

            # Wait until stopped
            await self._async_stop.wait()

        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"{self.name}: Connection error: {e}")
        finally:
            monitor.cancel()
            await self._cleanup_async()

    async def _establish_connection(self) -> None:
        """Establish WebRTC connection via signaling."""
        try:
            self._websocket = await websockets.connect(self._url)
            logger.debug(f"{self.name}: WebSocket connected to {self._url}")

            # Create offer
            self._pc.addTransceiver("video", direction="recvonly")
            offer = await self._pc.createOffer()
            await self._pc.setLocalDescription(offer)

            # Send offer
            await self._websocket.send(
                json.dumps(
                    {
                        "sdp": self._pc.localDescription.sdp,
                        "type": self._pc.localDescription.type,
                    }
                )
            )

            # Receive answer
            response = json.loads(await self._websocket.recv())
            if response["type"] == "answer":
                from aiortc import RTCSessionDescription

                await self._pc.setRemoteDescription(
                    RTCSessionDescription(response["sdp"], response["type"])
                )
                self._connected = True
            else:
                raise ValueError(f"Unexpected response: {response['type']}")
        except Exception as e:
            logger.error(f"{self.name}: Failed to establish connection: {e}")
            raise

    async def _receive_frames(self, track) -> None:
        """Receive and process video frames."""
        while not self._async_stop.is_set():
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                img = frame.to_ndarray(format="rgb24")
                self._process_frame(img)

                if not self._active:
                    self._active = True

            except asyncio.TimeoutError:
                # This is normal during waiting, only log if too frequent
                continue
            except Exception as e:
                if not self._async_stop.is_set():
                    logger.error(f"{self.name}: Frame receive error: {e}")
                break
        logger.debug(f"{self.name}: Frame receiver stopped")

    def _process_frame(self, frame: np.ndarray) -> None:
        """Process incoming video frame."""
        # Update statistics
        with self._data_lock:
            self._frame_count += 1
            self._last_receive_time_ns = time.time_ns()
            self._latest_frame = frame

        # Update buffer
        with self._buffer_lock:
            self._buffer.append(frame)
            if len(self._buffer) > self.buffer_size:
                self._buffer.pop(0)

        # Call callback
        if self.callback:
            try:
                self.callback(frame)
            except Exception as e:
                logger.error(f"{self.name}: Callback error: {e}")

    async def _monitor_stop(self) -> None:
        """Monitor stop event from main thread."""
        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)
        self._async_stop.set()

    async def _cleanup_async(self) -> None:
        """Clean up async resources."""
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass

        if self._pc and self._pc.connectionState != "closed":
            try:
                await self._pc.close()
            except Exception:
                pass

    def _cleanup_state(self) -> None:
        """Clean up connection state."""
        with self._data_lock:
            self._active = False
            self._connected = False

    # DexComm-compatible API

    def get_latest(self) -> np.ndarray | None:
        """Get the latest received frame."""
        with self._data_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_buffer(self) -> list[np.ndarray]:
        """Get all buffered frames."""
        with self._buffer_lock:
            return [f.copy() for f in self._buffer]

    def wait_for_message(self, timeout: float = 5.0) -> np.ndarray | None:
        """Wait for a frame to be received."""
        start = time.time()
        initial_count = self._frame_count

        while time.time() - start < timeout:
            if self._frame_count > initial_count:
                return self.get_latest()
            time.sleep(0.05)

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get subscriber statistics."""
        with self._data_lock:
            return {
                "topic": self.topic,
                "receive_count": self._frame_count,
                "last_receive_time_ns": self._last_receive_time_ns,
                "buffer_size": len(self._buffer),
                "active": self._active,
                "connected": self._connected,
            }

    def is_active(self) -> bool:
        """Check if actively receiving frames."""
        return self._active

    def shutdown(self) -> None:
        """Shutdown the subscriber."""
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(f"{self.name}: Thread didn't stop cleanly")

        self._cleanup_state()
        logger.debug(f"{self.name}: Shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()

    def __del__(self):
        self.shutdown()


def create_rtc_camera_subscriber(
    signaling_url: str | None = None,
    info_topic: str | None = None,
    callback: Callable[[np.ndarray], None] | None = None,
    buffer_size: int = 1,
) -> RTCSubscriber:
    """Create an RTC subscriber for camera streams.

    Args:
        signaling_url: Direct WebRTC signaling URL (preferred)
        info_topic: Topic to query for connection info (fallback)
        callback: Optional callback for frames
        buffer_size: Number of frames to buffer

    Returns:
        RTCSubscriber instance with DexComm-compatible API
    """
    if not signaling_url and not info_topic:
        raise ValueError("Either signaling_url or info_topic must be provided")

    return RTCSubscriber(
        signaling_url=signaling_url,
        info_topic=info_topic,
        callback=callback,
        buffer_size=buffer_size,
    )
