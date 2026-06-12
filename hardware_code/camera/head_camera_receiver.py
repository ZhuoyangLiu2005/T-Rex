#!/usr/bin/env python3
import struct
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import zmq


@dataclass
class StreamFrame:
    ts_sec: float
    image_rgb: np.ndarray


class HeadCameraReceiver:
    """Receiver for head camera stream (ZED X Mini via ZMQ).

    - Uses zmq.CONFLATE to ensure only the latest frame is kept.
    - Uses Single-Part messages (Header+Data) to prevent framing errors.
    """

    def __init__(
        self,
        im_h: int,
        im_w: int,
        sender_ip: str,
        ports: Dict[str, int],
        verbose: bool = True,
    ):
        self.im_h = int(im_h)
        self.im_w = int(im_w)
        self.sender_ip = sender_ip
        self.ports = ports
        self.verbose = verbose

        self._lock = threading.Lock()
        self._buffer: Dict[str, Optional[StreamFrame]] = {k: None for k in ports}
        self._ready_events = {k: threading.Event() for k in ports}

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._ctx = zmq.Context()
        self._socks = {}
        for name, port in ports.items():
            s = self._ctx.socket(zmq.SUB)

            # Keep only the last message
            s.setsockopt(zmq.CONFLATE, 1)
            # Drop packets immediately if blocked
            s.setsockopt(zmq.RCVHWM, 1)

            s.connect(f"tcp://{self.sender_ip}:{port}")
            s.setsockopt(zmq.SUBSCRIBE, b"")
            self._socks[name] = s

    def start_receiving(self, timeout: float | None = None):
        if self._thread is not None:
            raise RuntimeError("Receiver already started")

        if self.verbose:
            for name, port in self.ports.items():
                print(f"[ZMQ RX HEAD] {name} connecting to tcp://{self.sender_ip}:{port}")

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        start = time.time()
        for name, ev in self._ready_events.items():
            if self.verbose:
                print(f"[ZMQ RX HEAD] Waiting for first frame from {name}...")
            rem = None if timeout is None else max(0, timeout - (time.time() - start))
            if not ev.wait(timeout=rem):
                raise TimeoutError(f"Timeout waiting for first frame from {name}")

        if self.verbose:
            print("[ZMQ RX HEAD] All streams are live")

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        for s in self._socks.values():
            s.close()
        self._ctx.term()

    def get_data(self, name: str) -> StreamFrame:
        with self._lock:
            frame = self._buffer.get(name)
            if frame is None:
                raise RuntimeError(f"No frame received yet for {name}")
            ts = frame.ts_sec
            img = frame.image_rgb

        resized = cv2.resize(img, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return StreamFrame(ts_sec=ts, image_rgb=resized_rgb)

    def _recv_loop(self):
        poller = zmq.Poller()
        for s in self._socks.values():
            poller.register(s, zmq.POLLIN)

        if self.verbose:
            print("[ZMQ RX HEAD] Receive loop started")

        while not self._stop_evt.is_set():
            events = dict(poller.poll(timeout=100))

            for name, sock in self._socks.items():
                if sock in events:
                    try:
                        # Receive single packet (Header + Payload combined)
                        message = sock.recv(flags=zmq.NOBLOCK)

                        # Header is 3 integers (4 bytes each) = 12 bytes
                        header = message[:12]
                        payload = message[12:]

                        h, w, c = struct.unpack("III", header)

                        # Verify payload size matches expectations to prevent segfaults
                        if len(payload) != h * w * c:
                            continue

                        frame = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, c)

                        sf = StreamFrame(
                            ts_sec=time.time(),
                            image_rgb=frame.copy(),
                        )

                        with self._lock:
                            self._buffer[name] = sf
                            self._ready_events[name].set()

                    except zmq.Again:
                        pass
                    except Exception as e:
                        if self.verbose:
                            print(f"[ZMQ RX HEAD] Error decoding frame: {e}")


if __name__ == "__main__":
    rx = HeadCameraReceiver(
        im_h=360,
        im_w=640,
        sender_ip="192.168.50.22",
        ports={
            "HEAD_CAM": 5555,
        },
        verbose=True,
    )

    try:
        rx.start_receiving(timeout=10.0)
        print("Receiver fully initialized, head camera stream is live")
        prev_t = None
        while True:
            head = rx.get_data("HEAD_CAM")
            ts = head.ts_sec
            print(f"Received HEAD_CAM frame at {ts:.3f} sec")

            head_image_bgr = cv2.cvtColor(head.image_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("HEAD", head_image_bgr)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        rx.stop()
        cv2.destroyAllWindows()
