#!/usr/bin/env python3
"""Wrist camera stream sender for the two ZED One cameras on the ZED box.

Run this on the ZED box (the wrist camera host) to stream both wrist cameras
over ZMQ (one port each) to camera/wrist_camera_receiver.py. Requires the
ZED SDK (pyzed) with ZED One support.

Usage:
    python stream_sender_zed_box.py
"""

import signal
import struct
import threading
import time

import cv2
import pyzed.sl as sl
import zmq

# ----------------------------
# Settings
# ----------------------------
RIGHT_SERIAL = 307245838
LEFT_SERIAL = 300534739

LEFT_PORT = 5555
RIGHT_PORT = 5556

# Capture Resolution (Input)
# We use HD1080 because it is 16:9, matching our target 640x360
INPUT_RESOLUTION = sl.RESOLUTION.HD1080
FPS = 30

# Stream Resolution (Output - sent over network)
STREAM_W = 640
STREAM_H = 360

stop_event = threading.Event()


def camera_publisher_thread(serial, port, stop_event):
    # ZMQ Setup
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.CONFLATE, 1)  # Only keep last frame
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.bind(f"tcp://*:{port}")

    # Camera Setup
    cam = sl.CameraOne()
    init = sl.InitParametersOne()
    init.camera_resolution = INPUT_RESOLUTION
    init.camera_fps = FPS
    init.set_from_serial_number(serial)
    init.sdk_verbose = 0

    print(f"[Sender] Opening SN{serial}...")
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Failed to open SN{serial}: {err}")
        return

    print(f"[Sender] SN{serial} streaming on port {port} ({STREAM_W}x{STREAM_H})")

    mat = sl.Mat()

    while not stop_event.is_set():
        if cam.grab() == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat)

            # 1. Get Raw Data (BGRA)
            bgra = mat.get_data()

            # 2. Resize immediately (Fastest way to reduce load)
            # Resize from 1920x1080 -> 640x360
            resized_bgra = cv2.resize(bgra, (STREAM_W, STREAM_H), interpolation=cv2.INTER_LINEAR)

            # 3. Convert BGRA -> BGR (Now much cheaper since image is small)
            bgr = cv2.cvtColor(resized_bgra, cv2.COLOR_BGRA2BGR)

            # 4. Serialize and Send
            h, w, c = bgr.shape
            header = struct.pack("III", h, w, c)

            try:
                # Send header + image data in one non-blocking shot
                sock.send(header + bgr.tobytes(), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass  # Drop frame if network is busy
        else:
            time.sleep(0.001)

    cam.close()
    sock.close()
    ctx.term()
    print(f"[Sender] Closed SN{serial}")


if __name__ == "__main__":
    # Handle Ctrl+C cleanly
    def signal_handler(sig, frame):
        print("\n[Sender] Stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    t_left = threading.Thread(
        target=camera_publisher_thread, args=(LEFT_SERIAL, LEFT_PORT, stop_event), daemon=True
    )
    t_right = threading.Thread(
        target=camera_publisher_thread, args=(RIGHT_SERIAL, RIGHT_PORT, stop_event), daemon=True
    )

    t_left.start()
    t_right.start()

    print("[Sender] Streaming started...")

    # Keep main thread alive
    while not stop_event.is_set():
        time.sleep(1)

    t_left.join()
    t_right.join()
