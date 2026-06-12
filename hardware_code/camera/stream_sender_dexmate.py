#!/usr/bin/env python3
"""Head camera stream sender for the ZED X Mini on the robot Jetson.

Run this on the robot Jetson (the head camera host) to stream the head camera
over ZMQ to camera/head_camera_receiver.py. Requires the ZED SDK (pyzed).

Usage:
    python stream_sender_dexmate.py
"""

import signal
import struct
import time

import cv2
import pyzed.sl as sl
import zmq

# ----------------------------
# Settings
# ----------------------------
HEAD_SERIAL = 55793948  # ZED X Mini serial number
HEAD_PORT = 5555

# Capture Resolution (Input)
# We use HD1080 because it is 16:9, matching our target 640x360
INPUT_RESOLUTION = sl.RESOLUTION.HD1080
FPS = 30

# Stream Resolution (Output - sent over network)
STREAM_W = 640
STREAM_H = 360

stop_event = False


def signal_handler(sig, frame):
    global stop_event
    print("\n[Sender] Stopping...")
    stop_event = True


def main():
    global stop_event
    signal.signal(signal.SIGINT, signal_handler)

    # ZMQ Setup
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.CONFLATE, 1)  # Only keep last frame
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.bind(f"tcp://*:{HEAD_PORT}")

    # Camera Setup (ZED X Mini is a stereo camera -> use sl.Camera)
    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = INPUT_RESOLUTION
    init.camera_fps = FPS
    init.set_from_serial_number(HEAD_SERIAL)
    init.sdk_verbose = 0

    print(f"[Sender] Opening ZED X Mini SN{HEAD_SERIAL}...")
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Failed to open SN{HEAD_SERIAL}: {err}")
        sock.close()
        ctx.term()
        return

    print(f"[Sender] SN{HEAD_SERIAL} streaming on port {HEAD_PORT} ({STREAM_W}x{STREAM_H})")

    mat = sl.Mat()

    while not stop_event:
        if cam.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left view only
            cam.retrieve_image(mat, sl.VIEW.LEFT)

            # 1. Get Raw Data (BGRA)
            bgra = mat.get_data()

            # 2. Resize immediately (Fastest way to reduce load)
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
    print(f"[Sender] Closed SN{HEAD_SERIAL}")


if __name__ == "__main__":
    main()
