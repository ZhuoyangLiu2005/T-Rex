# Camera streaming

Senders and receivers for the head and wrist camera streams consumed by
`teleop/main_teleop.py`.

| File | Runs on | Role |
| --- | --- | --- |
| `stream_sender_dexmate.py` | Robot Jetson | Streams the head camera (ZED X Mini, left view) on port 5555 |
| `stream_sender_zed_box.py` | ZED box | Streams both wrist cameras (2× ZED One, one thread each) on ports 5555/5556 |
| `head_camera_receiver.py` | Teleop workstation | Receives the head camera stream |
| `wrist_camera_receiver.py` | Teleop workstation | Receives the wrist camera streams |
| `view_head_camera.py` | Teleop workstation | View the head camera directly via `dexcontrol` (no streamer needed) |

The two senders share the same pipeline (grab at HD1080/30 fps → resize to
640×360 → BGR → publish) and differ only in the ZED SDK camera class: the
head camera is a stereo ZED X Mini (`sl.Camera`, retrieving `VIEW.LEFT`),
the wrist cameras are monocular ZED Ones (`sl.CameraOne`).

## Deployment

The senders run on the robot hardware, not on the teleop workstation. Copy
the sender script to the corresponding machine (Jetson / ZED box), which must
have the [ZED SDK](https://www.stereolabs.com/developers/) with its Python
API (`pyzed`) installed, and adjust the serial numbers / ports at the top of
the script for your cameras. `teleop_launcher.sh` opens ssh tabs that start
them remotely. The receiver-side endpoints live in `config/default.yaml`
(`cameras.head` / `cameras.wrist`).

## Wire protocol

Senders publish on a ZMQ **PUB** socket (one port per camera) as
**single-part** messages:

```
[12-byte header: struct.pack("III", height, width, channels)] + [raw uint8 BGR bytes]
```

Both sides set `zmq.CONFLATE` so only the latest frame is kept; slow
consumers never see stale backlogs.
