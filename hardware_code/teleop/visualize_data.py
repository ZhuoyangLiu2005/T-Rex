"""Visualize recorded episode data including video feeds and tactile information.

Usage:
    python teleop/visualize_data.py episode_test_1234567890

This will look for:
    - episode_test_1234567890.h5 (tactile + joint data)
    - episode_test_1234567890_head_left_rgb.mp4
    - episode_test_1234567890_left_wrist.mp4
    - episode_test_1234567890_right_wrist.mp4
"""

import argparse
import os
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Finger names for tactile visualization
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def load_h5_data(h5_path: str) -> dict:
    """Load all datasets from an H5 file."""
    data = {}
    with h5py.File(h5_path, "r") as f:

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                data[name] = obj[:]

        f.visititems(visitor)
    return data


def load_video_frames(video_path: str) -> list:
    """Load all frames from a video file."""
    if not os.path.exists(video_path):
        print(f"  Video not found: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"  Loaded {len(frames)} frames from {os.path.basename(video_path)}")
    return frames


def load_tactile_map_videos(data_dir: Path, episode_name: str, h5_data: dict) -> dict:
    """Fill missing tactile map datasets by decoding the lossless sidecar videos.

    Newer episodes store the DEFORM/RAW maps as grayscale videos (one per hand
    per type, 5 fingers tiled horizontally; video frame k == HDF5 row k, see
    data_writer.py) instead of HDF5 datasets. This decodes them back into
    (T, 5, H, W) arrays under the legacy dataset keys so the rest of the
    viewer is agnostic to the storage format.
    """
    shapes = {"deform": (240, 240), "raw": (240, 320)}
    for side in ("left", "right"):
        for ttype, (h, w) in shapes.items():
            key = f"{side}_hand_tactile_{ttype}"
            if key in h5_data:
                continue
            video_path = data_dir / f"{episode_name}_{side}_hand_tactile_{ttype}.mkv"
            if not video_path.exists():
                continue
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.ndim == 3:
                    # Monochrome source decoded as BGR with equal channels;
                    # taking one channel is exact.
                    frame = frame[:, :, 0]
                # De-tile (H, 5*W) -> (5, H, W), fingers thumb..pinky
                frames.append(frame.reshape(h, 5, w).transpose(1, 0, 2))
            cap.release()
            if frames:
                h5_data[key] = np.stack(frames)
                print(f"  Loaded {len(frames)} tactile frames from {video_path.name}")
    return h5_data


def create_tactile_figure():
    """Create figure layout for tactile visualization.

    Layout:
    - 3 rows for left hand (F6 bars, DEFORM heatmaps, RAW images)
    - 3 rows for right hand (F6 bars, DEFORM heatmaps, RAW images)
    """
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Tactile Data", fontsize=14, fontweight="bold")

    gs = GridSpec(6, 5, figure=fig, hspace=0.4, wspace=0.3)

    axes = {
        "left_f6": [fig.add_subplot(gs[0, i]) for i in range(5)],
        "left_deform": [fig.add_subplot(gs[1, i]) for i in range(5)],
        "left_raw": [fig.add_subplot(gs[2, i]) for i in range(5)],
        "right_f6": [fig.add_subplot(gs[3, i]) for i in range(5)],
        "right_deform": [fig.add_subplot(gs[4, i]) for i in range(5)],
        "right_raw": [fig.add_subplot(gs[5, i]) for i in range(5)],
    }

    # Set titles
    for i in range(5):
        axes["left_f6"][i].set_title(f"L-{FINGER_NAMES[i]}", fontsize=9)
        axes["right_f6"][i].set_title(f"R-{FINGER_NAMES[i]}", fontsize=9)

    # Add row labels
    fig.text(0.02, 0.92, "LEFT F6", fontsize=9, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.78, "LEFT DEFORM", fontsize=9, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.64, "LEFT RAW", fontsize=9, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.50, "RIGHT F6", fontsize=9, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.36, "RIGHT DEFORM", fontsize=9, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.22, "RIGHT RAW", fontsize=9, fontweight="bold", rotation=90, va="center")

    return fig, axes


def update_tactile_figure(axes, h5_data, frame_idx):
    """Update tactile visualization for a given frame."""
    # F6 data
    left_f6 = h5_data.get("left_hand_tactile_f6")
    right_f6 = h5_data.get("right_hand_tactile_f6")

    # DEFORM data
    left_deform = h5_data.get("left_hand_tactile_deform")
    right_deform = h5_data.get("right_hand_tactile_deform")

    # RAW data
    left_raw = h5_data.get("left_hand_tactile_raw")
    right_raw = h5_data.get("right_hand_tactile_raw")

    # Update left F6
    if left_f6 is not None and frame_idx < len(left_f6):
        for i in range(5):
            ax = axes["left_f6"][i]
            ax.clear()
            ax.bar(range(6), left_f6[frame_idx, i, :], color="steelblue")
            ax.set_ylim(-5, 5)
            ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
            ax.set_xticks([])
            ax.set_title(f"L-{FINGER_NAMES[i]}", fontsize=9)

    # Update right F6
    if right_f6 is not None and frame_idx < len(right_f6):
        for i in range(5):
            ax = axes["right_f6"][i]
            ax.clear()
            ax.bar(range(6), right_f6[frame_idx, i, :], color="coral")
            ax.set_ylim(-5, 5)
            ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
            ax.set_xticks([])
            ax.set_title(f"R-{FINGER_NAMES[i]}", fontsize=9)

    # Update left DEFORM
    if left_deform is not None and frame_idx < len(left_deform):
        for i in range(5):
            ax = axes["left_deform"][i]
            ax.clear()
            ax.imshow(left_deform[frame_idx, i, :, :], cmap="plasma", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        for i in range(5):
            axes["left_deform"][i].clear()
            axes["left_deform"][i].text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                transform=axes["left_deform"][i].transAxes,
            )
            axes["left_deform"][i].set_xticks([])
            axes["left_deform"][i].set_yticks([])

    # Update left RAW
    if left_raw is not None and frame_idx < len(left_raw):
        for i in range(5):
            ax = axes["left_raw"][i]
            ax.clear()
            ax.imshow(left_raw[frame_idx, i, :, :], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        for i in range(5):
            axes["left_raw"][i].clear()
            axes["left_raw"][i].text(
                0.5, 0.5, "N/A", ha="center", va="center", transform=axes["left_raw"][i].transAxes
            )
            axes["left_raw"][i].set_xticks([])
            axes["left_raw"][i].set_yticks([])

    # Update right DEFORM
    if right_deform is not None and frame_idx < len(right_deform):
        for i in range(5):
            ax = axes["right_deform"][i]
            ax.clear()
            ax.imshow(right_deform[frame_idx, i, :, :], cmap="plasma", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        for i in range(5):
            axes["right_deform"][i].clear()
            axes["right_deform"][i].text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                transform=axes["right_deform"][i].transAxes,
            )
            axes["right_deform"][i].set_xticks([])
            axes["right_deform"][i].set_yticks([])

    # Update right RAW
    if right_raw is not None and frame_idx < len(right_raw):
        for i in range(5):
            ax = axes["right_raw"][i]
            ax.clear()
            ax.imshow(right_raw[frame_idx, i, :, :], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        for i in range(5):
            axes["right_raw"][i].clear()
            axes["right_raw"][i].text(
                0.5, 0.5, "N/A", ha="center", va="center", transform=axes["right_raw"][i].transAxes
            )
            axes["right_raw"][i].set_xticks([])
            axes["right_raw"][i].set_yticks([])


def main():
    parser = argparse.ArgumentParser(description="Visualize recorded episode data")
    parser.add_argument("--episode_name", type=str, help="Episode name (without extension)")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS (default: 30)")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index")
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Data directory (default: script directory)"
    )
    args = parser.parse_args()

    episode_name = args.episode_name

    # Use script's directory as base path for episode files
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = Path(__file__).parent.resolve() / "data"

    # Find files - use new naming convention
    h5_path = data_dir / f"{episode_name}.h5"
    head_cam_path = data_dir / f"{episode_name}_head_left_rgb.mp4"
    left_wrist_path = data_dir / f"{episode_name}_left_wrist.mp4"
    right_wrist_path = data_dir / f"{episode_name}_right_wrist.mp4"

    print(f"\n{'=' * 60}")
    print(f"Episode: {episode_name}")
    print(f"Data directory: {data_dir}")
    print(f"{'=' * 60}")

    # Load H5 data
    print("\nLoading H5 data...")
    if not h5_path.exists():
        print(f"  ERROR: H5 file not found: {h5_path}")
        return
    h5_data = load_h5_data(str(h5_path))
    # Newer episodes store tactile maps as lossless sidecar videos
    h5_data = load_tactile_map_videos(data_dir, episode_name, h5_data)
    print(f"  Loaded {len(h5_data)} datasets")
    for key, arr in h5_data.items():
        print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")

    # Load video frames
    print("\nLoading video files...")
    headcam_frames = load_video_frames(str(head_cam_path))
    left_wrist_frames = load_video_frames(str(left_wrist_path))
    right_wrist_frames = load_video_frames(str(right_wrist_path))

    # Determine total frames
    num_frames = len(h5_data.get("timestamp", []))
    if num_frames == 0:
        num_frames = max(len(headcam_frames), len(left_wrist_frames), len(right_wrist_frames))
    print(f"\nTotal frames: {num_frames}")

    # Create video display window
    print("\nStarting visualization...")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  LEFT/RIGHT: Step backward/forward")
    print("  Q or ESC: Quit")
    print()

    # Create tactile figure
    tactile_fig, tactile_axes = create_tactile_figure()
    plt.ion()
    plt.show(block=False)

    frame_idx = args.start_frame
    paused = False
    delay_ms = int(1000 / args.fps)

    while frame_idx < num_frames:
        # Compose video display
        display_images = []

        # Head camera
        if frame_idx < len(headcam_frames):
            img = headcam_frames[frame_idx]
            img = cv2.resize(img, (640, 360))
            display_images.append(("Head", img))

        # Left wrist
        if frame_idx < len(left_wrist_frames):
            img = left_wrist_frames[frame_idx]
            img = cv2.resize(img, (320, 180))
            display_images.append(("L-Wrist", img))

        # Right wrist
        if frame_idx < len(right_wrist_frames):
            img = right_wrist_frames[frame_idx]
            img = cv2.resize(img, (320, 180))
            display_images.append(("R-Wrist", img))

        # Stack videos
        if display_images:
            # Create composite image
            head_img = (
                display_images[0][1]
                if len(display_images) > 0
                else np.zeros((360, 640, 3), dtype=np.uint8)
            )

            # Wrist images side by side
            if len(display_images) >= 3:
                wrist_row = np.hstack([display_images[1][1], display_images[2][1]])
            elif len(display_images) >= 2:
                wrist_row = np.hstack(
                    [display_images[1][1], np.zeros((180, 320, 3), dtype=np.uint8)]
                )
            else:
                wrist_row = np.zeros((180, 640, 3), dtype=np.uint8)

            # Stack vertically
            composite = np.vstack([head_img, wrist_row])

            # Add frame info
            cv2.putText(
                composite,
                f"Frame: {frame_idx}/{num_frames - 1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Convert RGB to BGR for OpenCV display
            composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
            cv2.imshow("Video Feeds", composite_bgr)

        # Update tactile figure
        update_tactile_figure(tactile_axes, h5_data, frame_idx)
        tactile_fig.suptitle(
            f"Tactile Data - Frame {frame_idx}/{num_frames - 1}", fontsize=14, fontweight="bold"
        )
        tactile_fig.canvas.draw_idle()
        tactile_fig.canvas.flush_events()

        # Handle keyboard input
        key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF

        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == ord(" "):  # Space - pause/resume
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'} at frame {frame_idx}")
        elif key == 81 or key == ord("a"):  # Left arrow or A
            frame_idx = max(0, frame_idx - 1)
        elif key == 83 or key == ord("d"):  # Right arrow or D
            frame_idx = min(num_frames - 1, frame_idx + 1)
        elif not paused:
            frame_idx += 1

    # Cleanup
    cv2.destroyAllWindows()
    plt.close(tactile_fig)
    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
