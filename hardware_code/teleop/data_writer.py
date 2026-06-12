"""data_writer.py: Threaded async data writer for teleoperation episodes.

Handles:
- HDF5 writing for numeric data (arm/hand positions, poses, tactile)
- MP4 writing for camera images (head, left wrist, right wrist)
- Queue-based async writing to avoid blocking main control loop
"""

import queue
import re
import shutil
import threading
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# Video Writer Hack from main_node.py
import skvideo.io
import skvideo.io.abstract as skabs
from loguru import logger
from skvideo.utils import vshape


def _writeFrame_tobytes(self, im):
    """Sends ndarray frames to FFmpeg (patched version for tobytes support)."""
    vid = vshape(im)
    T, M, N, C = vid.shape
    if not self.warmStarted:
        self._warmStart(M, N, C, im.dtype)

    vid = vid.clip(0, (1 << (self.dtype.itemsize << 3)) - 1).astype(self.dtype)
    vid = self._prepareData(vid)
    T, M, N, C = vid.shape

    if (
        self.inputdict["-pix_fmt"].startswith("yuv444p")
        or self.inputdict["-pix_fmt"].startswith("yuvj444p")
        or self.inputdict["-pix_fmt"].startswith("yuva444p")
    ):
        vid = vid.transpose((0, 3, 1, 2))

    if M != self.inputheight or N != self.inputwidth:
        raise ValueError("All images in a movie should have same size")
    if C != self.inputNumChannels:
        raise ValueError("All images in a movie should have same number of channels")

    assert self._proc is not None

    try:
        self._proc.stdin.write(vid.tobytes())
    except IOError as e:
        msg = "{0:}\n\nFFMPEG COMMAND:\n{1:}\n\nFFMPEG STDERR OUTPUT:\n".format(e, self._cmd)
        raise IOError(msg)


skabs.VideoWriterAbstract.writeFrame = _writeFrame_tobytes
# End of Video Writer Hack


# =============================================================================
# Episode file utilities
# =============================================================================


def find_last_episode_index(data_dir: Path) -> int:
    """Find the last episode index from both success and failure directories.

    Looks for subdirectories matching pattern episode_XXXX in both success/ and failure/ subdirectories.
    Returns the highest index found, or -1 if no episodes exist.

    Args:
        data_dir: Base data directory containing success/ and failure/ subdirectories

    Returns:
        Last episode index (0-indexed), or -1 if no episodes found
    """
    max_index = -1
    pattern = re.compile(r"episode_(\d{4})$")

    for subdir_name in ["success", "failure"]:
        subdir = data_dir / subdir_name
        if not subdir.exists():
            continue

        # Look for episode subdirectories
        for episode_dir in subdir.iterdir():
            if episode_dir.is_dir():
                match = pattern.match(episode_dir.name)
                if match:
                    max_index = max(max_index, int(match.group(1)))

    return max_index


def move_episode_files(data_dir: Path, episode_name: str, success: bool) -> bool:
    """Move episode subdirectory from data_dir to success/ or failure/ subdirectory.

    Moves the entire episode subdirectory containing:
    - {episode_name}.h5
    - {episode_name}_head_left_rgb.mp4
    - {episode_name}_left_wrist.mp4
    - {episode_name}_right_wrist.mp4

    Args:
        data_dir: Base data directory
        episode_name: Episode name (without extension, also the subdirectory name)
        success: True to move to success/, False to move to failure/

    Returns:
        True if the episode subdirectory was moved successfully.
        False if the subdirectory was not found, or if the move errored.
    """
    target_dir = data_dir / ("success" if success else "failure")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Source is the episode subdirectory
    src_episode_dir = data_dir / episode_name
    dst_episode_dir = target_dir / episode_name

    if not src_episode_dir.exists():
        logger.error(f"Episode subdirectory not found: {src_episode_dir}")
        return False

    if dst_episode_dir.exists():
        logger.error(f"Target episode subdirectory already exists: {dst_episode_dir}")
        return False

    try:
        shutil.move(str(src_episode_dir), str(dst_episode_dir))
        logger.info(f"Moved episode subdirectory {episode_name} to {target_dir.name}/")
        return True
    except Exception as e:
        error_msg = f"Failed to move episode subdirectory {episode_name}: {e}"
        logger.error(error_msg)
        return False


# =============================================================================
# Configuration
# =============================================================================

# Image size for all cameras
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360

# Tactile data configuration
TACTILE_TYPES_TO_STORE = ["F6", "DEFORM", "RAW"]

# Per-finger tactile map shapes (5 fingers each, thumb..pinky)
TACTILE_MAP_SHAPES = {
    "deform": (240, 240),
    "raw": (240, 320),
}

# Hand joint count
HAND_JOINT_COUNT = 22

# Video settings
VIDEO_FPS = 30

# Lossless encoder settings for the grayscale tactile map streams.
# Both are mathematically lossless; libx264 -qp 0 encodes faster and is the
# default, ffv1 is the archival-grade alternative.
TACTILE_VIDEO_CODEC_SETTINGS = {
    "libx264": {
        "-vcodec": "libx264",
        "-qp": "0",  # lossless
        "-preset": "ultrafast",
        "-pix_fmt": "gray",
    },
    "ffv1": {
        "-vcodec": "ffv1",
        "-level": "3",
        "-pix_fmt": "gray",
    },
}


class DataWriter:
    """Threaded data writer with queue-based async writing.

    Writes HDF5 for numeric data and MP4s for camera images.
    Uses a background thread to avoid blocking the main control loop.
    """

    def __init__(
        self,
        episode_name: str,
        save_dir: Path,
        command_hz: float = 30.0,
        no_wrist_cam: bool = False,
        no_head_cam: bool = False,
        tactile_maps_as_video: bool = True,
        tactile_video_codec: str = "libx264",
    ):
        """Initialize DataWriter.

        Args:
            episode_name: Name for the episode files (without extension)
            save_dir: Directory to save files in
            command_hz: Command loop frequency (used for video FPS and metadata)
            no_wrist_cam: If True, disable wrist camera recording (no MP4 files will be created)
            no_head_cam: If True, disable head camera recording (no head MP4 file will be created)
            tactile_maps_as_video: If True, store the DEFORM/RAW tactile maps as
                losslessly compressed grayscale videos (one per hand per type,
                5 fingers tiled horizontally) instead of uncompressed HDF5
                datasets. F6 always stays in the HDF5. Video frame k
                corresponds to HDF5 row k.
            tactile_video_codec: 'libx264' (lossless via -qp 0, default) or 'ffv1'.
        """
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_name = episode_name
        self.command_hz = command_hz
        self.no_wrist_cam = no_wrist_cam
        self.no_head_cam = no_head_cam
        self.tactile_maps_as_video = tactile_maps_as_video
        if tactile_video_codec not in TACTILE_VIDEO_CODEC_SETTINGS:
            raise ValueError(
                f"Unknown tactile_video_codec '{tactile_video_codec}', "
                f"expected one of {sorted(TACTILE_VIDEO_CODEC_SETTINGS)}"
            )
        self.tactile_video_codec = tactile_video_codec

        # Create episode subdirectory
        self.episode_dir = self.save_dir / episode_name
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        # File paths (within episode subdirectory)
        self.hdf5_path = self.episode_dir / f"{episode_name}.h5"
        self.head_video_path = self.episode_dir / f"{episode_name}_head_left_rgb.mp4"
        self.left_wrist_video_path = self.episode_dir / f"{episode_name}_left_wrist.mp4"
        self.right_wrist_video_path = self.episode_dir / f"{episode_name}_right_wrist.mp4"
        # Lossless tactile map videos (MKV container; gray pixel format)
        self.tactile_video_paths = {
            f"{side}_{ttype}": self.episode_dir / f"{episode_name}_{side}_hand_tactile_{ttype}.mkv"
            for side in ("left", "right")
            for ttype in TACTILE_MAP_SHAPES
        }

        # Queue and threading
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_count = 0

        # Writers (initialized in thread)
        self._hdf5_file: Optional[h5py.File] = None
        self._datasets: dict = {}
        self._video_writers: dict = {}

        # Buffers for batched writing
        self._data_buffer: list = []
        self._head_frame_buffer: list = []
        self._left_wrist_frame_buffer: list = []
        self._right_wrist_frame_buffer: list = []

    def start(self):
        """Start the writer thread."""
        self._stop_event.clear()
        self._frame_count = 0

        # Initialize HDF5 file
        self._init_hdf5()

        # Initialize video writers
        self._init_video_writers()

        # Start writer thread
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

        logger.info(f"DataWriter started for episode: {self.episode_name}")
        logger.info(f"  HDF5: {self.hdf5_path}")
        if not self.no_head_cam:
            logger.info(f"  Head video: {self.head_video_path}")
        else:
            logger.info("  Head camera: disabled")
        if not self.no_wrist_cam:
            logger.info(f"  Left wrist video: {self.left_wrist_video_path}")
            logger.info(f"  Right wrist video: {self.right_wrist_video_path}")
        else:
            logger.info("  Wrist cameras: disabled")
        if self.tactile_maps_as_video:
            logger.info(
                f"  Tactile maps: lossless {self.tactile_video_codec} videos "
                f"({self.episode_dir}/{self.episode_name}_*_hand_tactile_*.mkv)"
            )

    def _init_hdf5(self):
        """Initialize HDF5 file and create datasets."""
        self._hdf5_file = h5py.File(self.hdf5_path, "w")

        self._datasets = {
            # Timestamps
            "timestamp": self._hdf5_file.create_dataset(
                "timestamp", (0,), maxshape=(None,), dtype=np.float64, chunks=True
            ),
            "vive_timestamp": self._hdf5_file.create_dataset(
                "vive_timestamp", (0,), maxshape=(None,), dtype=np.float64, chunks=True
            ),
            "arm_timestamp": self._hdf5_file.create_dataset(
                "arm_timestamp", (0,), maxshape=(None,), dtype=np.float64, chunks=True
            ),
            "hand_timestamp": self._hdf5_file.create_dataset(
                "hand_timestamp", (0,), maxshape=(None,), dtype=np.float64, chunks=True
            ),
            # Vive poses
            "left_vive_pose": self._hdf5_file.create_dataset(
                "left_vive_pose", (0, 4, 4), maxshape=(None, 4, 4), dtype=np.float64, chunks=True
            ),
            "right_vive_pose": self._hdf5_file.create_dataset(
                "right_vive_pose", (0, 4, 4), maxshape=(None, 4, 4), dtype=np.float64, chunks=True
            ),
            # Arm targets
            "left_arm_target_dofs": self._hdf5_file.create_dataset(
                "left_arm_target_dofs", (0, 7), maxshape=(None, 7), dtype=np.float64, chunks=True
            ),
            "right_arm_target_dofs": self._hdf5_file.create_dataset(
                "right_arm_target_dofs", (0, 7), maxshape=(None, 7), dtype=np.float64, chunks=True
            ),
            "left_arm_target_pose": self._hdf5_file.create_dataset(
                "left_arm_target_pose",
                (0, 4, 4),
                maxshape=(None, 4, 4),
                dtype=np.float64,
                chunks=True,
            ),
            "right_arm_target_pose": self._hdf5_file.create_dataset(
                "right_arm_target_pose",
                (0, 4, 4),
                maxshape=(None, 4, 4),
                dtype=np.float64,
                chunks=True,
            ),
            # Arm actual state
            "left_arm_joint_positions": self._hdf5_file.create_dataset(
                "left_arm_joint_positions",
                (0, 7),
                maxshape=(None, 7),
                dtype=np.float64,
                chunks=True,
            ),
            "right_arm_joint_positions": self._hdf5_file.create_dataset(
                "right_arm_joint_positions",
                (0, 7),
                maxshape=(None, 7),
                dtype=np.float64,
                chunks=True,
            ),
            "left_arm_current_pose": self._hdf5_file.create_dataset(
                "left_arm_current_pose",
                (0, 4, 4),
                maxshape=(None, 4, 4),
                dtype=np.float64,
                chunks=True,
            ),
            "right_arm_current_pose": self._hdf5_file.create_dataset(
                "right_arm_current_pose",
                (0, 4, 4),
                maxshape=(None, 4, 4),
                dtype=np.float64,
                chunks=True,
            ),
            # Hand targets
            "left_hand_target_joint_positions": self._hdf5_file.create_dataset(
                "left_hand_target_joint_positions",
                (0, HAND_JOINT_COUNT),
                maxshape=(None, HAND_JOINT_COUNT),
                dtype=np.float64,
                chunks=True,
            ),
            "right_hand_target_joint_positions": self._hdf5_file.create_dataset(
                "right_hand_target_joint_positions",
                (0, HAND_JOINT_COUNT),
                maxshape=(None, HAND_JOINT_COUNT),
                dtype=np.float64,
                chunks=True,
            ),
            # Hand actual state
            "left_hand_joint_positions": self._hdf5_file.create_dataset(
                "left_hand_joint_positions",
                (0, HAND_JOINT_COUNT),
                maxshape=(None, HAND_JOINT_COUNT),
                dtype=np.float64,
                chunks=True,
            ),
            "right_hand_joint_positions": self._hdf5_file.create_dataset(
                "right_hand_joint_positions",
                (0, HAND_JOINT_COUNT),
                maxshape=(None, HAND_JOINT_COUNT),
                dtype=np.float64,
                chunks=True,
            ),
        }

        # Tactile datasets
        if "F6" in TACTILE_TYPES_TO_STORE:
            self._datasets["left_hand_tactile_f6"] = self._hdf5_file.create_dataset(
                "left_hand_tactile_f6",
                (0, 5, 6),
                maxshape=(None, 5, 6),
                dtype=np.float32,
                chunks=True,
            )
            self._datasets["right_hand_tactile_f6"] = self._hdf5_file.create_dataset(
                "right_hand_tactile_f6",
                (0, 5, 6),
                maxshape=(None, 5, 6),
                dtype=np.float32,
                chunks=True,
            )

        # DEFORM/RAW maps go to lossless videos by default (see
        # tactile_maps_as_video); the uncompressed HDF5 datasets are only
        # created in the legacy mode.
        if not self.tactile_maps_as_video:
            if "DEFORM" in TACTILE_TYPES_TO_STORE:
                self._datasets["left_hand_tactile_deform"] = self._hdf5_file.create_dataset(
                    "left_hand_tactile_deform",
                    (0, 5, 240, 240),
                    maxshape=(None, 5, 240, 240),
                    dtype=np.uint8,
                    chunks=True,
                )
                self._datasets["right_hand_tactile_deform"] = self._hdf5_file.create_dataset(
                    "right_hand_tactile_deform",
                    (0, 5, 240, 240),
                    maxshape=(None, 5, 240, 240),
                    dtype=np.uint8,
                    chunks=True,
                )

            if "RAW" in TACTILE_TYPES_TO_STORE:
                self._datasets["left_hand_tactile_raw"] = self._hdf5_file.create_dataset(
                    "left_hand_tactile_raw",
                    (0, 5, 240, 320),
                    maxshape=(None, 5, 240, 320),
                    dtype=np.uint8,
                    chunks=True,
                )
                self._datasets["right_hand_tactile_raw"] = self._hdf5_file.create_dataset(
                    "right_hand_tactile_raw",
                    (0, 5, 240, 320),
                    maxshape=(None, 5, 240, 320),
                    dtype=np.uint8,
                    chunks=True,
                )

    def _init_video_writers(self):
        """Initialize video writers for cameras."""
        video_fps = str(int(self.command_hz))

        video_settings = {
            "-vcodec": "libx264rgb",
            "-crf": "18",  # 18 = visually lossless
            "-preset": "ultrafast",  # Fast encoding for real-time capture
            "-pix_fmt": "rgb24",
            "-r": video_fps,
        }

        self._video_writers = {}

        # Only create head camera writer if enabled
        if not self.no_head_cam:
            self._video_writers["head"] = skvideo.io.FFmpegWriter(
                str(self.head_video_path), inputdict={"-r": video_fps}, outputdict=video_settings
            )

        # Only create wrist camera writers if enabled
        if not self.no_wrist_cam:
            self._video_writers["left_wrist"] = skvideo.io.FFmpegWriter(
                str(self.left_wrist_video_path),
                inputdict={"-r": video_fps},
                outputdict=video_settings,
            )
            self._video_writers["right_wrist"] = skvideo.io.FFmpegWriter(
                str(self.right_wrist_video_path),
                inputdict={"-r": video_fps},
                outputdict=video_settings,
            )

        # Lossless grayscale writers for the tactile maps. Frames are single
        # channel (H, 5*W) uint8 (5 fingers tiled horizontally), and the gray
        # pixel format is kept end-to-end so the encode is bit-exact.
        if self.tactile_maps_as_video:
            tactile_settings = dict(TACTILE_VIDEO_CODEC_SETTINGS[self.tactile_video_codec])
            tactile_settings["-r"] = video_fps
            for ttype in TACTILE_MAP_SHAPES:
                if ttype.upper() not in TACTILE_TYPES_TO_STORE:
                    continue
                for side in ("left", "right"):
                    key = f"{side}_{ttype}"
                    self._video_writers[f"tactile_{key}"] = skvideo.io.FFmpegWriter(
                        str(self.tactile_video_paths[key]),
                        inputdict={"-r": video_fps},
                        outputdict=tactile_settings,
                    )

    def queue_frame(self, data: dict):
        """Queue a single frame of data for writing.

        Args:
            data: Dictionary containing all data for this timestep:
                - timestamp, vive_timestep, arm_timestep, hand_timestep
                - left_vive_pose, right_vive_pose
                - left_arm_target_dofs, right_arm_target_dofs
                - left_arm_target_pose, right_arm_target_pose
                - left_arm_joint_positions, right_arm_joint_positions
                - left_arm_current_pose, right_arm_current_pose
                - left_hand_target_joint_positions, right_hand_target_joint_positions
                - left_hand_joint_positions, right_hand_joint_positions
                - head_image (optional, RGB numpy array)
                - left_wrist_image (optional, RGB numpy array)
                - right_wrist_image (optional, RGB numpy array)
                - left_tactile (optional, dict with f6, deform, raw, etc.)
                - right_tactile (optional, dict with f6, deform, raw, etc.)
        """
        self._queue.put(data)

    def _writer_loop(self):
        """Background thread that processes queued frames."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                # Get data from queue with timeout
                data = self._queue.get(timeout=0.1)
                self._process_frame(data)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"DataWriter error processing frame: {e}")

    def _process_frame(self, data: dict):
        """Process a single frame of data."""
        self._frame_count += 1
        idx = self._frame_count

        # Write numeric data to HDF5
        for key in self._datasets:
            if key not in data:
                continue

            self._datasets[key].resize(idx, axis=0)
            value = data[key]

            if len(self._datasets[key].shape) == 1:
                self._datasets[key][-1] = value
            elif len(self._datasets[key].shape) == 2:
                self._datasets[key][-1, :] = value
            elif len(self._datasets[key].shape) == 3:
                self._datasets[key][-1, :, :] = value
            else:
                self._datasets[key][-1, :, :, :] = value

        # Write tactile data. In video mode a frame must be written for every
        # control step (even if tactile data is missing) so that video frame k
        # stays aligned with HDF5 row k; missing data becomes a zero frame,
        # matching the zero-fill behavior of the legacy HDF5 path.
        self._write_tactile_data("left", data.get("left_tactile"), idx)
        self._write_tactile_data("right", data.get("right_tactile"), idx)

        # Write video frames
        if (
            "head_image" in data
            and data["head_image"] is not None
            and "head" in self._video_writers
        ):
            self._video_writers["head"].writeFrame(data["head_image"])

        if "left_wrist_image" in data and data["left_wrist_image"] is not None:
            self._video_writers["left_wrist"].writeFrame(data["left_wrist_image"])

        if "right_wrist_image" in data and data["right_wrist_image"] is not None:
            self._video_writers["right_wrist"].writeFrame(data["right_wrist_image"])

        # Flush HDF5 periodically
        if self._frame_count % 100 == 0:
            self._hdf5_file.flush()

    def _write_tactile_data(self, side: str, tactile: Optional[dict], idx: int):
        """Write tactile data for one hand (tactile=None writes zero map frames)."""
        prefix = f"{side}_hand_tactile"
        tactile = tactile or {}

        if "F6" in TACTILE_TYPES_TO_STORE and "f6" in tactile:
            key = f"{prefix}_f6"
            self._datasets[key].resize(idx, axis=0)
            self._datasets[key][-1, :, :] = tactile["f6"]

        for ttype in ("deform", "raw"):
            if ttype.upper() not in TACTILE_TYPES_TO_STORE:
                continue
            maps = tactile.get(ttype)

            if self.tactile_maps_as_video:
                writer = self._video_writers.get(f"tactile_{side}_{ttype}")
                if writer is None:
                    continue
                if maps is None:
                    h, w = TACTILE_MAP_SHAPES[ttype]
                    frame = np.zeros((h, 5 * w), dtype=np.uint8)
                else:
                    # (5, H, W) -> (H, 5*W): fingers tiled left-to-right (thumb..pinky)
                    frame = np.hstack(np.asarray(maps, dtype=np.uint8))
                writer.writeFrame(frame)
            elif maps is not None:
                key = f"{prefix}_{ttype}"
                self._datasets[key].resize(idx, axis=0)
                self._datasets[key][-1, :, :, :] = maps

    def stop(self, episode_duration: float = 0.0) -> int:
        """Stop the writer, flush all data, and close files.

        Args:
            episode_duration: Total episode duration in seconds (for metadata)

        Returns:
            Total number of frames written
        """
        logger.info("Stopping DataWriter, flushing remaining frames...")

        # Signal thread to stop
        self._stop_event.set()

        # Wait for queue to drain
        self._queue.join()

        # Wait for thread to finish
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        # Add metadata and close HDF5
        if self._hdf5_file is not None:
            self._hdf5_file.attrs["total_steps"] = self._frame_count
            self._hdf5_file.attrs["command_hz"] = self.command_hz
            self._hdf5_file.attrs["episode_duration"] = episode_duration
            self._hdf5_file.attrs["tactile_maps_storage"] = (
                "video" if self.tactile_maps_as_video else "hdf5"
            )
            if self.tactile_maps_as_video:
                self._hdf5_file.attrs["tactile_video_codec"] = self.tactile_video_codec
                self._hdf5_file.attrs["tactile_video_layout"] = (
                    "per hand+type mkv; 5 fingers (thumb..pinky) tiled horizontally; "
                    "video frame k == hdf5 row k"
                )
            self._hdf5_file.close()
            self._hdf5_file = None

        # Close video writers
        for name, writer in self._video_writers.items():
            try:
                writer.close()
            except Exception as e:
                logger.warning(f"Error closing {name} video writer: {e}")
        self._video_writers.clear()

        frames = self._frame_count
        logger.info(f"DataWriter stopped. Saved {frames} frames to {self.hdf5_path}")

        return frames
