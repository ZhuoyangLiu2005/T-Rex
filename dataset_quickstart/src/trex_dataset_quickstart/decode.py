"""Decode per-episode video frames + numeric arrays directly with av/pandas.

Works on the merged v3.0 layout (size-bounded multi-episode mp4s): each episode's
slice of a video file is located via the `videos/<key>/from_timestamp` pointer in
the episodes parquet. No LeRobot loader / torch required.
"""

from __future__ import annotations

import glob
from pathlib import Path

import av
import numpy as np
import pandas as pd

from .schema import FPS, N_FINGERS, TACTILE_F6_DIM


FORCE_COMPS = (0, 1, 2)  # fx, fy, fz
MOMENT_COMPS = (3, 4, 5)  # mx, my, mz


def decode_key(
    root: str | Path, key: str, ep_row: pd.Series, length: int, gray: bool
) -> np.ndarray:
    """Decode exactly `length` frames of one video key for one episode.

    Returns (length, H, W) for gray tactile, else (length, H, W, 3) RGB.
    """
    root = Path(root)
    ci = int(ep_row[f"videos/{key}/chunk_index"])
    fi = int(ep_row[f"videos/{key}/file_index"])
    from_ts = float(ep_row[f"videos/{key}/from_timestamp"])
    path = root / "videos" / key / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"video file missing: {path}")
    start_idx = int(round(from_ts * FPS))
    fmt = "gray" if gray else "rgb24"

    def collect(do_seek: bool) -> list[np.ndarray] | None:
        out: list[np.ndarray] = []
        first: int | None = None
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            tb = st.time_base
            if do_seek and start_idx > 0:
                # seek to a keyframe just before our start, then drop the lead-in
                c.seek(max(int((from_ts - 0.1) / tb), 0), stream=st, backward=True, any_frame=False)
            for fr in c.decode(video=0):
                if fr.pts is None:
                    return None  # can't align by timestamp; caller falls back
                idx = int(round(float(fr.pts * tb) * FPS))
                if idx < start_idx:
                    continue
                if first is None:
                    first = idx
                out.append(fr.to_ndarray(format=fmt))
                if len(out) >= length:
                    break
        if first != start_idx or len(out) < length:
            return None
        return out

    got = collect(do_seek=True)
    if got is None:
        got = collect(do_seek=False)  # decode from file start, no seek
    if got is None:
        raise RuntimeError(f"could not extract {length} aligned frames for {key} from {path}")
    return np.stack(got)


def _episode_data(root: str | Path, episode_index: int, columns: list[str]) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(root) / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no data parquet under {root}/data")
    cols = ["episode_index", "frame_index", *columns]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    return df[df["episode_index"] == episode_index].sort_values("frame_index")


def load_episode_f6(root: str | Path, episode_index: int, length: int) -> np.ndarray:
    """Load the (length, 60) tactile-force array for one episode."""
    g = _episode_data(root, episode_index, ["observation.tactile_force"])
    f6 = np.stack(g["observation.tactile_force"].to_numpy()).astype(np.float32)
    expected = (length, 2 * N_FINGERS * TACTILE_F6_DIM)
    if f6.shape != expected:
        raise ValueError(f"f6 shape {f6.shape} != {expected}")
    return f6


def load_episode_state(root: str | Path, episode_index: int, length: int) -> np.ndarray:
    """Load the (length, 58) `observation.state` array for one episode (for replay)."""
    g = _episode_data(root, episode_index, ["observation.state"])
    state = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
    if state.shape[0] != length:
        raise ValueError(f"state length {state.shape[0]} != {length}")
    return state


def load_episode_action(root: str | Path, episode_index: int, length: int) -> np.ndarray:
    """Load the (length, 58) `action` (target joint positions) array for one episode."""
    g = _episode_data(root, episode_index, ["action"])
    action = np.stack(g["action"].to_numpy()).astype(np.float64)
    if action.shape[0] != length:
        raise ValueError(f"action length {action.shape[0]} != {length}")
    return action


def f6_slice(f6: np.ndarray, side: str, finger_idx: int) -> np.ndarray:
    """Return (..., 6) for one hand/finger."""
    base = (0 if side == "left" else 30) + finger_idx * TACTILE_F6_DIM
    return f6[..., base : base + TACTILE_F6_DIM]


def force_magnitude(f6: np.ndarray) -> np.ndarray:
    """(T, 2, N_FINGERS) per-frame sqrt(fx^2+fy^2+fz^2) for each hand/finger."""
    out = np.zeros((f6.shape[0], 2, N_FINGERS), dtype=np.float32)
    for s, side in enumerate(("left", "right")):
        for fi in range(N_FINGERS):
            comp = f6_slice(f6, side, fi)[:, FORCE_COMPS[0] : FORCE_COMPS[-1] + 1]
            out[:, s, fi] = np.linalg.norm(comp, axis=-1)
    return out
