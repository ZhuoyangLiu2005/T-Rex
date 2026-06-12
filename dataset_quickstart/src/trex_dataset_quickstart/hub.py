"""Fetch just what one episode needs from the HF Hub — never the whole dataset.

Two-step access:
  1. `meta_root` pulls only `meta/` (a few MB) so browsing/filtering is cheap.
  2. `fetch_episode` resolves the exact data parquet + 23 video files an episode
     points at (via the episodes parquet + `info.json` templates), size-checks
     against a hard cap, and downloads only those.

`source` is either a local dataset dir (used as-is) or an HF `repo_id`. Videos are
merged into ~200 MB files holding many episodes, so one episode pulls the whole
file(s) that contain it per key — `max_gb` (default 10) is a hard stop.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import pandas as pd


def is_local_dataset(source: str | Path) -> bool:
    """True if `source` is a local dataset dir (has meta/info.json)."""
    p = Path(source)
    return p.exists() and (p / "meta" / "info.json").exists()


def default_cache_dir(repo_id: str) -> Path:
    return Path.home() / ".cache" / "trex_dataset_quickstart" / repo_id.replace("/", "__")


def read_info(root: str | Path) -> dict:
    return json.loads((Path(root) / "meta" / "info.json").read_text())


def load_episodes(root: str | Path) -> pd.DataFrame:
    """Concatenate all episode-metadata parquet rows under `root/meta/episodes`."""
    files = sorted(
        glob.glob(str(Path(root) / "meta" / "episodes" / "**" / "*.parquet"), recursive=True)
    )
    if not files:
        raise FileNotFoundError(f"no episodes parquet under {root}/meta/episodes")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def download_meta(repo_id: str, cache_dir: str | Path, revision: str | None = None) -> Path:
    """Download only `meta/` for `repo_id` into `cache_dir`; return the local root."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["meta/**"],
        local_dir=str(cache_dir),
    )
    return Path(local)


def meta_root(
    source: str | Path, cache_dir: str | Path | None = None, revision: str | None = None
) -> Path:
    """Return a local root with `meta/` present (browse only — no frame download)."""
    if is_local_dataset(source):
        return Path(source)
    repo_id = str(source)
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir(repo_id)
    if not (cache_dir / "meta" / "info.json").exists():
        download_meta(repo_id, cache_dir, revision)
    return cache_dir


def _video_keys(info: dict) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def resolve_episode_files(info: dict, ep_rows: pd.DataFrame, include_videos: bool) -> list[str]:
    """Repo-relative paths the given episode rows reference (1 data parquet + video files)."""
    files: set[str] = set()
    for _, row in ep_rows.iterrows():
        files.add(
            info["data_path"].format(
                chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
            )
        )
    if include_videos:
        for key in _video_keys(info):
            for _, row in ep_rows.iterrows():
                files.add(
                    info["video_path"].format(
                        video_key=key,
                        chunk_index=int(row[f"videos/{key}/chunk_index"]),
                        file_index=int(row[f"videos/{key}/file_index"]),
                    )
                )
    return sorted(files)


def estimate_bytes(repo_id: str, files: list[str], revision: str | None = None) -> int:
    from huggingface_hub import HfApi

    repo = HfApi().repo_info(
        repo_id=repo_id, repo_type="dataset", revision=revision, files_metadata=True
    )
    sizes = {s.rfilename: (s.size or 0) for s in repo.siblings}
    return sum(sizes.get(f, 0) for f in files)


def fetch_episode(
    source: str | Path,
    episode_index: int,
    cache_dir: str | Path | None = None,
    max_gb: float = 10.0,
    include_videos: bool = True,
    revision: str | None = None,
) -> Path:
    """Make one episode available locally; return a dataset root.

    Local `source` is used as-is (no download). For a Hub `repo_id`, only the files
    that episode needs are downloaded, after a `max_gb` size check. Pass
    `include_videos=False` for replay (needs only the tiny data parquet).
    """
    if is_local_dataset(source):
        return Path(source)

    repo_id = str(source)
    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir(repo_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not (cache_dir / "meta" / "info.json").exists():
        download_meta(repo_id, cache_dir, revision)
    info = read_info(cache_dir)

    rows = load_episodes(cache_dir)
    rows = rows[rows["episode_index"] == episode_index]
    if len(rows) == 0:
        raise ValueError(f"episode_index {episode_index} not in {repo_id}")

    files = resolve_episode_files(info, rows, include_videos)
    gb = estimate_bytes(repo_id, files, revision) / 1e9
    logging.info("episode %d needs %d files (%.2f GB)", episode_index, len(files), gb)
    if gb > max_gb:
        raise RuntimeError(
            f"episode {episode_index} would download {gb:.2f} GB, over the {max_gb} GB cap. "
            "Raise max_gb, or use include_videos=False (replay) for a tiny pull."
        )
    for f in files:
        hf_hub_download(
            repo_id=repo_id,
            filename=f,
            repo_type="dataset",
            revision=revision,
            local_dir=str(cache_dir),
        )
    return cache_dir
