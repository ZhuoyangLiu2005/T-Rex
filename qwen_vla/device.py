"""Device selection helpers: CUDA, Intel XPU or CPU."""

import torch


def get_default_device() -> torch.device:
    """CUDA when available, else Intel XPU, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def empty_cache(device_type: str) -> None:
    """Release cached allocator memory on the given device type (no-op on CPU)."""
    backend = getattr(torch, device_type, None)
    if backend is not None and hasattr(backend, "empty_cache") and backend.is_available():
        backend.empty_cache()
