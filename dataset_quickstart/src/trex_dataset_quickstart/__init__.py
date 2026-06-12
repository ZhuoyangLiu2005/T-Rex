"""TRex tactile-play quickstart toolkit.

Light core (`hub`, `decode`, `schema`) has no heavy deps. `robot` pulls in
pinocchio + viser and the URDFs under `third_party/` — import it only when doing
replay (see `scripts/replay.ipynb`).
"""

__version__ = "0.1.0"
