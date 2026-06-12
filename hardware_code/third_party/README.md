# Third-party dependencies

This directory holds all vendored third-party code and is the recommended
location for the Sharpa SDKs that the teleop stack depends on. The SDKs are
distributed by Sharpa Robotics and are **not** vendored in this repository.

## Contents (vendored)

- `dexcontrol/` (AGPL-3.0): Dexmate's robot control client, pinned at
  0.4.4 with local patches (sensor configs, `get_robot_model`). Install with
  `pip install -e third_party/dexcontrol`; its `dexbot-utils` dependency
  installs from PyPI.
- `dexmate-urdf/` (Apache-2.0): Dexmate robot descriptions (URDF/SRDF/meshes).
  Install with `pip install -e third_party/dexmate-urdf`.
- `sharpa-urdf-usd-xml/` (Apache-2.0): URDF/MJCF/USD descriptions of
  the Sharpa Wave hand. `teleop/robot_descriptions.py` loads the
  `wave_01/{left,right}_sharpa_wave/*_with_wrist.xml` models from here.

## Sharpa Wave SDK (hand control)

The `sharpa` Python module (`SharpaWave`, `SharpaWaveManager`, ...) used by
`teleop/main_teleop.py` and `teleop/arm_hand_control.py` comes from the
Sharpa Wave SDK. Download it from:

> https://github.com/sharpa-robotics/sharpa-wave-sdk

Follow that repository's instructions: grab the package for your architecture
from its Releases page, e.g. on x86_64 Ubuntu:

```bash
sudo dpkg -i sharpa-wave-sdk_<version>_amd64.deb
```

The SDK installs to `/opt/sharpa-wave-sdk/`. Make the `sharpa` module
importable in your environment, e.g.:

```bash
export PYTHONPATH=/opt/sharpa-wave-sdk/python:$PYTHONPATH
```

(`setup.sh` at the repository root does this automatically when the SDK is
installed at the default location.) Note the SDK supports Python 3.10-3.12.

## Sharpa Manus SDK (glove client + hand retargeting)

Teleoperation of the hands uses Manus MetaGloves Pro. The glove client and the
glove-to-Sharpa-hand retargeting pipeline are provided by:

> https://github.com/sharpa-robotics/sharpa-manus-sdk

Clone it into this directory (the SDK's license does not permit
redistribution, so it cannot be vendored here; the commit below is the
version this stack was last used with):

```bash
git clone https://github.com/sharpa-robotics/sharpa-manus-sdk.git third_party/sharpa-manus-sdk
git -C third_party/sharpa-manus-sdk checkout b87bff567c82905dd591d91f08be59fb90a604df
```

and set it up following its README / user manual. The teleop stack interacts
with it in two ways:

1. You run its glove client (`client/`) and retargeting publisher
   (`retargeting_alg_release_*/retargeting_manus_demo_multiprocess.py`)
   alongside the teleop processes (see `teleop_launcher.sh`).
2. `teleop/teleop_targets.py` imports the `sharpa_hand_pb2` protobuf bindings from
   the SDK's `include/proto_hand/` directory to decode the retargeted hand
   joint targets published over ZMQ. If you cloned the SDK somewhere else, set
   `SHARPA_MANUS_PROTO_DIR` to the directory containing `sharpa_hand_pb2.py`.
