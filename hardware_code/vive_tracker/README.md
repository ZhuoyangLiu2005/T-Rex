# Vive Ultimate Tracker Data Server

A ZeroMQ-based server and client system for streaming real-time data from Vive Ultimate Trackers.

## Features

- **Real-time tracker streaming** via ZeroMQ (REQ/REP pattern)
- **OpenVR integration** for Vive Ultimate Trackers
- **Live visualization client** with continuous updates
- **Simple test client** for quick validation
- **Support for any number of trackers** via an explicit name → serial mapping
- **Works without VR headset** - tracker-only mode

## Requirements

### Hardware
- 2x Vive Ultimate Trackers
- USB dongle (included with trackers)
- Windows 10/11 PC

### Software
- **Vive Hub** - For tracker pairing and management
- **SteamVR** - Provides OpenVR runtime (free, no headset required)
- **Python 3.8+**

## Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. One-Time Setup

#### Install Vive Hub
1. Download from https://www.vive.com/us/setup/
2. Launch Vive Hub
3. Pair your trackers with the USB dongle
4. Verify trackers show as "Ready" (green status)

#### Install SteamVR
1. Install Steam from https://store.steampowered.com/
2. In Steam: Library → Tools → Install "SteamVR"
3. Launch SteamVR once to initialize
4. The Ultimate Tracker driver will be automatically detected

#### Configure SteamVR for Trackers
Run the one-time configuration script:
```powershell
python -c "import json; from pathlib import Path; config_file = Path('C:/Program Files (x86)/Steam/config/steamvr.vrsettings'); config = json.load(open(config_file)) if config_file.exists() else {}; config.setdefault('steamvr', {}).update({'requireHmd': False, 'activateMultipleDrivers': True}); config.setdefault('driver_vive_ultimate_tracker', {})['enable'] = True; json.dump(config, open(config_file, 'w'), indent=2); print('✓ Configuration complete!')"
```

If SteamVR still refuses to start without a headset after this, install a
null (virtual) headset driver — see `null_driver_setup.md`
(https://github.com/username223/SteamVRNoHeadset).

### 3. Run the System

#### Terminal 1 - Start the Server
```powershell
python server.py --vive-names XXX YYY --vive-serials ZZZ WWW
```

**Expected output:**
```
[ViveTrackerReader] Initializing OpenVR...
[ViveTrackerReader] OpenVR initialized (Background mode)
[ViveTrackerReader] Discovering trackers...
  Found tracker #1: index=1, serial=58-A33400239
  Found tracker #2: index=2, serial=58-A33400783
[ViveTrackerReader] Found 2 tracker(s)
[server] listening on tcp://0.0.0.0:5555, fps=60, mode=REAL
```

#### Terminal 2 - Start the Client

**Option A: Transformation Matrices (for your application)**
```powershell
python client.py --vive-names XXX YYY
```

**Option B: Live Visualization (for monitoring)**
```powershell
python test_live_client.py --vive-names XXX YYY
```

## Project Structure

```
vive_tracker/
├── server.py              # Main server (reads trackers, serves via ZeroMQ)
├── client.py              # Main client with transformation matrices
├── test_live_client.py    # Live visualization client for testing
├── test_client.py         # Simple test client for debugging
├── vive_streamer.py       # ViveStreamer class for integration
├── base_streamer.py       # Base streamer interface
├── requirements.txt       # Python dependencies (Windows server side)
├── null_driver_setup.md   # Running SteamVR with a virtual headset driver
└── README.md              # This file
```

## Usage

### Server Options

```powershell
python server.py [OPTIONS]

Options:
  --vive-names NAME...    Tracker names (arbitrary labels, e.g. left_tracker right_tracker)
  --vive-serials SN...    Tracker serial numbers, one per name (printed at discovery)
  --bind IP               Bind IP address (default: 0.0.0.0)
  --port PORT         Server port (default: 5555)
  --fps FPS           Update rate (default: 60)
  --mock              Use simulated data instead of real trackers
```

### Main Client (Transformation Matrices)

```powershell
python client.py [OPTIONS]

Options:
  --vive-names NAME...    Tracker names (must match the server's --vive-names)
  --ip IP             Server IP address (default: 127.0.0.1)
  --port PORT         Server port (default: 5555)
  --fps FPS           Loop rate (default: 20)
  --print-every N     Print every N frames (default: 10)
```

### Live Visualization Client (Testing)

```powershell
python test_live_client.py [OPTIONS]

Options:
  --vive-names NAME...    Tracker names (must match the server's --vive-names)
  --server IP         Server IP address (default: localhost)
  --port PORT         Server port (default: 5555)
  --fps FPS           Display update rate (default: 20)
  --no-clear          Don't clear screen (scroll mode)
```

### Using with Your Code

```python
from vive_tracker.vive_streamer import ViveStreamer

# Create streamer (names must match the server's --vive-names)
streamer = ViveStreamer(
    vive_names=["left_tracker", "right_tracker"], ip="127.0.0.1", port=5555, fps=20
)
streamer.start_streaming()

# Get tracker data
output = streamer.get()

# Access transformation matrices
if output.vive_data:
    left_transform = output.vive_data.get("left_tracker")   # 4x4 matrix
    right_transform = output.vive_data.get("right_tracker") # 4x4 matrix

# Clean up
streamer.stop_streaming()
```

This is exactly how `teleop/teleop_targets.py` consumes the trackers: launch the
server on the Windows PC with

```powershell
python server.py --vive-names left_tracker right_tracker --vive-serials <LEFT_SERIAL> <RIGHT_SERIAL> --bind 0.0.0.0
```

(the serials are printed at server startup during tracker discovery), and set
the `vive:` section of `config/default.yaml` (IP of the Windows PC, tracker
names) accordingly.

## Data Format

The server sends JSON data with position (meters) and orientation (quaternion):

```json
{
  "left_tracker": {
    "position": {
      "x": 0.1234,
      "y": 1.2345,
      "z": -0.5678
    },
    "orientation": {
      "x": 0.0012,
      "y": 0.0034,
      "z": 0.0056,
      "w": 0.9998
    }
  },
  "right_tracker": {
    "position": { "x": 0.4567, "y": 1.2456, "z": -0.7890 },
    "orientation": { "x": 0.0023, "y": 0.0045, "z": 0.0067, "w": 0.9997 }
  }
}
```

## Troubleshooting

### No trackers found
- **Check Vive Hub:** Trackers should show as "Ready" (green)
- **Wake trackers:** Move them to exit sleep mode
- **Restart services:**
  1. Close SteamVR
  2. Close Vive Hub
  3. Start Vive Hub first
  4. Then start SteamVR
  5. Try the server again

### OpenVR initialization failed
- **Make sure SteamVR is running** before starting the server
- **Check system tray:** Look for the SteamVR icon
- **Restart SteamVR:** Exit completely and relaunch

### Server crashes or connection issues
- **Check firewall:** Allow Python through Windows Firewall
- **Port conflict:** Make sure port 5555 is not in use
- **Try different port:** Use `--port 5556` on both server and client

### Trackers detected but no data
- **Move the trackers:** They might be in sleep mode
- **Check battery:** Low battery can cause issues
- **Re-pair in Vive Hub:** Unpair and pair again

## Network Usage

### Local Machine (Default)
```powershell
# Server
python server.py --vive-names left_tracker right_tracker --vive-serials <SN1> <SN2>

# Client (same machine)
python test_live_client.py --vive-names left_tracker right_tracker --server localhost
```

### Remote Access
```powershell
# Server (on PC with trackers)
python server.py --vive-names left_tracker right_tracker --vive-serials <SN1> <SN2> --bind 0.0.0.0

# Client (on another PC, same network)
python test_live_client.py --vive-names left_tracker right_tracker --server 192.168.1.100
```

Replace `192.168.1.100` with your server PC's IP address.

## Architecture

```
[Vive Ultimate Trackers]
        ↓ (wireless via dongle)
    [Vive Hub]
        ↓ (driver)
    [SteamVR OpenVR Driver]
        ↓ (OpenVR API)
    [server.py]
        ↓ (ZeroMQ REQ/REP)
    [client.py / vive_streamer.py]
        ↓
    [Your Application]
```

## Notes

- **Tracker Assignment:** trackers are identified by serial number via the
  `--vive-names`/`--vive-serials` mapping (names are arbitrary; serials are
  printed during discovery at server startup)
- **Coordinate System:** OpenVR standing universe (Y-up, right-handed)
- **Latency:** ~16-33ms typical (depending on FPS settings)
- **Headset Not Required:** System works with trackers only
- **Vive Hub Required:** Must be running for tracker communication
- **SteamVR Required:** Provides the OpenVR runtime

## Useful Links

- [Vive Developer Portal](https://developer.vive.com/)
- [OpenVR Documentation](https://github.com/ValveSoftware/openvr)
- [Vive Hub Download](https://www.vive.com/us/setup/)
- [SteamVR](https://store.steampowered.com/app/250820/SteamVR/)

## License

Covered by the repository's top-level MIT license.

