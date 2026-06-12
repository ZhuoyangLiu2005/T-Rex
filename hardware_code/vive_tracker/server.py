import argparse
import json
import math
import time
from typing import Dict, Optional, List

import zmq

try:
    import openvr
    OPENVR_AVAILABLE = True
except ImportError:
    OPENVR_AVAILABLE = False
    print("[WARNING] OpenVR not available. Install with: pip install openvr")


def mock_tracker_pose(t: float, phase: float = 0.0) -> Dict[str, Dict[str, float]]:
    x = 0.30 + 0.05 * math.sin(t + phase)
    y = 0.20 + 0.03 * math.cos(0.7 * t + phase)
    z = 0.10 + 0.04 * math.sin(1.3 * t + phase)
    # Unit quaternion (no rotation)
    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    return {
        "position": {"x": x, "y": y, "z": z},
        "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
    }


class ViveTrackerReader:
    """Read data from real Vive trackers using OpenVR."""
    
    def __init__(self, vive_name_to_serial: Dict[str, str]):
        if not OPENVR_AVAILABLE:
            raise RuntimeError("OpenVR is not installed. Install with: pip install openvr")
        
        print("[ViveTrackerReader] Initializing OpenVR...")
        try:
            # Try Background mode first (what ViveTrackerServer uses)
            try:
                self.vr_system = openvr.init(openvr.VRApplication_Background)
                print("[ViveTrackerReader] OpenVR initialized (Background mode)")
            except:
                # Fall back to Utility mode
                self.vr_system = openvr.init(openvr.VRApplication_Utility)
                print("[ViveTrackerReader] OpenVR initialized (Utility mode)")
        except openvr.OpenVRError as e:
            raise RuntimeError(f"Failed to initialize OpenVR. Is SteamVR running? Error: {e}")
        assert vive_name_to_serial, "vive_name_to_serial mapping is required to identify trackers by serial number"
        self.vive_name_to_serial = vive_name_to_serial
        self.serial_to_steamvr_index = {}
        self._discover_trackers()
    
    def _discover_trackers(self):
        """Find all connected Vive trackers."""
        print("[ViveTrackerReader] Discovering trackers...")
        self.serial_to_steamvr_index = {}
        
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.vr_system.getTrackedDeviceClass(i)
            if device_class == openvr.TrackedDeviceClass_GenericTracker:
                # Get device serial number for identification
                serial = self._get_device_property(i, openvr.Prop_SerialNumber_String)
                self.serial_to_steamvr_index[serial] = i
                print(f"  Found tracker #{len(self.serial_to_steamvr_index)}: index={i}, serial={serial}")
        
        if not self.serial_to_steamvr_index:
            print("[WARNING] No Vive trackers found! Make sure:")
            print("  1. SteamVR is running")
            print("  2. Trackers are turned on and paired")
            print("  3. Trackers appear in SteamVR interface")
        else:
            print(f"[ViveTrackerReader] Found {len(self.serial_to_steamvr_index)} tracker(s)")
            
            # If serials are configured, show assignment
            for vive_name, vive_serial in self.vive_name_to_serial.items():
                if vive_serial not in self.serial_to_steamvr_index:
                    raise RuntimeError(f"[WARNING] Configured vive name {vive_name} with serial {vive_serial} not found among connected trackers")
    
    def _get_device_property(self, device_index: int, prop: int) -> str:
        """Get a string property from a device."""
        try:
            result = self.vr_system.getStringTrackedDeviceProperty(device_index, prop)
            return result.decode('utf-8') if isinstance(result, bytes) else result
        except:
            return "unknown"
    
    def _matrix_to_quaternion(self, m):
        """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
        trace = m[0][0] + m[1][1] + m[2][2]
        
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2][1] - m[1][2]) * s
            y = (m[0][2] - m[2][0]) * s
            z = (m[1][0] - m[0][1]) * s
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
            w = (m[2][1] - m[1][2]) / s
            x = 0.25 * s
            y = (m[0][1] + m[1][0]) / s
            z = (m[0][2] + m[2][0]) / s
        elif m[1][1] > m[2][2]:
            s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
            w = (m[0][2] - m[2][0]) / s
            x = (m[0][1] + m[1][0]) / s
            y = 0.25 * s
            z = (m[1][2] + m[2][1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
            w = (m[1][0] - m[0][1]) / s
            x = (m[0][2] + m[2][0]) / s
            y = (m[1][2] + m[2][1]) / s
            z = 0.25 * s
        
        return w, x, y, z

    def _steam_vr_pose_to_dict(self, pose) -> Dict[str, Dict[str, float]]:
        # Extract transformation matrix
        mat = pose.mDeviceToAbsoluteTracking
        
        # Extract position (last column, first 3 rows)
        x = mat[0][3]
        y = mat[1][3]
        z = mat[2][3]
        
        # Extract rotation matrix and convert to quaternion
        rotation_matrix = [
            [mat[0][0], mat[0][1], mat[0][2]],
            [mat[1][0], mat[1][1], mat[1][2]],
            [mat[2][0], mat[2][1], mat[2][2]]
        ]
        qw, qx, qy, qz = self._matrix_to_quaternion(rotation_matrix)
        
        return {
            "position": {"x": float(x), "y": float(y), "z": float(z)},
            "orientation": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)},
        }

    def get_all_tracker_poses(self) -> Dict[str, Optional[Dict[str, Dict[str, float]]]]:
        # Get poses directly (don't use VRCompositor - it requires HMD)
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )
        result = {}
        for vive_name, vive_serial in self.vive_name_to_serial.items():
            tracker_index = self.serial_to_steamvr_index[vive_serial]
            pose = poses[tracker_index]
            if not pose.bPoseIsValid:
                result[vive_name] = None
            else:
                result[vive_name] = self._steam_vr_pose_to_dict(pose)
        return result

    def get_tracker_pose_by_serial(self, serial: str) -> Optional[Dict[str, Dict[str, float]]]:
        # Get poses directly (don't use VRCompositor - it requires HMD)
        if serial not in self.serial_to_steamvr_index:
            return None
        tracker_index = self.serial_to_steamvr_index[serial]
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )
        pose = poses[tracker_index]
        if not pose.bPoseIsValid:
            return None
        else:
            return self._steam_vr_pose_to_dict(pose)

    def shutdown(self):
        """Shutdown OpenVR."""
        try:
            openvr.shutdown()
            print("[ViveTrackerReader] OpenVR shut down")
        except:
            pass


def run_server(
    vive_name_to_serial: Dict[str, str],
    bind_ip: str,
    port: int,
    fps: int,
    use_mock: bool = False
) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    endpoint = f"tcp://{bind_ip}:{port}"
    sock.bind(endpoint)
    
    # Initialize tracker reader
    tracker_reader = None
    if not use_mock:
        tracker_reader = ViveTrackerReader(vive_name_to_serial=vive_name_to_serial)
    
    mode_str = "MOCK" if use_mock else "REAL"
    print(f"[server] listening on {endpoint}, fps={fps}, mode={mode_str}")
    dt = 1.0 / float(fps)
    
    try:
        while True:
            try:
                # Use polling with timeout so Ctrl+C can interrupt
                if sock.poll(timeout=100):  # 100ms timeout
                    msg = sock.recv_string(flags=zmq.NOBLOCK)
                else:
                    continue
            except zmq.Again:
                # No message available, continue
                continue
            except zmq.ZMQError as e:
                print(f"[server] recv error: {e}")
                continue
            
            if msg != "get_vive_data":
                sock.send_string(json.dumps({"error": "unknown_request"}))
                continue
            
            # Get tracker data (real or mock)
            if use_mock:
                now = time.time()
                total_vives = max(1, len(vive_name_to_serial))
                payload = {}
                for idx, vive_name in enumerate(vive_name_to_serial.keys()):
                    payload[vive_name] = mock_tracker_pose(now, phase=idx * math.pi / (2 * total_vives))
            else:
                # If serials are specified, use them to get specific trackers
                payload = tracker_reader.get_all_tracker_poses()
            
            try:
                sock.send_string(json.dumps(payload))
            except zmq.ZMQError as e:
                print(f"[server] send error: {e}")
            
            # simple pacing (not required for REP, but helps CPU)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        if tracker_reader:
            tracker_reader.shutdown()
        sock.close(0)
        ctx.term()
        print("[server] stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vive tracker data server (ZeroMQ REP).")
    parser.add_argument(
        "--vive-names",
        type=str,
        nargs="+",
        required=True,
        help="Space separated list of vive names (e.g. left_tracker,right_tracker,torso_tracker). These will be used as keys in the output JSON.",
    )
    parser.add_argument(
        "--vive-serials",
        type=str,
        nargs="+",
        required=True,
        help="Space separated list of vive serial numbers corresponding to the vive names.",
    )
    parser.add_argument("--bind", type=str, default="127.0.0.1", help="Bind IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5555, help="Bind port (default: 5555)")
    parser.add_argument("--fps", type=int, default=60, help="Update pacing (default: 60)")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of real Vive trackers")
    args = parser.parse_args()
    assert len(args.vive_names) == len(args.vive_serials), "Number of vive names must match number of serials"
    assert len(args.vive_names) > 0, "At least one vive name and serial must be provided"
    run_server(
        vive_name_to_serial=dict(zip(args.vive_names, args.vive_serials)),
        bind_ip=args.bind,
        port=args.port, 
        fps=args.fps, 
        use_mock=args.mock
    )


if __name__ == "__main__":
    main()


