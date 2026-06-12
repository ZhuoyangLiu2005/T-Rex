"""
Live streaming client that continuously displays tracker data.
"""
import argparse
import json
import time
import zmq
import sys
import os

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_tracker_data(tracker_name, data, color):
    """Format tracker data for display."""
    if data is None:
        return f"{color}{tracker_name}: NOT AVAILABLE{Colors.ENDC}"
    
    pos = data['position']
    ori = data['orientation']
    
    output = f"{color}{Colors.BOLD}{tracker_name}:{Colors.ENDC}\n"
    output += f"  {color}Position:{Colors.ENDC}\n"
    output += f"    X: {pos['x']:>8.4f} m\n"
    output += f"    Y: {pos['y']:>8.4f} m\n"
    output += f"    Z: {pos['z']:>8.4f} m\n"
    output += f"  {color}Orientation (Quaternion):{Colors.ENDC}\n"
    output += f"    X: {ori['x']:>8.4f}\n"
    output += f"    Y: {ori['y']:>8.4f}\n"
    output += f"    Z: {ori['z']:>8.4f}\n"
    output += f"    W: {ori['w']:>8.4f}\n"
    
    return output


def run_live_client(vive_names: list[str], server_ip: str, port: int, fps: int = 20, clear: bool = True) -> None:
    """Run continuous live streaming client."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    endpoint = f"tcp://{server_ip}:{port}"
    sock.connect(endpoint)
    sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
    
    print(f"{Colors.OKGREEN}[Live Client] Connected to {endpoint}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Press Ctrl+C to stop{Colors.ENDC}\n")
    time.sleep(1)
    
    frame_count = 0
    dt = 1.0 / fps
    error_count = 0
    
    try:
        while True:
            loop_start = time.time()
            
            try:
                # Send request
                sock.send_string("get_vive_data")
                
                # Receive response
                response = sock.recv_string()
                data = json.loads(response)
                
                # Clear screen for live update
                if clear:
                    clear_screen()
                
                # Display header
                print("="*60)
                print(f"{Colors.BOLD}{Colors.HEADER}  VIVE TRACKER LIVE STREAM{Colors.ENDC}")
                print("="*60)
                print(f"Frame: {frame_count:>6d}  |  FPS Target: {fps}  |  Errors: {error_count}")
                print("="*60)
                print()
                
                # Display tracker data
                for vive_name in vive_names:
                    tracker_data = data.get(vive_name)
                    color = Colors.OKBLUE if "left" in vive_name else Colors.OKGREEN if "right" in vive_name else Colors.OKCYAN
                    print(format_tracker_data(vive_name.upper(), tracker_data, color))
                    print()
                print("="*60)
                print(f"{Colors.OKCYAN}Press Ctrl+C to stop{Colors.ENDC}")
                
                frame_count += 1
                error_count = 0  # Reset error count on success
                
            except zmq.Again:
                error_count += 1
                if not clear:
                    print(f"{Colors.WARNING}[Warning] Timeout waiting for response{Colors.ENDC}")
                if error_count > 5:
                    print(f"\n{Colors.FAIL}[Error] Too many timeouts. Is the server running?{Colors.ENDC}")
                    break
            except json.JSONDecodeError as e:
                error_count += 1
                if not clear:
                    print(f"{Colors.FAIL}[Error] Invalid JSON response: {e}{Colors.ENDC}")
            except Exception as e:
                error_count += 1
                if not clear:
                    print(f"{Colors.FAIL}[Error] {e}{Colors.ENDC}")
            
            # Sleep to maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.OKGREEN}[Live Client] Stopped by user{Colors.ENDC}")
    finally:
        sock.close()
        ctx.term()
        print(f"{Colors.OKGREEN}[Live Client] Disconnected{Colors.ENDC}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live streaming client for Vive tracker data.")
    parser.add_argument("--vive-names", type=str, nargs="+", required=True, help="Space separated list of vive names to display (e.g. left_tracker right_tracker torso_tracker).")
    parser.add_argument("--server", type=str, default="127.0.0.1", help="Server IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5555, help="Server port (default: 5555)")
    parser.add_argument("--fps", type=int, default=20, help="Update rate in FPS (default: 20)")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear screen (scroll mode)")
    args = parser.parse_args()
    
    run_live_client(
        vive_names=args.vive_names,
        server_ip=args.server,
        port=args.port,
        fps=args.fps,
        clear=not args.no_clear
    )


if __name__ == "__main__":
    main()

