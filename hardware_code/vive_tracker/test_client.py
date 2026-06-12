import argparse
import json
import zmq


def run_client(server_ip: str, port: int, num_requests: int = 10) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    endpoint = f"tcp://{server_ip}:{port}"
    sock.connect(endpoint)
    sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
    print(f"[client] connected to {endpoint}")
    
    try:
        for i in range(num_requests):
            # Send request
            sock.send_string("get_vive_data")
            print(f"\n[{i+1}] Sent request: get_vive_data")
            
            # Receive response with timeout
            try:
                response = sock.recv_string()
                data = json.loads(response)
                
                # Pretty print the response
                print(json.dumps(data, indent=2))
            except zmq.Again:
                print(f"[client] timeout waiting for response")
                break
            
    except KeyboardInterrupt:
        print("\n[client] interrupted")
    except Exception as e:
        print(f"[client] error: {e}")
    finally:
        sock.close()
        ctx.term()
        print("[client] disconnected")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test client for Vive data server (ZeroMQ REQ).")
    parser.add_argument("--server", type=str, default="localhost", help="Server IP (default: localhost)")
    parser.add_argument("--port", type=int, default=5555, help="Server port (default: 5555)")
    parser.add_argument("--requests", type=int, default=10, help="Number of requests to make (default: 10)")
    args = parser.parse_args()
    run_client(server_ip=args.server, port=args.port, num_requests=args.requests)


if __name__ == "__main__":
    main()

