import argparse
import time
from typing import Optional

import numpy as np

try:
    from .vive_streamer import ViveStreamer
except ImportError:
    from vive_streamer import ViveStreamer


def format_matrix(matrix: np.ndarray) -> str:
    rows = []
    for r in range(matrix.shape[0]):
        rows.append(" ".join(f"{val: .4f}" for val in matrix[r]))
    return "\n".join(rows)


def run(vive_names: list[str], ip: str, port: int, fps: int, print_every: int) -> None:
    streamer = ViveStreamer(vive_names=vive_names, ip=ip, port=port, fps=fps)
    streamer.start_streaming()
    period_s = 1.0 / float(fps)
    iter_count = 0
    try:
        while True:
            loop_start = time.time()
            out = streamer.get()
            if print_every > 0 and (iter_count % print_every == 0):
                if out.vive_data:
                    for vive_name in vive_names:
                        if vive_name in out.vive_data:
                            print(f"[{vive_name}]")
                            print(format_matrix(out.vive_data[vive_name]))
                else:
                    print("No Vive data received in this frame.")
            iter_count += 1
            elapsed = time.time() - loop_start
            to_sleep = period_s - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop_streaming()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ViveStreamer loop and print wrist transforms.")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP of the Vive data server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5555, help="Port of the Vive data server (default: 5555)")
    parser.add_argument("--fps", type=int, default=20, help="Loop rate in frames per second (default: 20)")
    parser.add_argument(
        "--vive-names",
        type=str,
        nargs="+",
        required=True,
        help="Space separated list of vive names to display (e.g. left_tracker right_tracker torso_tracker).",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="How many frames between prints (0 disables printing, default: 10)",
    )
    args = parser.parse_args()
    run(vive_names=args.vive_names, ip=args.ip, port=args.port, fps=args.fps, print_every=args.print_every)


if __name__ == "__main__":
    main()
