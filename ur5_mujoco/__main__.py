from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller import ControllerConfig
from .simulation import SimulationConfig, TrackingSimulation
from .trajectories import CircleTrajectory, LineTrajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Track smooth TCP trajectories with a simulated UR5 + 2F-85")
    parser.add_argument("--trajectory", choices=("line", "circle"), default="line")
    parser.add_argument("--seconds", type=float, default=None, help="run duration in seconds (default: 60; camera follow runs until closed)")
    parser.add_argument("--speed", type=float, default=0.02, help="trajectory speed in metres/second")
    parser.add_argument("--radius", type=float, default=0.04, help="line amplitude or circle radius in metres")
    parser.add_argument("--gripper", choices=("open", "closed"), default="open")
    parser.add_argument("--viewer", action="store_true", help="show the interactive MuJoCo viewer")
    parser.add_argument("--camera-follow", action="store_true", help="keep the MuJoCo viewer open while the Mac camera tracks a manually selected object")
    parser.add_argument("--move-arm", action="store_true", help="allow camera-follow input to move the simulated UR5 arm; default is marker-only preview")
    parser.add_argument("--camera", type=int, default=0, help="camera device index for --camera-follow")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--object-distance",
        type=float,
        default=0.30,
        help="assumed distance in metres from the gripper TCP to the visual object marker",
    )
    parser.add_argument("--no-realtime", action="store_true", help="run the viewer as fast as possible")
    parser.add_argument("--output", type=Path, default=None, help="CSV log path (default: logs/tracking-<time>.csv)")
    args = parser.parse_args()

    if args.speed <= 0 or args.radius <= 0:
        parser.error("--speed and --radius must be positive")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("--camera-width and --camera-height must be positive")
    if args.object_distance <= 0:
        parser.error("--object-distance must be positive")
    if args.move_arm and not args.camera_follow:
        parser.error("--move-arm requires --camera-follow")
    if args.camera_follow and args.no_realtime:
        parser.error("--camera-follow requires real-time simulation; remove --no-realtime")
    def make_trajectory(start):
        if args.trajectory == "line":
            return LineTrajectory(start=start, amplitude=args.radius, speed=args.speed)
        return CircleTrajectory(start=start, radius=args.radius, speed=args.speed)

    config = SimulationConfig(
        seconds=args.seconds if args.seconds is not None else 60.0,
        gripper_command=255.0 if args.gripper == "open" else 0.0,
        output_path=args.output,
        viewer=args.viewer,
        realtime=not args.no_realtime,
    )
    if args.camera_follow:
        from .follow_session import CAMERA_FOLLOW_CONTROLLER_CONFIG, run_camera_follow

        result = run_camera_follow(
            config,
            CAMERA_FOLLOW_CONTROLLER_CONFIG,
            camera_index=args.camera,
            width=args.camera_width,
            height=args.camera_height,
            max_seconds=args.seconds,
            marker_only=not args.move_arm,
            object_distance_m=args.object_distance,
        )
    else:
        simulation = TrackingSimulation(config, make_trajectory, ControllerConfig())
        result = simulation.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
