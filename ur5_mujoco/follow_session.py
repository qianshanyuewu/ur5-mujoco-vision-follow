from __future__ import annotations

from collections import deque
import multiprocessing as mp
import queue
from multiprocessing.queues import Queue
from typing import Any

import numpy as np

from .controller import ControllerConfig
from .follow import CartesianTargetLimiter, PlanarMappingConfig, PlanarTargetMapper, TargetLimiterConfig
from .simulation import LiveTargetSample, SimulationConfig, TrackingSimulation
from .vision.follow_camera import run_camera_follow_process


MAX_OBSERVATION_AGE_MS = 150.0
TRACK_PAUSE_GRACE_MS = 750.0
DEFAULT_OBJECT_DISTANCE_M = 0.30
ARM_FOLLOW_CENTER_HISTORY_FRAMES = 3
ARM_FOLLOW_CENTER_DEADBAND_PX = 3.0
ARM_FOLLOW_STILLNESS_WINDOW_NS = 400_000_000
ARM_FOLLOW_STILLNESS_RANGE_PX = 8.0
ARM_FOLLOW_MOTION_THRESHOLD_PX = 6.0
CAMERA_FOLLOW_CONTROLLER_CONFIG = ControllerConfig(
    position_gain=5.0,
    max_linear_speed=0.08,
    max_joint_speed=0.45,
    max_joint_acceleration=2.0,
)


def _publish_latest(channel: Queue, value: dict[str, Any]) -> None:
    try:
        channel.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        channel.get_nowait()
    except queue.Empty:
        pass
    try:
        channel.put_nowait(value)
    except queue.Full:
        pass


class CameraFollowTargetSource:
    """Join the latest camera observation stream to the 100 Hz sim loop."""

    def __init__(
        self,
        observations: Queue,
        commands: Queue,
        statuses: Queue,
        marker_only: bool,
        object_distance_m: float = DEFAULT_OBJECT_DISTANCE_M,
    ) -> None:
        self.observations = observations
        self.commands = commands
        self.statuses = statuses
        self.marker_only = marker_only
        if object_distance_m <= 0:
            raise ValueError("object distance must be positive")
        self.object_distance_m = float(object_distance_m)
        self.mapper = PlanarTargetMapper(
            PlanarMappingConfig(
                horizontal_metres_per_normalized_unit=0.25,
                vertical_metres_per_normalized_unit=0.20,
            )
        )
        self._arm_center_history: deque[np.ndarray] = deque(maxlen=ARM_FOLLOW_CENTER_HISTORY_FRAMES)
        self._arm_center_frame_id: int | None = None
        self._arm_filtered_center: np.ndarray | None = None
        self._arm_output_center: np.ndarray | None = None
        self._arm_stationary = True
        self._arm_stationary_center: np.ndarray | None = None
        self._arm_motion_candidates = 0
        self._arm_motion_history: deque[tuple[int, np.ndarray]] = deque()
        self._arm_motion_started_ns: int | None = None
        self.limiter = CartesianTargetLimiter(
            TargetLimiterConfig(
                max_speed_m_s=0.35,
                max_acceleration_m_s2=2.0,
                position_gain_s_inv=10.0,
                position_deadband_m=0.001 if not marker_only else 0.0,
            )
        )
        self.hold_position: np.ndarray | None = None
        self.forward_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        self.anchor_object_position: np.ndarray | None = None
        self.latest: dict[str, Any] | None = None
        self.active_track_id: int | None = None
        self.pending_selection_track_id: int | None = None
        self.pending_selection_frame_id: int | None = None
        self.pending_selection_deadline_ns: int | None = None
        self.track_pause_started_ns: int | None = None
        self.last_marker_position: np.ndarray | None = None
        self.follow_enabled = False
        self.mapping_enabled = False
        self.quit = False
        self.status = "Waiting for a manual selection; follow is off"
        self._initialised = False

    def reset(
        self,
        initial_position: np.ndarray,
        initial_rotation: np.ndarray | None = None,
    ) -> None:
        initial_position = np.asarray(initial_position, dtype=float)
        if initial_position.shape != (3,) or not np.all(np.isfinite(initial_position)):
            raise ValueError("initial TCP position must be a finite 3-vector")
        if initial_rotation is not None:
            rotation = np.asarray(initial_rotation, dtype=float).reshape(3, 3)
            forward = rotation[:, 2]
            forward_norm = float(np.linalg.norm(forward))
            if not np.all(np.isfinite(forward)) or forward_norm < 1e-9:
                raise ValueError("initial TCP forward axis must be finite and non-zero")
            self.forward_axis = forward / forward_norm
        self.hold_position = initial_position.copy()
        self.anchor_object_position = initial_position + self.object_distance_m * self.forward_axis
        self.limiter.reset(self.anchor_object_position)
        self.last_marker_position = self.anchor_object_position.copy()
        self.track_pause_started_ns = None
        self._initialised = True
        self._publish_status()

    def _publish_status(self) -> None:
        _publish_latest(
            self.statuses,
            {"status": self.status, "follow_enabled": self.follow_enabled},
        )

    def _read_latest_observations(self) -> None:
        while True:
            try:
                message = self.observations.get_nowait()
            except queue.Empty:
                return
            if message.get("type") == "closed":
                self.quit = True
                self.follow_enabled = False
                self.mapping_enabled = False
                self.status = "Camera window closed"
                self._publish_status()
                continue
            if "frame_id" not in message:
                continue
            if self.latest is None or int(message["frame_id"]) >= int(self.latest["frame_id"]):
                self.latest = message

    @staticmethod
    def _center(message: dict[str, Any]) -> tuple[float, float] | None:
        center = message.get("center_uv")
        if center is not None and len(center) == 2:
            return (float(center[0]), float(center[1]))
        bbox = message.get("bbox_xywh")
        width = int(message.get("image_width", 0))
        height = int(message.get("image_height", 0))
        if bbox is None or width <= 0 or height <= 0:
            return None
        x, y, box_width, box_height = (float(value) for value in bbox)
        return ((x + box_width * 0.5) / width, (y + box_height * 0.5) / height)

    def _arm_follow_center(self, message: dict[str, Any]) -> tuple[float, float] | None:
        center = self._center(message)
        if center is None or self.marker_only:
            return center
        frame_id = int(message.get("frame_id", -1))
        if frame_id == self._arm_center_frame_id and self._arm_output_center is not None:
            return tuple(float(value) for value in self._arm_output_center)

        raw_center = np.asarray(center, dtype=float)
        self._arm_center_history.append(raw_center.copy())
        filtered = np.median(np.stack(tuple(self._arm_center_history)), axis=0)
        if self._arm_filtered_center is None:
            self._arm_filtered_center = filtered
        else:
            width = max(1, int(message.get("image_width", 640)))
            height = max(1, int(message.get("image_height", 480)))
            pixel_scale = np.array([width, height], dtype=float)
            delta_px = (filtered - self._arm_filtered_center) * pixel_scale
            if float(np.linalg.norm(delta_px)) >= ARM_FOLLOW_CENTER_DEADBAND_PX:
                self._arm_filtered_center = filtered
        self._arm_center_frame_id = frame_id

        assert self._arm_stationary_center is not None
        timestamp_ns = int(message.get("timestamp_ns", 0))
        pixel_scale = np.array(
            [max(1, int(message.get("image_width", 640))), max(1, int(message.get("image_height", 480)))],
            dtype=float,
        )
        if self._arm_stationary:
            displacement_px = float(
                np.linalg.norm((self._arm_filtered_center - self._arm_stationary_center) * pixel_scale)
            )
            if displacement_px >= ARM_FOLLOW_MOTION_THRESHOLD_PX:
                self._arm_motion_candidates += 1
            else:
                self._arm_motion_candidates = 0
            if self._arm_motion_candidates >= 2:
                self._arm_stationary = False
                self._arm_motion_candidates = 0
                self._arm_motion_history.clear()
                self._arm_motion_history.append((timestamp_ns, self._arm_filtered_center.copy()))
                self._arm_motion_started_ns = timestamp_ns
                output_center = self._arm_filtered_center
            else:
                output_center = self._arm_stationary_center
        else:
            if self._arm_motion_started_ns is None:
                self._arm_motion_started_ns = timestamp_ns
            self._arm_motion_history.append((timestamp_ns, self._arm_filtered_center.copy()))
            cutoff_ns = timestamp_ns - ARM_FOLLOW_STILLNESS_WINDOW_NS
            while len(self._arm_motion_history) > 1 and self._arm_motion_history[0][0] < cutoff_ns:
                self._arm_motion_history.popleft()
            # Keep the dwell timer separate because this deque only holds the trailing window.
            motion_age_ns = timestamp_ns - self._arm_motion_started_ns
            if (
                len(self._arm_motion_history) >= ARM_FOLLOW_CENTER_HISTORY_FRAMES
                and motion_age_ns >= ARM_FOLLOW_STILLNESS_WINDOW_NS
            ):
                window_centers = np.stack([sample[1] for sample in self._arm_motion_history])
                span_px = np.ptp(window_centers * pixel_scale, axis=0)
                if float(np.max(span_px)) <= ARM_FOLLOW_STILLNESS_RANGE_PX:
                    self._arm_stationary = True
                    self._arm_stationary_center = np.median(window_centers, axis=0)
                    self._arm_filtered_center = self._arm_stationary_center.copy()
                    self._arm_center_history.clear()
                    self._arm_center_history.extend(
                        self._arm_stationary_center.copy()
                        for _ in range(ARM_FOLLOW_CENTER_HISTORY_FRAMES)
                    )
                    self._arm_motion_history.clear()
                    self._arm_motion_started_ns = None
                    output_center = self._arm_stationary_center
                else:
                    output_center = self._arm_filtered_center
            else:
                output_center = self._arm_filtered_center
        self._arm_output_center = output_center.copy()
        return tuple(float(value) for value in self._arm_output_center)

    def _valid(self, message: dict[str, Any] | None, now_ns: int) -> tuple[bool, str, float | None]:
        if message is None:
            return False, "No camera observation", None
        timestamp_ns = int(message.get("timestamp_ns", 0))
        age_ms = max(0.0, (now_ns - timestamp_ns) / 1e6) if timestamp_ns else None
        if message.get("track_state") != "tracking" or message.get("bbox_xywh") is None:
            return False, f"Tracking {message.get('track_state', 'unavailable')}", age_ms
        if age_ms is None or age_ms > MAX_OBSERVATION_AGE_MS:
            return False, f"Camera observation stale ({age_ms if age_ms is not None else 'unknown'} ms)", age_ms
        if self._center(message) is None:
            return False, "Invalid object box", age_ms
        return True, "Tracking", age_ms

    def _select(self, message: dict[str, Any], current_position: np.ndarray, now_ns: int) -> None:
        valid, reason, _ = self._valid(message, now_ns)
        if not valid or int(message.get("track_id") or -1) < 0:
            self.follow_enabled = False
            self.mapping_enabled = False
            self.active_track_id = None
            self.status = f"Selection rejected: {reason}"
            self._publish_status()
            return
        center = self._center(message)
        assert center is not None
        self._arm_center_history.clear()
        self._arm_center_history.extend(
            np.asarray(center, dtype=float).copy()
            for _ in range(ARM_FOLLOW_CENTER_HISTORY_FRAMES)
        )
        self._arm_center_frame_id = int(message.get("frame_id", -1))
        self._arm_filtered_center = np.asarray(center, dtype=float)
        self._arm_output_center = self._arm_filtered_center.copy()
        self._arm_stationary = True
        self._arm_stationary_center = self._arm_filtered_center.copy()
        self._arm_motion_candidates = 0
        self._arm_motion_history.clear()
        self._arm_motion_started_ns = None
        self.hold_position = np.asarray(current_position, dtype=float).copy()
        self.anchor_object_position = self.hold_position + self.object_distance_m * self.forward_axis
        self.mapper.anchor(center, self.anchor_object_position)
        self.limiter.reset(self.anchor_object_position)
        self.last_marker_position = self.anchor_object_position.copy()
        self.track_pause_started_ns = None
        self.active_track_id = int(message["track_id"])
        self.mapping_enabled = True
        self.follow_enabled = False
        if self.marker_only:
            self.status = f"Object marker active at {self.object_distance_m:.2f} m; arm is held"
        else:
            self.status = f"Object marker active at {self.object_distance_m:.2f} m; press space to move the arm"
        self._publish_status()

    def _handle_command(self, command: dict[str, Any], current_position: np.ndarray, now_ns: int) -> None:
        kind = command.get("type")
        if kind == "ready":
            self.status = "MuJoCo is open; select an object in the camera window"
            self._publish_status()
        elif kind in ("select", "arm"):
            message = command.get("observation")
            if isinstance(message, dict) and (
                self.latest is None or int(message.get("frame_id", -1)) >= int(self.latest["frame_id"])
            ):
                self.latest = message
            if not isinstance(message, dict):
                self.status = "Selection rejected: no observation"
                self._publish_status()
                return
            track_id = int(message.get("track_id") or -1)
            if kind == "select":
                if (
                    track_id < 0
                    or message.get("track_state") != "tracking"
                    or message.get("bbox_xywh") is None
                ):
                    self._clear_pending_selection()
                    self.status = "Selection rejected: the selected box did not produce a tracked target"
                    self._publish_status()
                    return
                self.follow_enabled = False
                self.mapping_enabled = False
                self.active_track_id = None
                self.pending_selection_track_id = track_id
                self.pending_selection_frame_id = int(message.get("frame_id", -1))
                self.pending_selection_deadline_ns = now_ns + 2_000_000_000
                self.hold_position = np.asarray(current_position, dtype=float).copy()
                latest = self.latest
                if (
                    track_id >= 0
                    and latest is not None
                    and int(latest.get("track_id") or -1) == track_id
                    and int(latest.get("frame_id", -1)) > self.pending_selection_frame_id
                ):
                    valid, _, _ = self._valid(latest, now_ns)
                    if valid:
                        self._select(latest, current_position, now_ns)
                        self._clear_pending_selection()
                        return
                self.status = "Selection received; waiting for a fresh tracked frame"
                self._publish_status()
                return
            if not self.mapping_enabled or track_id != self.active_track_id:
                self._select(message, current_position, now_ns)
                return
            valid, reason, _ = self._valid(message, now_ns)
            if not valid or track_id < 0:
                self.follow_enabled = False
                self.mapping_enabled = False
                self.active_track_id = None
                self.status = f"Follow request rejected: {reason}"
                self._publish_status()
                return
            self.follow_enabled = not self.marker_only
            if self.marker_only:
                self.status = f"Object marker active at {self.object_distance_m:.2f} m; arm is held"
            else:
                self.status = f"Following track {self.active_track_id}; object distance {self.object_distance_m:.2f} m"
            self._publish_status()
        elif kind == "deselect":
            self._clear_pending_selection()
            self.follow_enabled = False
            self.mapping_enabled = False
            self.active_track_id = None
            self.hold_position = np.asarray(current_position, dtype=float).copy()
            self.status = str(command.get("reason") or "Object selection cleared")
            self._publish_status()
        elif kind in ("disarm", "error"):
            self.follow_enabled = False
            self.hold_position = np.asarray(current_position, dtype=float).copy()
            if kind == "error":
                self._clear_pending_selection()
                self.mapping_enabled = False
                self.active_track_id = None
            self.status = str(command.get("reason") or "Follow paused")
            self._publish_status()
            if kind == "error":
                self.quit = True
        elif kind == "shutdown":
            self._clear_pending_selection()
            self.follow_enabled = False
            self.mapping_enabled = False
            self.active_track_id = None
            self.hold_position = np.asarray(current_position, dtype=float).copy()
            self.status = str(command.get("reason") or "Camera session ended")
            self._publish_status()
            self.quit = True

    def _clear_pending_selection(self) -> None:
        self.pending_selection_track_id = None
        self.pending_selection_frame_id = None
        self.pending_selection_deadline_ns = None

    def _activate_pending_selection(self, current_position: np.ndarray, now_ns: int) -> None:
        track_id = self.pending_selection_track_id
        if track_id is None:
            return
        latest = self.latest
        if (
            latest is not None
            and int(latest.get("track_id") or -1) == track_id
            and int(latest.get("frame_id", -1)) > int(self.pending_selection_frame_id or -1)
        ):
            valid, _, _ = self._valid(latest, now_ns)
            if valid:
                self._select(latest, current_position, now_ns)
                self._clear_pending_selection()
                return
        if self.pending_selection_deadline_ns is not None and now_ns >= self.pending_selection_deadline_ns:
            self._clear_pending_selection()
            self.status = "Selection timed out before a fresh tracked frame arrived; select again"
            self._publish_status()

    def _read_commands(self, current_position: np.ndarray, now_ns: int) -> None:
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            self._handle_command(command, current_position, now_ns)

    def update(self, current_position: np.ndarray, now_ns: int, dt: float) -> LiveTargetSample:
        if not self._initialised:
            raise RuntimeError("live target source must be reset before use")
        self._read_latest_observations()
        self._read_commands(current_position, now_ns)
        self._activate_pending_selection(current_position, now_ns)

        desired_position: np.ndarray | None = None
        observation_age_ms: float | None = None
        valid = False
        holding_during_pause = False
        reason = self.status
        if self.mapping_enabled:
            valid, reason, observation_age_ms = self._valid(self.latest, now_ns)
            if valid and self.latest is not None:
                current_track_id = int(self.latest.get("track_id") or -1)
                if current_track_id != self.active_track_id:
                    valid = False
                    reason = "Track identity changed; reselect and arm again"
                else:
                    center = self._arm_follow_center(self.latest)
                    assert center is not None
                    desired_position = self.mapper.map_center(center)
                    self.track_pause_started_ns = None
            if not valid and self.latest is not None:
                same_track = int(self.latest.get("track_id") or -1) == self.active_track_id
                transient_pause = same_track and (
                    self.latest.get("track_state") == "paused"
                    or reason.startswith("Camera observation stale")
                )
                if transient_pause:
                    if self.track_pause_started_ns is None:
                        self.track_pause_started_ns = now_ns
                    pause_age_ms = (now_ns - self.track_pause_started_ns) / 1e6
                    if pause_age_ms <= TRACK_PAUSE_GRACE_MS:
                        valid = True
                        holding_during_pause = True
                        desired_position = (
                            self.last_marker_position.copy()
                            if self.last_marker_position is not None
                            else self.limiter.position.copy()
                        )
                        if not self.status.startswith("Tracking paused"):
                            self.status = "Tracking paused; holding object marker"
                            self._publish_status()
                    else:
                        reason = f"Tracking pause exceeded {TRACK_PAUSE_GRACE_MS:.0f} ms"
            if not valid:
                self.follow_enabled = False
                self.mapping_enabled = False
                self.active_track_id = None
                self.track_pause_started_ns = None
                self.hold_position = np.asarray(current_position, dtype=float).copy()
                self.status = f"Follow stopped: {reason}"
                self._publish_status()
            elif not holding_during_pause and self.status.startswith("Tracking paused"):
                self.status = (
                    f"Object marker active at {self.object_distance_m:.2f} m; arm is held"
                    if self.marker_only
                    else f"Object marker active at {self.object_distance_m:.2f} m; press space to move the arm"
                )
                self._publish_status()
        elif self.latest is not None:
            _, reason, observation_age_ms = self._valid(self.latest, now_ns)

        marker_target = (
            self.limiter.update(desired_position, dt)
            if self.mapping_enabled and desired_position is not None
            else None
        )
        if marker_target is not None:
            self.last_marker_position = marker_target.copy()
        assert self.hold_position is not None
        control_target = (
            marker_target - self.object_distance_m * self.forward_axis
            if self.follow_enabled and marker_target is not None
            else self.hold_position
        )
        latest = self.latest or {}
        metadata: dict[str, object] = {
            "frame_id": latest.get("frame_id", ""),
            "timestamp_ns": latest.get("timestamp_ns", ""),
            "observation_age_ms": observation_age_ms,
            "track_id": latest.get("track_id", ""),
            "track_state": latest.get("track_state", "searching"),
            "follow_enabled": self.follow_enabled,
            "mapping_enabled": self.mapping_enabled,
            "status": self.status,
            "desired_position": control_target if self.follow_enabled else None,
        }
        return LiveTargetSample(control_target, metadata, self.quit, marker_position=marker_target)


def run_camera_follow(
    simulation_config: SimulationConfig,
    controller_config: ControllerConfig,
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    max_seconds: float | None = None,
    marker_only: bool = True,
    object_distance_m: float = DEFAULT_OBJECT_DISTANCE_M,
) -> dict[str, float | str]:
    """Start an isolated camera UI process and keep the MuJoCo viewer in front."""
    context = mp.get_context("spawn")
    observations = context.Queue(maxsize=1)
    commands = context.Queue(maxsize=8)
    statuses = context.Queue(maxsize=1)
    stop_event = context.Event()
    ready_event = context.Event()
    source = CameraFollowTargetSource(
        observations,
        commands,
        statuses,
        marker_only,
        object_distance_m=object_distance_m,
    )
    simulation = TrackingSimulation(simulation_config, trajectory_factory=None, controller_config=controller_config)
    camera_process = context.Process(
        target=run_camera_follow_process,
        name="mac-camera-follow",
        args=(
            camera_index,
            width,
            height,
            observations,
            commands,
            statuses,
            stop_event,
            ready_event,
            MAX_OBSERVATION_AGE_MS,
            marker_only,
        ),
    )
    camera_process.start()
    try:
        if not ready_event.wait(timeout=20.0):
            raise TimeoutError("camera process did not finish starting within 20 seconds")
        if not camera_process.is_alive():
            camera_process.join(timeout=0.5)
            try:
                startup_message = commands.get(timeout=0.25)
            except queue.Empty:
                startup_message = None
            reason = startup_message.get("reason") if isinstance(startup_message, dict) else None
            raise RuntimeError(reason or "camera process exited before the MuJoCo session started")
        return simulation.run_live(source, max_seconds=max_seconds)
    finally:
        stop_event.set()
        camera_process.join(timeout=3.0)
        if camera_process.is_alive():
            camera_process.terminate()
            camera_process.join(timeout=2.0)
        for channel in (observations, commands, statuses):
            channel.close()
            channel.join_thread()
