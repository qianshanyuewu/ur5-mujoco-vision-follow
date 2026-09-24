from __future__ import annotations

import queue
import time
from multiprocessing.queues import Queue
from threading import Event
from typing import Any

import cv2

from .camera import LatestFrameCamera
from .hand import MediaPipeHandDetector, draw_hand_observations
from .tracking import CSRTObjectTracker, TrackObservation, TrackState


def _put_latest(channel: Queue, value: dict[str, Any]) -> None:
    """Publish only the newest observation; old frames must never build up."""
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
        # The consumer raced with this producer; the next camera frame replaces it.
        pass


def _put_command(channel: Queue, value: dict[str, Any]) -> None:
    try:
        channel.put_nowait(value)
    except queue.Full:
        # Keep the latest operator command if the simulation is briefly busy.
        while True:
            try:
                channel.get_nowait()
            except queue.Empty:
                break
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass


def observation_message(
    frame_id: int,
    timestamp_ns: int,
    image_width: int,
    image_height: int,
    hand_count: int,
    observation: TrackObservation,
) -> dict[str, Any]:
    center = _center_uv(observation, image_width, image_height)
    return {
        "frame_id": frame_id,
        "timestamp_ns": timestamp_ns,
        "image_width": image_width,
        "image_height": image_height,
        "hand_count": hand_count,
        "track_id": observation.track_id,
        "track_state": observation.state.value,
        "bbox_xywh": observation.bbox_xywh,
        "center_uv": center,
        "initialization_source": observation.initialization_source,
        "reason": observation.reason,
    }


def _center_uv(observation: TrackObservation, width: int, height: int) -> tuple[float, float] | None:
    if observation.bbox_xywh is None or width <= 0 or height <= 0:
        return None
    x, y, box_width, box_height = observation.bbox_xywh
    return ((x + box_width * 0.5) / width, (y + box_height * 0.5) / height)


def _window_is_visible(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return True


def run_camera_follow_process(
    camera_index: int,
    width: int,
    height: int,
    observations: Queue,
    commands: Queue,
    statuses: Queue,
    stop_event: Event,
    ready_event: Event,
    max_observation_age_ms: float = 150.0,
    marker_only: bool = True,
) -> None:
    """Camera UI and vision worker process for the Mac MuJoCo follow demo."""
    window_name = "UR5 camera follow - manual object selection"
    tracker = CSRTObjectTracker(max_missed_frames=15)
    latest_id = 0
    follow_requested = False
    marker_active = False
    deferred_key = -1
    displayed_status = "Waiting for target; press s to select"
    parent_status = "MuJoCo simulation is starting"
    fps_start = time.perf_counter()
    fps_frames = 0
    displayed_fps = 0.0

    try:
        with LatestFrameCamera(camera_index, width, height) as camera:
            with MediaPipeHandDetector() as detector:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, width, height)
                _put_command(commands, {"type": "ready"})
                ready_event.set()
                last_track = TrackObservation(
                    frame_id=0,
                    timestamp_ns=time.monotonic_ns(),
                    track_id=None,
                    state=TrackState.SEARCHING,
                    bbox_xywh=None,
                    initialization_source=None,
                    reason="waiting for camera frame",
                )
                last_packet = None

                while not stop_event.is_set():
                    while True:
                        try:
                            parent_message = statuses.get_nowait()
                        except queue.Empty:
                            break
                        parent_status = str(parent_message.get("status", ""))
                        if parent_status:
                            displayed_status = parent_status
                            if parent_status.startswith(("Selection received", "Selection rejected", "Follow stopped")):
                                marker_active = False
                            elif parent_status.startswith(("Object marker active", "Tracking paused", "Following track")):
                                marker_active = True
                        if parent_message.get("follow_enabled") is False:
                            follow_requested = False

                    packet = camera.latest(latest_id)
                    if packet is None:
                        if camera.error:
                            raise RuntimeError(camera.error)
                        if last_packet is not None:
                            view = last_packet["view"].copy()
                            age_ms = max(0.0, (time.monotonic_ns() - last_packet["timestamp_ns"]) / 1e6)
                            cv2.putText(
                                view,
                                f"{parent_status} | observation age {age_ms:.0f} ms",
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.imshow(window_name, view)
                            key = cv2.waitKey(1) & 0xFF
                            if key in (ord("q"), 27):
                                _put_command(commands, {"type": "shutdown", "reason": "camera window closed"})
                                break
                            if not _window_is_visible(window_name):
                                _put_command(commands, {"type": "shutdown", "reason": "camera window closed"})
                                break
                            if key in (ord("s"), ord(" ")):
                                deferred_key = key
                        time.sleep(0.002)
                        continue

                    latest_id = packet.frame_id
                    hands, _ = detector.detect(packet.bgr, packet.captured_monotonic_ns)
                    last_track = tracker.update(packet.bgr, packet.frame_id, packet.captured_monotonic_ns)
                    image_height, image_width = packet.bgr.shape[:2]
                    now_ns = time.monotonic_ns()
                    age_ms = max(0.0, (now_ns - packet.captured_monotonic_ns) / 1e6)
                    tracking_lost = last_track.state in (TrackState.SEARCHING, TrackState.LOST)
                    observation_expired = age_ms > max_observation_age_ms + 750.0
                    if marker_active and (tracking_lost or observation_expired):
                        marker_active = False

                    if follow_requested and (
                        tracking_lost
                        or observation_expired
                    ):
                        follow_requested = False
                        displayed_status = "Follow paused: tracking lost or camera frame stale"
                        _put_command(commands, {"type": "disarm", "reason": displayed_status})

                    message = observation_message(
                        packet.frame_id,
                        packet.captured_monotonic_ns,
                        image_width,
                        image_height,
                        len(hands),
                        last_track,
                    )
                    _put_latest(observations, message)
                    fps_frames += 1
                    elapsed = time.perf_counter() - fps_start
                    if elapsed >= 1.0:
                        displayed_fps = fps_frames / elapsed
                        fps_frames = 0
                        fps_start = time.perf_counter()

                    view = draw_hand_observations(packet.bgr, hands)
                    if last_track.state == TrackState.TRACKING and last_track.bbox_xywh:
                        x, y, box_width, box_height = last_track.bbox_xywh
                        cv2.rectangle(view, (x, y), (x + box_width, y + box_height), (0, 165, 255), 2)
                        cv2.putText(
                            view,
                            f"manual track {last_track.track_id}",
                            (x, min(image_height - 5, y + box_height + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    if marker_only:
                        follow_label = "OBJECT MARKER ACTIVE | ARM HELD" if marker_active else "ARM HELD"
                    else:
                        follow_label = "ARM FOLLOWING" if follow_requested else (
                            "OBJECT MARKER ACTIVE | ARM HELD" if marker_active else "ARM HELD"
                        )
                    cv2.putText(
                        view,
                        f"hands {len(hands)} | {last_track.state.value} | {follow_label} | {displayed_fps:.1f} FPS | age {age_ms:.0f} ms",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        view,
                        displayed_status,
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        view,
                        (
                            f"{parent_status} | s: select object | q: quit"
                            if marker_only
                            else f"{parent_status} | s: select object | space: start/pause arm | q: quit"
                        ),
                        (10, image_height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, view)
                    last_packet = {
                        "bgr": packet.bgr,
                        "view": view,
                        "frame_id": packet.frame_id,
                        "timestamp_ns": packet.captured_monotonic_ns,
                        "hands": hands,
                        "observation": last_track,
                    }

                    key = cv2.waitKey(1) & 0xFF
                    if key not in (ord("q"), 27) and deferred_key != -1:
                        key = deferred_key
                    deferred_key = -1
                    if key in (ord("q"), 27) or not _window_is_visible(window_name):
                        _put_command(commands, {"type": "shutdown", "reason": "camera window closed"})
                        break
                    if key == ord("s"):
                        follow_requested = False
                        marker_active = False
                        _put_command(commands, {"type": "deselect", "reason": "new manual selection"})
                        if len(hands) != 1:
                            displayed_status = "Selection needs exactly one visible hand"
                            continue
                        displayed_status = "Select one object box; MuJoCo remains open"
                        cv2.putText(
                            view,
                            "Drag a box around the object, then press ENTER or SPACE",
                            (10, 72),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 220, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(window_name, view)
                        roi = cv2.selectROI(window_name, view, fromCenter=False, showCrosshair=True)
                        x, y, box_width, box_height = (int(value) for value in roi)
                        if box_width <= 0 or box_height <= 0:
                            displayed_status = "Selection cancelled; follow remains paused"
                            continue
                        acquired = tracker.acquire(
                            packet.bgr,
                            (x, y, box_width, box_height),
                            packet.frame_id,
                            packet.captured_monotonic_ns,
                            "manual",
                        )
                        last_track = acquired
                        displayed_status = (
                            "Selection sent; waiting for MuJoCo to confirm the fresh track"
                        )
                        marker_active = False
                        selected_message = observation_message(
                            packet.frame_id,
                            packet.captured_monotonic_ns,
                            image_width,
                            image_height,
                            len(hands),
                            acquired,
                        )
                        _put_latest(observations, selected_message)
                        _put_command(commands, {"type": "select", "observation": selected_message})
                    if key == ord(" "):
                        if marker_only:
                            displayed_status = "Object marker follows the selected item; arm remains still"
                        elif follow_requested:
                            follow_requested = False
                            displayed_status = "Follow paused by operator"
                            _put_command(commands, {"type": "disarm", "reason": displayed_status})
                        else:
                            center_uv = _center_uv(last_track, image_width, image_height)
                            current_age_ms = max(
                                0.0, (time.monotonic_ns() - packet.captured_monotonic_ns) / 1e6
                            )
                            if (
                                last_track.state == TrackState.TRACKING
                                and last_track.bbox_xywh is not None
                                and center_uv is not None
                                and current_age_ms <= max_observation_age_ms
                            ):
                                arm_message = dict(message)
                                arm_message["center_uv"] = center_uv
                                _put_command(commands, {"type": "arm", "observation": arm_message})
                                follow_requested = True
                                displayed_status = f"Requesting follow for track {last_track.track_id}"
                            else:
                                displayed_status = "Cannot follow: select a fresh tracked object first"

                if stop_event.is_set():
                    _put_command(commands, {"type": "shutdown", "reason": "simulation stopped"})
                _put_command(commands, {"type": "disarm", "reason": "camera window closed"})
                _put_latest(observations, {"type": "closed", "timestamp_ns": time.monotonic_ns()})
                cv2.destroyAllWindows()
    except BaseException as exc:
        ready_event.set()
        _put_command(commands, {"type": "error", "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        ready_event.set()
        cv2.destroyAllWindows()
