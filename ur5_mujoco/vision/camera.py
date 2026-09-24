from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .hand import MediaPipeHandDetector, draw_hand_observations
from .tracking import CSRTObjectTracker, TrackState


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: int
    captured_monotonic_ns: int
    bgr: np.ndarray


class LatestFrameCamera:
    """Continuously capture frames while keeping only the newest one."""

    def __init__(self, camera_index: int, width: int, height: int) -> None:
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame: CapturedFrame | None = None
        self._error: str | None = None
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open camera {self._camera_index}. Check macOS Camera permission "
                "for the terminal running this command."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap = cap
        self._thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        assert self._cap is not None
        frame_id = 0
        consecutive_failures = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            captured_ns = time.monotonic_ns()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    self._error = "camera stopped returning frames"
                    self._stop.set()
                    break
                time.sleep(0.005)
                continue
            consecutive_failures = 0
            frame_id += 1
            packet = CapturedFrame(frame_id, captured_ns, frame)
            with self._lock:
                self._frame = packet

    def latest(self, after_frame_id: int = 0) -> CapturedFrame | None:
        with self._lock:
            packet = self._frame
            if packet is None or packet.frame_id <= after_frame_id:
                return None
            return CapturedFrame(packet.frame_id, packet.captured_monotonic_ns, packet.bgr.copy())

    @property
    def error(self) -> str | None:
        return self._error

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def __enter__(self) -> LatestFrameCamera:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def run_camera(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    max_frames: int | None = None,
    display: bool = True,
) -> dict[str, float | int]:
    processed = 0
    frames_with_hands = 0
    latest_id = 0
    inference_ms: list[float] = []
    elapsed_start = time.perf_counter()
    fps_start = elapsed_start
    fps_frames = 0
    displayed_fps = 0.0
    tracker = CSRTObjectTracker()
    selection_message = "press s to select a held object after one hand is detected"
    window_name = "UR5 phase 3 - live hand and object tracking"
    with LatestFrameCamera(camera_index, width, height) as camera:
        with MediaPipeHandDetector() as detector:
            try:
                while max_frames is None or processed < max_frames:
                    packet = camera.latest(latest_id)
                    if packet is None:
                        if camera.error:
                            raise RuntimeError(camera.error)
                        time.sleep(0.002)
                        continue
                    latest_id = packet.frame_id
                    hands, elapsed_ms = detector.detect(packet.bgr, packet.captured_monotonic_ns)
                    inference_ms.append(elapsed_ms)
                    processed += 1
                    frames_with_hands += int(bool(hands))
                    fps_frames += 1
                    track = tracker.update(packet.bgr, packet.frame_id, packet.captured_monotonic_ns)

                    now = time.perf_counter()
                    if now - fps_start >= 1.0:
                        displayed_fps = fps_frames / (now - fps_start)
                        fps_frames = 0
                        fps_start = now

                    if display:
                        view = draw_hand_observations(packet.bgr, hands)
                        if track.state == TrackState.TRACKING and track.bbox_xywh:
                            x, y, box_width, box_height = track.bbox_xywh
                            cv2.rectangle(view, (x, y), (x + box_width, y + box_height), (0, 165, 255), 2)
                            cv2.putText(
                                view,
                                f"manual track {track.track_id}",
                                (x, min(view.shape[0] - 5, y + box_height + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 165, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        object_status = f"manual seed track: {track.state.value}"
                        hand_status = f"hands: {len(hands)}" if hands else "hands: none"
                        age_ms = max(0.0, (time.monotonic_ns() - packet.captured_monotonic_ns) / 1_000_000)
                        cv2.putText(
                            view,
                            f"{hand_status} | {object_status} | {displayed_fps:.1f} FPS | age {age_ms:.0f} ms",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            view,
                            selection_message,
                            (10, view.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(window_name, view)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                        if key == ord("s"):
                            if len(hands) != 1:
                                selection_message = "manual selection needs exactly one visible hand"
                                continue
                            roi = cv2.selectROI(window_name, view, fromCenter=False, showCrosshair=True)
                            x, y, box_width, box_height = (int(value) for value in roi)
                            if box_width <= 0 or box_height <= 0:
                                selection_message = "selection cancelled"
                                continue
                            acquired = tracker.acquire(
                                packet.bgr,
                                (x, y, box_width, box_height),
                                packet.frame_id,
                                packet.captured_monotonic_ns,
                                "manual",
                            )
                            selection_message = (
                                f"tracking manually selected object as track {acquired.track_id}; press s to replace"
                            )
            finally:
                if display:
                    cv2.destroyAllWindows()
    elapsed = max(1e-9, time.perf_counter() - elapsed_start)
    return {
        "frames_processed": processed,
        "frames_with_hands": frames_with_hands,
        "elapsed_s": elapsed,
        "processed_fps": processed / elapsed,
        "inference_p50_ms": float(np.percentile(inference_ms, 50)) if inference_ms else 0.0,
        "inference_p95_ms": float(np.percentile(inference_ms, 95)) if inference_ms else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Mac camera and MediaPipe hand landmarks")
    parser.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=None, help="stop after processing this many frames")
    parser.add_argument("--no-display", action="store_true", help="run a finite/headless camera benchmark")
    args = parser.parse_args()
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    print(run_camera(args.camera, args.width, args.height, args.frames, not args.no_display))


if __name__ == "__main__":
    main()
