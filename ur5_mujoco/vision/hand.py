from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"


@dataclass(frozen=True)
class HandObservation:
    """One hand's 2D landmarks and padded pixel search region."""

    hand_index: int
    label: str
    handedness_score: float
    bbox_xywh: tuple[int, int, int, int]
    landmarks_uv: tuple[tuple[float, float], ...]


def observations_from_result(
    result: Any,
    image_width: int,
    image_height: int,
    padding_fraction: float = 0.35,
) -> list[HandObservation]:
    """Convert MediaPipe normalized landmarks into padded image pixel boxes."""
    observations: list[HandObservation] = []
    hands = getattr(result, "hand_landmarks", ()) or ()
    handedness = getattr(result, "handedness", ()) or ()

    for index, landmarks in enumerate(hands):
        if not landmarks:
            continue
        xs = np.asarray([float(point.x) for point in landmarks], dtype=float)
        ys = np.asarray([float(point.y) for point in landmarks], dtype=float)
        x0 = float(np.clip(xs.min(), 0.0, 1.0)) * image_width
        x1 = float(np.clip(xs.max(), 0.0, 1.0)) * image_width
        y0 = float(np.clip(ys.min(), 0.0, 1.0)) * image_height
        y1 = float(np.clip(ys.max(), 0.0, 1.0)) * image_height
        pad_x = max(8.0, (x1 - x0) * padding_fraction)
        pad_y = max(8.0, (y1 - y0) * padding_fraction)
        left = max(0, int(np.floor(x0 - pad_x)))
        top = max(0, int(np.floor(y0 - pad_y)))
        right = min(image_width, int(np.ceil(x1 + pad_x)))
        bottom = min(image_height, int(np.ceil(y1 + pad_y)))

        label = "Unknown"
        score = 0.0
        if index < len(handedness) and handedness[index]:
            category = handedness[index][0]
            label = str(getattr(category, "category_name", "Unknown"))
            score = float(getattr(category, "score", 0.0) or 0.0)

        uv = tuple(
            (
                float(np.clip(point.x, 0.0, 1.0)) * image_width,
                float(np.clip(point.y, 0.0, 1.0)) * image_height,
            )
            for point in landmarks
        )
        observations.append(
            HandObservation(
                hand_index=index,
                label=label,
                handedness_score=score,
                bbox_xywh=(left, top, max(0, right - left), max(0, bottom - top)),
                landmarks_uv=uv,
            )
        )
    return observations


class MediaPipeHandDetector:
    """Stateful two-hand detector for ordered live camera frames."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand model not found: {model_path}. Run "
                "'.venv/bin/python tools/download_hand_landmarker.py' first."
            )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(
        self,
        bgr_frame: np.ndarray,
        captured_monotonic_ns: int | None = None,
    ) -> tuple[list[HandObservation], float]:
        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError("camera frame must be a 3-channel BGR image")
        started = time.perf_counter()
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ns = captured_monotonic_ns or time.monotonic_ns()
        timestamp_ms = max(timestamp_ns // 1_000_000, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._detector.detect_for_video(image, timestamp_ms)
        height, width = bgr_frame.shape[:2]
        observations = observations_from_result(result, width, height)
        return observations, (time.perf_counter() - started) * 1000.0

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> MediaPipeHandDetector:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def draw_hand_observations(frame: np.ndarray, hands: list[HandObservation]) -> np.ndarray:
    """Draw landmarks and expanded search regions onto a camera frame."""
    output = frame.copy()
    for hand in hands:
        x, y, width, height = hand.bbox_xywh
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 220, 0), 2)
        cv2.putText(
            output,
            f"{hand.label} {hand.handedness_score:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
        points = [(int(round(u)), int(round(v))) for u, v in hand.landmarks_uv]
        for start, end in HAND_CONNECTIONS:
            cv2.line(output, points[start], points[end], (40, 210, 255), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(output, point, 3, (255, 90, 40), -1, cv2.LINE_AA)
    return output
