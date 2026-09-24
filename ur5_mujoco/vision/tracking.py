from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np


class TrackState(StrEnum):
    SEARCHING = "searching"
    TRACKING = "tracking"
    PAUSED = "paused"
    LOST = "lost"


@dataclass(frozen=True)
class TrackObservation:
    frame_id: int
    timestamp_ns: int
    track_id: int | None
    state: TrackState
    bbox_xywh: tuple[int, int, int, int] | None
    initialization_source: str | None
    reason: str


def _tracker_csrt():
    factory = getattr(cv2, "TrackerCSRT_create", None)
    if factory is None:
        factory = getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None)
    if factory is None:
        raise RuntimeError("OpenCV CSRT tracker is unavailable; install opencv-contrib-python")
    return factory()


class CSRTObjectTracker:
    """Track one object after an explicit bounding-box seed.

    CSRT does not discover objects or infer hand-object contact. Call acquire()
    only with a box from a trusted detector or an explicit user selection.
    """

    def __init__(self, max_missed_frames: int = 3) -> None:
        if max_missed_frames < 1:
            raise ValueError("max_missed_frames must be positive")
        self.max_missed_frames = max_missed_frames
        self._tracker = None
        self._track_id: int | None = None
        self._next_track_id = 1
        self._initialization_source: str | None = None
        self._frame_shape: tuple[int, int] | None = None
        self._bbox: tuple[int, int, int, int] | None = None
        self._missed_frames = 0
        self._state = TrackState.SEARCHING

    @property
    def state(self) -> TrackState:
        return self._state

    def acquire(
        self,
        frame: np.ndarray,
        bbox_xywh: tuple[int, int, int, int],
        frame_id: int,
        timestamp_ns: int,
        initialization_source: str,
    ) -> TrackObservation:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a 3-channel image")
        if not initialization_source.strip():
            raise ValueError("initialization_source must be provided")
        x, y, width, height = (int(value) for value in bbox_xywh)
        image_height, image_width = frame.shape[:2]
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > image_width or y + height > image_height:
            raise ValueError("bounding box must be positive and fully inside the frame")

        tracker = _tracker_csrt()
        initialized = tracker.init(frame, (x, y, width, height))
        if initialized is False:
            raise RuntimeError("OpenCV CSRT tracker failed to initialize")

        self._tracker = tracker
        self._track_id = self._next_track_id
        self._next_track_id += 1
        self._initialization_source = initialization_source
        self._frame_shape = (image_height, image_width)
        self._bbox = (x, y, width, height)
        self._missed_frames = 0
        self._state = TrackState.TRACKING
        return self._observation(frame_id, timestamp_ns, self._bbox, "target acquired")

    def update(self, frame: np.ndarray, frame_id: int, timestamp_ns: int) -> TrackObservation:
        if self._tracker is None or self._track_id is None:
            self._state = TrackState.SEARCHING
            return self._observation(frame_id, timestamp_ns, None, "no target initialized")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a 3-channel image")
        if frame.shape[:2] != self._frame_shape:
            return self._miss(frame_id, timestamp_ns, "camera frame size changed")

        ok, box = self._tracker.update(frame)
        if ok:
            x, y, width, height = box
            image_height, image_width = frame.shape[:2]
            left, top = max(0, int(round(x))), max(0, int(round(y)))
            right = min(image_width, int(round(x + width)))
            bottom = min(image_height, int(round(y + height)))
            if right > left and bottom > top:
                self._bbox = (left, top, right - left, bottom - top)
                self._missed_frames = 0
                self._state = TrackState.TRACKING
                return self._observation(frame_id, timestamp_ns, self._bbox, "CSRT update")
        return self._miss(frame_id, timestamp_ns, "tracker update failed")

    def reset(self) -> None:
        self._tracker = None
        self._track_id = None
        self._initialization_source = None
        self._frame_shape = None
        self._bbox = None
        self._missed_frames = 0
        self._state = TrackState.SEARCHING

    def _miss(self, frame_id: int, timestamp_ns: int, reason: str) -> TrackObservation:
        self._missed_frames += 1
        self._state = (
            TrackState.PAUSED if self._missed_frames <= self.max_missed_frames else TrackState.LOST
        )
        if self._state == TrackState.LOST:
            lost_track_id = self._track_id
            lost_source = self._initialization_source
            self._tracker = None
            self._bbox = None
            self._track_id = None
            self._initialization_source = None
            return TrackObservation(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                track_id=lost_track_id,
                state=self._state,
                bbox_xywh=None,
                initialization_source=lost_source,
                reason=reason,
            )
        return self._observation(frame_id, timestamp_ns, None, reason)

    def _observation(
        self,
        frame_id: int,
        timestamp_ns: int,
        bbox: tuple[int, int, int, int] | None,
        reason: str,
    ) -> TrackObservation:
        return TrackObservation(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            track_id=self._track_id,
            state=self._state,
            bbox_xywh=bbox,
            initialization_source=self._initialization_source,
            reason=reason,
        )
