from __future__ import annotations

import unittest

import numpy as np

from ur5_mujoco.vision.tracking import CSRTObjectTracker, TrackState


def _frame(x: int) -> np.ndarray:
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    image[40:70, x : x + 30] = (40, 210, 250)
    return image


class CSRTObjectTrackerTests(unittest.TestCase):
    def test_tracks_a_moving_seeded_object(self) -> None:
        tracker = CSRTObjectTracker()
        initial = tracker.acquire(_frame(20), (20, 40, 30, 30), 1, 1_000_000, "manual")
        self.assertEqual(initial.state, TrackState.TRACKING)
        self.assertEqual(initial.initialization_source, "manual")

        observations = [
            tracker.update(_frame(x), frame_id, frame_id * 1_000_000)
            for frame_id, x in enumerate(range(23, 54, 3), start=2)
        ]
        self.assertTrue(all(observation.state == TrackState.TRACKING for observation in observations))
        self.assertTrue(all(observation.track_id == initial.track_id for observation in observations))
        self.assertTrue(all(observation.bbox_xywh is not None for observation in observations))
        final_x = observations[-1].bbox_xywh[0]
        self.assertAlmostEqual(final_x, 50, delta=8)

    def test_invalid_seed_is_rejected(self) -> None:
        tracker = CSRTObjectTracker()
        with self.assertRaises(ValueError):
            tracker.acquire(_frame(20), (190, 100, 30, 30), 1, 1, "manual")

    def test_frame_size_change_pauses_then_loses_track(self) -> None:
        tracker = CSRTObjectTracker(max_missed_frames=2)
        initial = tracker.acquire(_frame(20), (20, 40, 30, 30), 1, 1, "manual")
        changed_size = np.zeros((100, 160, 3), dtype=np.uint8)
        first = tracker.update(changed_size, 2, 2)
        second = tracker.update(changed_size, 3, 3)
        third = tracker.update(changed_size, 4, 4)
        self.assertEqual(first.state, TrackState.PAUSED)
        self.assertEqual(second.state, TrackState.PAUSED)
        self.assertEqual(third.state, TrackState.LOST)
        self.assertEqual(third.track_id, initial.track_id)
        self.assertIsNone(third.bbox_xywh)


if __name__ == "__main__":
    unittest.main()
