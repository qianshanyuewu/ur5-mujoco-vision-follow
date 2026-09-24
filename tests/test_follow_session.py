from __future__ import annotations

import queue
import unittest

import numpy as np

from ur5_mujoco.follow_session import (
    ARM_FOLLOW_CENTER_HISTORY_FRAMES,
    CameraFollowTargetSource,
)


class CameraFollowStillnessTests(unittest.TestCase):
    def test_arm_returns_to_stationary_after_target_stops(self) -> None:
        source = CameraFollowTargetSource(
            observations=queue.Queue(),
            commands=queue.Queue(),
            statuses=queue.Queue(),
            marker_only=False,
        )
        initial_center = np.array([0.5, 0.5])
        source._arm_center_history.extend(
            initial_center.copy() for _ in range(ARM_FOLLOW_CENTER_HISTORY_FRAMES)
        )
        source._arm_filtered_center = initial_center.copy()
        source._arm_output_center = initial_center.copy()
        source._arm_stationary_center = initial_center.copy()

        def observe(frame_id: int, timestamp_ns: int, center_x: float) -> None:
            source._arm_follow_center(
                {
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "center_uv": (center_x, 0.5),
                    "image_width": 640,
                    "image_height": 480,
                }
            )

        # The target moves far enough to resume following after the 3-frame median.
        for frame_id in range(3):
            observe(frame_id, frame_id * 66_000_000, 0.53)
        self.assertFalse(source._arm_stationary)

        # It then stays still. With 66 ms frame intervals, pruning the old history
        # used to keep its measured age below 400 ms forever.
        for frame_id in range(3, 11):
            observe(frame_id, frame_id * 66_000_000, 0.53)

        self.assertTrue(source._arm_stationary)
        np.testing.assert_allclose(source._arm_stationary_center, [0.53, 0.5])


if __name__ == "__main__":
    unittest.main()
