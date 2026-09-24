from __future__ import annotations

import unittest
from types import SimpleNamespace

from ur5_mujoco.vision.hand import observations_from_result


class HandObservationTests(unittest.TestCase):
    def test_landmarks_become_clamped_padded_pixel_box(self) -> None:
        points = [SimpleNamespace(x=0.4, y=0.3) for _ in range(21)]
        points[0] = SimpleNamespace(x=0.0, y=0.0)
        points[20] = SimpleNamespace(x=1.0, y=1.0)
        result = SimpleNamespace(
            hand_landmarks=[points],
            handedness=[[SimpleNamespace(category_name="Right", score=0.91)]],
        )
        hand = observations_from_result(result, 640, 480)[0]
        self.assertEqual(hand.bbox_xywh, (0, 0, 640, 480))
        self.assertEqual(hand.label, "Right")
        self.assertAlmostEqual(hand.handedness_score, 0.91)
        self.assertEqual(len(hand.landmarks_uv), 21)

    def test_missing_hands_produce_no_observations(self) -> None:
        result = SimpleNamespace(hand_landmarks=[], handedness=[])
        self.assertEqual(observations_from_result(result, 640, 480), [])

    def test_each_hand_is_returned_even_when_handedness_is_missing(self) -> None:
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        result = SimpleNamespace(hand_landmarks=[points, points], handedness=[])
        hands = observations_from_result(result, 640, 480)
        self.assertEqual(len(hands), 2)
        self.assertEqual([hand.hand_index for hand in hands], [0, 1])
        self.assertTrue(all(hand.label == "Unknown" for hand in hands))


if __name__ == "__main__":
    unittest.main()
