from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LineTrajectory:
    start: np.ndarray
    amplitude: float = 0.04
    speed: float = 0.02

    def __post_init__(self) -> None:
        if self.amplitude <= 0 or self.speed <= 0:
            raise ValueError("amplitude and speed must be positive")

    def position(self, time_s: float) -> np.ndarray:
        point = np.asarray(self.start, dtype=float).copy()
        point[0] += self.amplitude * np.sin((self.speed / self.amplitude) * time_s)
        return point


@dataclass(frozen=True)
class CircleTrajectory:
    start: np.ndarray
    radius: float = 0.04
    speed: float = 0.02

    def __post_init__(self) -> None:
        if self.radius <= 0 or self.speed <= 0:
            raise ValueError("radius and speed must be positive")

    def position(self, time_s: float) -> np.ndarray:
        start = np.asarray(self.start, dtype=float)
        angle = (self.speed / self.radius) * time_s - np.pi / 2
        center = start + np.array([0.0, self.radius, 0.0])
        return center + self.radius * np.array([np.cos(angle), np.sin(angle), 0.0])

