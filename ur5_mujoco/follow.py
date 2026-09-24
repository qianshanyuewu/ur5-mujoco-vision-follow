from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clip_vector(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > maximum > 0:
        return vector * (maximum / norm)
    return vector


@dataclass(frozen=True)
class PlanarMappingConfig:
    horizontal_axis: int = 0
    vertical_axis: int = 2
    horizontal_metres_per_normalized_unit: float = 0.10
    vertical_metres_per_normalized_unit: float = 0.08
    horizontal_limit_m: float = 0.05
    vertical_limit_m: float = 0.04

    def __post_init__(self) -> None:
        if self.horizontal_axis not in range(3) or self.vertical_axis not in range(3):
            raise ValueError("mapping axes must be 0, 1, or 2")
        if self.horizontal_axis == self.vertical_axis:
            raise ValueError("horizontal and vertical mapping axes must differ")
        if min(
            self.horizontal_metres_per_normalized_unit,
            self.vertical_metres_per_normalized_unit,
            self.horizontal_limit_m,
            self.vertical_limit_m,
        ) <= 0:
            raise ValueError("mapping scales and limits must be positive")


class PlanarTargetMapper:
    """Map a selected image point's relative motion into a bounded robot plane."""

    def __init__(self, config: PlanarMappingConfig = PlanarMappingConfig()) -> None:
        self.config = config
        self._anchor_uv: np.ndarray | None = None
        self._anchor_tcp: np.ndarray | None = None

    @property
    def anchored(self) -> bool:
        return self._anchor_uv is not None and self._anchor_tcp is not None

    def anchor(self, center_uv: tuple[float, float], tcp_position: np.ndarray) -> None:
        uv = np.asarray(center_uv, dtype=float)
        tcp = np.asarray(tcp_position, dtype=float)
        if uv.shape != (2,) or not np.all(np.isfinite(uv)) or np.any(uv < 0) or np.any(uv > 1):
            raise ValueError("image center must be a finite normalized 2-vector")
        if tcp.shape != (3,) or not np.all(np.isfinite(tcp)):
            raise ValueError("TCP position must be a finite 3-vector")
        self._anchor_uv = uv
        self._anchor_tcp = tcp.copy()

    def map_center(self, center_uv: tuple[float, float]) -> np.ndarray:
        if not self.anchored:
            raise RuntimeError("target mapper must be anchored before mapping")
        uv = np.asarray(center_uv, dtype=float)
        if uv.shape != (2,) or not np.all(np.isfinite(uv)) or np.any(uv < 0) or np.any(uv > 1):
            raise ValueError("image center must be a finite normalized 2-vector")

        assert self._anchor_uv is not None and self._anchor_tcp is not None
        delta_uv = uv - self._anchor_uv
        target = self._anchor_tcp.copy()
        h_axis = self.config.horizontal_axis
        v_axis = self.config.vertical_axis
        target[h_axis] += delta_uv[0] * self.config.horizontal_metres_per_normalized_unit
        # Image v increases downwards, while the robot's vertical axis increases upwards.
        target[v_axis] -= delta_uv[1] * self.config.vertical_metres_per_normalized_unit
        target[h_axis] = np.clip(
            target[h_axis],
            self._anchor_tcp[h_axis] - self.config.horizontal_limit_m,
            self._anchor_tcp[h_axis] + self.config.horizontal_limit_m,
        )
        target[v_axis] = np.clip(
            target[v_axis],
            self._anchor_tcp[v_axis] - self.config.vertical_limit_m,
            self._anchor_tcp[v_axis] + self.config.vertical_limit_m,
        )
        return target


@dataclass(frozen=True)
class TargetLimiterConfig:
    max_speed_m_s: float = 0.02
    max_acceleration_m_s2: float = 0.10
    position_gain_s_inv: float = 5.0
    position_deadband_m: float = 0.0

    def __post_init__(self) -> None:
        if min(self.max_speed_m_s, self.max_acceleration_m_s2, self.position_gain_s_inv) <= 0:
            raise ValueError("target limiter parameters must be positive")
        if self.position_deadband_m < 0:
            raise ValueError("target position deadband cannot be negative")


class CartesianTargetLimiter:
    """Smooth and rate-limit the Cartesian reference; None means brake to a hold."""

    def __init__(self, config: TargetLimiterConfig = TargetLimiterConfig()) -> None:
        self.config = config
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(3, dtype=float)

    def reset(self, position: np.ndarray) -> None:
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("initial target must be a finite 3-vector")
        self.position = position.copy()
        self.velocity[:] = 0.0

    def update(self, desired_position: np.ndarray | None, dt: float) -> np.ndarray:
        if self.position is None:
            raise RuntimeError("target limiter must be reset before use")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if desired_position is None:
            desired_velocity = np.zeros(3, dtype=float)
        else:
            desired = np.asarray(desired_position, dtype=float)
            if desired.shape != (3,) or not np.all(np.isfinite(desired)):
                raise ValueError("desired target must be a finite 3-vector")
            position_error = desired - self.position
            if float(np.linalg.norm(position_error)) <= self.config.position_deadband_m:
                desired_velocity = np.zeros(3, dtype=float)
            else:
                desired_velocity = _clip_vector(
                    position_error * self.config.position_gain_s_inv,
                    self.config.max_speed_m_s,
                )

        delta_velocity = desired_velocity - self.velocity
        max_delta = self.config.max_acceleration_m_s2 * dt
        self.velocity += _clip_vector(delta_velocity, max_delta)
        self.position += self.velocity * dt
        return self.position.copy()
