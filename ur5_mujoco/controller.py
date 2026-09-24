from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .model import RobotBindings, arm_positions


@dataclass(frozen=True)
class ControllerConfig:
    position_gain: float = 3.0
    orientation_gain: float = 2.0
    max_linear_speed: float = 0.05
    max_angular_speed: float = 0.6
    max_joint_speed: float = 0.3
    max_joint_acceleration: float = 1.5
    damping: float = 0.035
    singularity_threshold: float = 0.12
    joint_limit_margin: float = 0.08


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the world-frame rotation vector taking current to target."""
    error = np.asarray(target) @ np.asarray(current).T
    cosine = float(np.clip((np.trace(error) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = 0.5 * np.array(
        [error[2, 1] - error[1, 2], error[0, 2] - error[2, 0], error[1, 0] - error[0, 1]]
    )
    if angle < 1e-7:
        return vee
    if np.pi - angle < 1e-5:
        values, vectors = np.linalg.eigh((error + np.eye(3)) * 0.5)
        axis = vectors[:, int(np.argmax(values))]
        if np.dot(axis, vee) < 0:
            axis = -axis
        return angle * axis
    return (angle / np.sin(angle)) * vee


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > maximum > 0:
        return vector * (maximum / norm)
    return vector


class DampedJacobianController:
    """Position-level TCP servo using a damped least-squares Jacobian inverse."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        bindings: RobotBindings,
        target_rotation: np.ndarray,
        config: ControllerConfig = ControllerConfig(),
    ) -> None:
        self.model = model
        self.bindings = bindings
        self.target_rotation = np.asarray(target_rotation, dtype=float).reshape(3, 3).copy()
        self.config = config
        self.q_reference = arm_positions(data, bindings)
        self.previous_qdot = np.zeros(6, dtype=float)
        self._jacp = np.zeros((3, model.nv), dtype=float)
        self._jacr = np.zeros((3, model.nv), dtype=float)

    def update(
        self,
        data: mujoco.MjData,
        target_position: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        if dt <= 0:
            raise ValueError("controller dt must be positive")
        target_position = np.asarray(target_position, dtype=float)
        if target_position.shape != (3,) or not np.all(np.isfinite(target_position)):
            raise ValueError("target position must be a finite 3-vector in metres")

        site_id = self.bindings.tcp_site_id
        current_position = data.site_xpos[site_id].copy()
        current_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
        position_error = target_position - current_position
        orientation_error = rotation_error(self.target_rotation, current_rotation)

        linear_velocity = _clip_norm(self.config.position_gain * position_error, self.config.max_linear_speed)
        angular_velocity = _clip_norm(
            self.config.orientation_gain * orientation_error, self.config.max_angular_speed
        )
        twist = np.concatenate((linear_velocity, angular_velocity))

        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, site_id)
        jacobian = np.vstack(
            (
                self._jacp[:, self.bindings.dof_addresses],
                self._jacr[:, self.bindings.dof_addresses],
            )
        )
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        smallest = float(singular_values[-1])
        damping = self.config.damping
        if smallest < self.config.singularity_threshold:
            ratio = 1.0 - smallest / self.config.singularity_threshold
            damping = max(damping, self.config.damping + 0.12 * ratio * ratio)
        system = jacobian @ jacobian.T + (damping * damping) * np.eye(6)
        qdot = jacobian.T @ np.linalg.solve(system, twist)
        qdot = np.clip(qdot, -self.config.max_joint_speed, self.config.max_joint_speed)
        max_delta = self.config.max_joint_acceleration * dt
        qdot = np.clip(qdot, self.previous_qdot - max_delta, self.previous_qdot + max_delta)

        q = arm_positions(data, self.bindings)
        lower = self.bindings.joint_ranges[:, 0] + self.config.joint_limit_margin
        upper = self.bindings.joint_ranges[:, 1] - self.config.joint_limit_margin
        qdot[(q <= lower) & (qdot < 0)] = 0.0
        qdot[(q >= upper) & (qdot > 0)] = 0.0
        self.q_reference = np.clip(self.q_reference + qdot * dt, lower, upper)
        self.previous_qdot = qdot
        return self.q_reference.copy()
