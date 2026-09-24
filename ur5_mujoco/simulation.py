from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import mujoco
import numpy as np

from .controller import ControllerConfig, DampedJacobianController, rotation_error
from .model import HOME_Q, PROJECT_ROOT, RobotBindings, arm_positions, load_model


@dataclass(frozen=True)
class SimulationConfig:
    physics_dt: float = 0.002
    control_dt: float = 0.01
    settle_seconds: float = 0.75
    seconds: float = 60.0
    gripper_command: float = 255.0
    output_path: Path | None = None
    viewer: bool = False
    realtime: bool = True


@dataclass(frozen=True)
class LiveTargetSample:
    target_position: np.ndarray
    metadata: dict[str, object]
    quit: bool = False
    marker_position: np.ndarray | None = None


class LiveTargetSource(Protocol):
    def reset(self, initial_position: np.ndarray, initial_rotation: np.ndarray | None = None) -> None: ...

    def update(self, current_position: np.ndarray, now_ns: int, dt: float) -> LiveTargetSample: ...


class TrackingSimulation:
    def __init__(
        self,
        config: SimulationConfig,
        trajectory_factory: Callable[[np.ndarray], object] | None,
        controller_config: ControllerConfig = ControllerConfig(),
    ) -> None:
        self.config = config
        self.trajectory_factory = trajectory_factory
        self.controller_config = controller_config
        self.model, self.data, self.bindings = load_model()
        if not np.isclose(self.model.opt.timestep, config.physics_dt):
            raise ValueError(
                f"Model timestep is {self.model.opt.timestep}, expected {config.physics_dt}; "
                "keep assets/ur5/ur5.xml and SimulationConfig in sync"
            )
        control_steps = config.control_dt / config.physics_dt
        if not np.isclose(control_steps, round(control_steps)):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        self.control_steps = int(round(control_steps))
        self.controller: DampedJacobianController | None = None
        self.trajectory = None
        self.target_rotation: np.ndarray | None = None

    def _settle(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.bindings.qpos_addresses] = HOME_Q
        self.data.ctrl[self.bindings.actuator_ids] = HOME_Q
        self.data.ctrl[self.bindings.gripper_actuator_id] = self.config.gripper_command
        mujoco.mj_forward(self.model, self.data)
        for _ in range(round(self.config.settle_seconds / self.config.physics_dt)):
            self._step_physics()
        mujoco.mj_forward(self.model, self.data)

        start_position = self.data.site_xpos[self.bindings.tcp_site_id].copy()
        self.target_rotation = self.data.site_xmat[self.bindings.tcp_site_id].reshape(3, 3).copy()
        self.trajectory = self.trajectory_factory(start_position) if self.trajectory_factory else None
        self.controller = DampedJacobianController(
            self.model, self.data, self.bindings, self.target_rotation, self.controller_config
        )
        self.data.mocap_pos[self.bindings.target_mocap_id] = start_position
        self.data.mocap_pos[self.bindings.object_marker_mocap_id] = np.array([0.0, 0.0, -10.0])

    def _step_physics(self) -> None:
        # The XML position actuators regulate joint position, but do not
        # compensate for gravity. Feed the UR5 actuators the current
        # generalized bias force so the configured home pose is actually
        # held during settling and low-speed trajectory tracking.
        dofs = self.bindings.dof_addresses
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[dofs] = self.data.qfrc_bias[dofs]
        mujoco.mj_step(self.model, self.data)

    def _output_file(self) -> Path:
        if self.config.output_path is not None:
            path = self.config.output_path
        else:
            name = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = PROJECT_ROOT / "logs" / f"tracking-{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def run(self) -> dict[str, float | str]:
        self._settle()
        assert self.trajectory is not None and self.controller is not None
        output = self._output_file()
        total_steps = round(self.config.seconds / self.config.physics_dt)
        if total_steps <= 0:
            raise ValueError("seconds must be positive")

        errors: list[float] = []
        max_joint_speed = 0.0
        max_joint_command = 0.0
        qpos_names = [f"q{i + 1}" for i in range(6)]
        qvel_names = [f"qd{i + 1}" for i in range(6)]
        columns = [
            "time_s",
            "target_x_m",
            "target_y_m",
            "target_z_m",
            "tcp_x_m",
            "tcp_y_m",
            "tcp_z_m",
            "position_error_m",
            "orientation_error_rad",
            *qpos_names,
            *qvel_names,
            "gripper_command",
        ]

        def record(writer: csv.writer, time_s: float, target: np.ndarray) -> None:
            site_id = self.bindings.tcp_site_id
            tcp_position = self.data.site_xpos[site_id].copy()
            tcp_rotation = self.data.site_xmat[site_id].reshape(3, 3).copy()
            position_error = float(np.linalg.norm(target - tcp_position))
            orientation_error = float(np.linalg.norm(rotation_error(self.target_rotation, tcp_rotation)))
            q = arm_positions(self.data, self.bindings)
            qd = self.data.qvel[self.bindings.dof_addresses].copy()
            errors.append(position_error)
            writer.writerow(
                [
                    f"{time_s:.6f}",
                    *[f"{x:.8f}" for x in target],
                    *[f"{x:.8f}" for x in tcp_position],
                    f"{position_error:.8f}",
                    f"{orientation_error:.8f}",
                    *[f"{x:.8f}" for x in q],
                    *[f"{x:.8f}" for x in qd],
                    f"{self.config.gripper_command:.2f}",
                ]
            )

        def run_loop(writer: csv.writer, start_wall: float, viewer=None) -> None:
            nonlocal max_joint_speed, max_joint_command
            for physics_step in range(total_steps):
                if physics_step % self.control_steps == 0:
                    time_s = physics_step * self.config.physics_dt
                    target = np.asarray(self.trajectory.position(time_s), dtype=float)
                    q_reference = self.controller.update(self.data, target, self.config.control_dt)
                    self.data.ctrl[self.bindings.actuator_ids] = q_reference
                    self.data.ctrl[self.bindings.gripper_actuator_id] = self.config.gripper_command
                    self.data.mocap_pos[self.bindings.target_mocap_id] = target
                    max_joint_command = max(max_joint_command, float(np.max(np.abs(self.controller.previous_qdot))))
                    record(writer, time_s, target)

                self._step_physics()
                max_joint_speed = max(
                    max_joint_speed,
                    float(np.max(np.abs(self.data.qvel[self.bindings.dof_addresses]))),
                )
                if viewer is not None and (physics_step + 1) % self.control_steps == 0:
                    viewer.sync()
                    if self.config.realtime:
                        expected = (physics_step + 1) * self.config.physics_dt
                        delay = expected - (time.perf_counter() - start_wall)
                        if delay > 0:
                            time.sleep(delay)
                    if not viewer.is_running():
                        break

        start_wall = time.perf_counter()
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            if self.config.viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    run_loop(writer, start_wall, viewer)
            else:
                run_loop(writer, start_wall)

        if not errors:
            raise RuntimeError("simulation ended before recording any tracking samples")
        error_array = np.asarray(errors)
        result: dict[str, float | str] = {
            "samples": float(len(errors)),
            "position_rms_m": float(np.sqrt(np.mean(error_array**2))),
            "position_peak_m": float(np.max(error_array)),
            "max_joint_speed_rad_s": max_joint_speed,
            "max_joint_command_rad_s": max_joint_command,
            "log": str(output),
        }
        return result

    def run_live(
        self,
        target_source: LiveTargetSource,
        max_seconds: float | None = None,
    ) -> dict[str, float | str]:
        """Run a real-time external-target session with the MuJoCo viewer open."""
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be positive when specified")
        if not self.config.realtime:
            raise ValueError("live external-target sessions require real-time simulation")
        self._settle()
        assert self.controller is not None
        initial_position = self.data.site_xpos[self.bindings.tcp_site_id].copy()
        initial_rotation = self.data.site_xmat[self.bindings.tcp_site_id].reshape(3, 3).copy()
        target_source.reset(initial_position, initial_rotation)
        output = self._output_file()
        position_errors: list[float] = []
        follow_errors: list[float] = []
        max_joint_speed = 0.0
        max_joint_command = 0.0
        max_observation_age_ms = 0.0
        follow_samples = 0
        mapping_samples = 0
        columns = [
            "wall_time_s",
            "target_x_m", "target_y_m", "target_z_m",
            "tcp_x_m", "tcp_y_m", "tcp_z_m",
            "position_error_m", "orientation_error_rad",
            *(f"q{i + 1}" for i in range(6)),
            *(f"qd{i + 1}" for i in range(6)),
            "vision_frame_id", "vision_timestamp_ns", "observation_age_ms",
            "track_id", "track_state", "follow_enabled", "status",
            "desired_x_m", "desired_y_m", "desired_z_m",
            "mapping_enabled", "marker_x_m", "marker_y_m", "marker_z_m",
        ]
        wall_start = time.perf_counter()
        next_tick = wall_start

        def run_loop(writer: csv.writer) -> None:
            nonlocal max_joint_speed, max_joint_command, max_observation_age_ms, follow_samples, mapping_samples
            nonlocal next_tick
            site_id = self.bindings.tcp_site_id
            self.data.mocap_pos[self.bindings.target_mocap_id] = np.array([0.0, 0.0, -10.0])
            with mujoco.viewer.launch_passive(self.model, self.data) as active_viewer:
                viewer = active_viewer
                while viewer.is_running():
                    wall_now = time.perf_counter()
                    if max_seconds is not None and wall_now - wall_start >= max_seconds:
                        break
                    now_ns = time.monotonic_ns()
                    current_position = self.data.site_xpos[site_id].copy()
                    sample = target_source.update(current_position, now_ns, self.config.control_dt)
                    if sample.quit:
                        break
                    target = np.asarray(sample.target_position, dtype=float)
                    q_reference = self.controller.update(self.data, target, self.config.control_dt)
                    self.data.ctrl[self.bindings.actuator_ids] = q_reference
                    self.data.ctrl[self.bindings.gripper_actuator_id] = self.config.gripper_command
                    if sample.marker_position is None:
                        self.data.mocap_pos[self.bindings.object_marker_mocap_id] = np.array(
                            [0.0, 0.0, -10.0]
                        )
                    else:
                        self.data.mocap_pos[self.bindings.object_marker_mocap_id] = np.asarray(
                            sample.marker_position, dtype=float
                        )
                    max_joint_command = max(
                        max_joint_command,
                        float(np.max(np.abs(self.controller.previous_qdot))),
                    )

                    for _ in range(self.control_steps):
                        self._step_physics()
                    max_joint_speed = max(
                        max_joint_speed,
                        float(np.max(np.abs(self.data.qvel[self.bindings.dof_addresses]))),
                    )

                    tcp_position = self.data.site_xpos[site_id].copy()
                    tcp_rotation = self.data.site_xmat[site_id].reshape(3, 3).copy()
                    position_error = float(np.linalg.norm(target - tcp_position))
                    orientation_error = float(
                        np.linalg.norm(rotation_error(self.target_rotation, tcp_rotation))
                    )
                    q = arm_positions(self.data, self.bindings)
                    qd = self.data.qvel[self.bindings.dof_addresses].copy()
                    metadata = sample.metadata
                    follow_enabled = bool(metadata.get("follow_enabled", False))
                    if bool(metadata.get("mapping_enabled", False)):
                        mapping_samples += 1
                    position_errors.append(position_error)
                    if follow_enabled:
                        follow_errors.append(position_error)
                        follow_samples += 1
                    observation_age = metadata.get("observation_age_ms")
                    if isinstance(observation_age, (int, float)):
                        max_observation_age_ms = max(max_observation_age_ms, float(observation_age))
                    desired = metadata.get("desired_position")
                    desired_position = (
                        np.asarray(desired, dtype=float)
                        if desired is not None
                        else np.full(3, np.nan, dtype=float)
                    )
                    marker_position = (
                        np.asarray(sample.marker_position, dtype=float)
                        if sample.marker_position is not None
                        else np.full(3, np.nan, dtype=float)
                    )
                    writer.writerow(
                        [
                            f"{time.perf_counter() - wall_start:.6f}",
                            *[f"{value:.8f}" for value in target],
                            *[f"{value:.8f}" for value in tcp_position],
                            f"{position_error:.8f}", f"{orientation_error:.8f}",
                            *[f"{value:.8f}" for value in q],
                            *[f"{value:.8f}" for value in qd],
                            metadata.get("frame_id", ""),
                            metadata.get("timestamp_ns", ""),
                            "" if observation_age is None else f"{float(observation_age):.3f}",
                            metadata.get("track_id", ""),
                            metadata.get("track_state", ""),
                            int(follow_enabled),
                            metadata.get("status", ""),
                            *["" if not np.isfinite(value) else f"{value:.8f}" for value in desired_position],
                            int(bool(metadata.get("mapping_enabled", False))),
                            *[f"{value:.8f}" for value in marker_position],
                        ]
                    )

                    viewer.sync()
                    if not viewer.is_running():
                        break
                    if self.config.realtime:
                        next_tick += self.config.control_dt
                        delay = next_tick - time.perf_counter()
                        if delay > 0:
                            time.sleep(delay)
                        elif delay < -self.config.control_dt:
                            # Do not run a burst of stale external-target updates to catch up.
                            next_tick = time.perf_counter()

        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            run_loop(writer)

        error_array = np.asarray(follow_errors if follow_errors else position_errors, dtype=float)
        return {
            "samples": float(len(position_errors)),
            "follow_samples": float(follow_samples),
            "mapping_samples": float(mapping_samples),
            "position_rms_m": float(np.sqrt(np.mean(error_array**2))) if len(error_array) else 0.0,
            "position_peak_m": float(np.max(error_array)) if len(error_array) else 0.0,
            "max_joint_speed_rad_s": max_joint_speed,
            "max_joint_command_rad_s": max_joint_command,
            "max_observation_age_ms": max_observation_age_ms,
            "log": str(output),
        }
