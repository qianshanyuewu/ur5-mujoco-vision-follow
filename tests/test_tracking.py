from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from ur5_mujoco.controller import ControllerConfig
from ur5_mujoco.model import ARM_ACTUATOR_NAMES, ARM_JOINT_NAMES, HOME_Q, load_model
from ur5_mujoco.simulation import SimulationConfig, TrackingSimulation
from ur5_mujoco.trajectories import CircleTrajectory, LineTrajectory


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _transform(position: tuple[float, float, float], rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def _official_ur5_flange_pose(q: np.ndarray) -> np.ndarray:
    """Forward-kinematics reference using the official UR5 xacro transforms."""
    from math import pi

    transform = _transform((0, 0, 0), _rpy(0, 0, pi))  # base_link_inertia frame
    joints = (
        ((0, 0, 0.089159), _rpy(0, 0, 0)),
        ((0, 0, 0), _rpy(pi / 2, 0, 0)),
        ((-0.425, 0, 0), _rpy(0, 0, 0)),
        ((-0.39225, 0, 0.10915), _rpy(0, 0, 0)),
        ((0, -0.09465, 0), _rpy(pi / 2, 0, 0)),
        ((0, 0.0823, 0), _rpy(pi / 2, pi, pi)),
    )
    for (position, rotation), angle in zip(joints, q, strict=True):
        transform = transform @ _transform(position, rotation) @ _transform((0, 0, 0), _rpy(0, 0, angle))
    return transform @ _transform((0, 0, 0), _rpy(0, -pi / 2, -pi / 2))


class ModelTests(unittest.TestCase):
    def test_model_loads_arm_gripper_and_tcp(self) -> None:
        model, _, bindings = load_model()
        self.assertEqual((model.nq, model.nv, model.nu), (14, 14, 7))
        self.assertEqual(len(ARM_JOINT_NAMES), 6)
        self.assertEqual(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, bindings.tcp_site_id), "tcp")
        pinch_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.assertGreaterEqual(pinch_site_id, 0)
        np.testing.assert_allclose(
            model.site_pos[bindings.tcp_site_id], model.site_pos[pinch_site_id], atol=1e-12
        )
        self.assertEqual(len(ARM_ACTUATOR_NAMES), 6)

    def test_arm_joints_are_independently_controllable(self) -> None:
        model, data, bindings = load_model()
        data.qpos[bindings.qpos_addresses] = HOME_Q
        mujoco.mj_forward(model, data)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, bindings.flange_site_id)
        jacobian = np.vstack((jacp[:, bindings.dof_addresses], jacr[:, bindings.dof_addresses]))
        self.assertEqual(jacobian.shape, (6, 6))
        self.assertTrue(np.all(np.linalg.norm(jacobian, axis=0) > 1e-3))
        self.assertEqual(np.linalg.matrix_rank(jacobian), 6)
        np.testing.assert_array_equal(model.actuator_trnid[bindings.actuator_ids, 0], bindings.joint_ids)

    def test_flange_forward_kinematics_matches_official_ur5_description(self) -> None:
        model, data, bindings = load_model()
        configs = (
            HOME_Q,
            np.array([0.2, -0.8, 1.1, -1.4, 0.5, -0.3]),
            np.zeros(6),
        )
        for q in configs:
            data.qpos[bindings.qpos_addresses] = q
            mujoco.mj_forward(model, data)
            expected = _official_ur5_flange_pose(q)
            actual_position = data.site_xpos[bindings.flange_site_id]
            actual_rotation = data.site_xmat[bindings.flange_site_id].reshape(3, 3)
            np.testing.assert_allclose(actual_position, expected[:3, 3], atol=1e-8)
            np.testing.assert_allclose(actual_rotation, expected[:3, :3], atol=1e-8)


class InitializationTests(unittest.TestCase):
    def test_gravity_compensation_holds_the_home_pose(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ur5-settle-") as directory:
            simulation = TrackingSimulation(
                SimulationConfig(seconds=1.0, output_path=Path(directory) / "tracking.csv"),
                lambda start: LineTrajectory(start=start),
            )
            simulation._settle()
            np.testing.assert_allclose(
                simulation.data.qpos[simulation.bindings.qpos_addresses], HOME_Q, atol=1e-3
            )
            self.assertLess(
                np.max(np.abs(simulation.data.qvel[simulation.bindings.dof_addresses])), 0.01
            )


class TrajectoryTests(unittest.TestCase):
    def _run_trajectory(self, kind: str) -> dict[str, float | str]:
        with tempfile.TemporaryDirectory(prefix=f"ur5-{kind}-") as directory:
            output = Path(directory) / "tracking.csv"

            def factory(start: np.ndarray):
                if kind == "line":
                    return LineTrajectory(start=start, amplitude=0.04, speed=0.02)
                return CircleTrajectory(start=start, radius=0.04, speed=0.02)

            simulation = TrackingSimulation(
                SimulationConfig(seconds=60.0, output_path=output, gripper_command=255.0),
                factory,
                ControllerConfig(),
            )
            result = simulation.run()
            self.assertTrue(output.exists())
            self.assertGreater(result["samples"], 5000)
            self.assertLessEqual(result["position_rms_m"], 0.010)
            self.assertLessEqual(result["position_peak_m"], 0.025)
            self.assertLessEqual(result["max_joint_speed_rad_s"], 0.300001)
            self.assertLessEqual(result["max_joint_command_rad_s"], 0.300001)
            return result

    def test_60_second_line_tracking(self) -> None:
        self._run_trajectory("line")

    def test_60_second_circle_tracking(self) -> None:
        self._run_trajectory("circle")


if __name__ == "__main__":
    unittest.main()
