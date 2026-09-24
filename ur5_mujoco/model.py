from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets/ur5/ur5_2f85.xml"

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_ACTUATOR_NAMES = (
    "shoulder_pan_servo",
    "shoulder_lift_servo",
    "elbow_servo",
    "wrist_1_servo",
    "wrist_2_servo",
    "wrist_3_servo",
)
# The gripper points horizontally along -Y; its jaw axis is horizontal along -X.
HOME_Q = np.array([-1.811664, -1.407528, 1.714162, 2.834958, -1.329929, 0.0], dtype=float)


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise ValueError(f"MuJoCo model is missing {object_type.name} '{name}'")
    return result


@dataclass(frozen=True)
class RobotBindings:
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    actuator_ids: np.ndarray
    joint_ranges: np.ndarray
    tcp_site_id: int
    flange_site_id: int
    target_body_id: int
    target_mocap_id: int
    object_marker_mocap_id: int
    gripper_actuator_id: int


def load_model(path: Path | str = MODEL_PATH) -> tuple[mujoco.MjModel, mujoco.MjData, RobotBindings]:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)

    joint_ids = np.array(
        [_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES], dtype=int
    )
    actuator_ids = np.array(
        [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_ACTUATOR_NAMES], dtype=int
    )
    target_body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
    target_mocap_id = int(model.body_mocapid[target_body_id])
    if target_mocap_id < 0:
        raise ValueError("target_marker must be a mocap body")
    object_marker_body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "object_marker")
    object_marker_mocap_id = int(model.body_mocapid[object_marker_body_id])
    if object_marker_mocap_id < 0:
        raise ValueError("object_marker must be a mocap body")

    bindings = RobotBindings(
        joint_ids=joint_ids,
        qpos_addresses=model.jnt_qposadr[joint_ids].copy(),
        dof_addresses=model.jnt_dofadr[joint_ids].copy(),
        actuator_ids=actuator_ids,
        joint_ranges=model.jnt_range[joint_ids].copy(),
        tcp_site_id=_id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp"),
        flange_site_id=_id(model, mujoco.mjtObj.mjOBJ_SITE, "flange"),
        target_body_id=target_body_id,
        target_mocap_id=target_mocap_id,
        object_marker_mocap_id=object_marker_mocap_id,
        gripper_actuator_id=_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"),
    )
    if model.nv < 14 or model.nu != 7:
        raise ValueError(f"Unexpected UR5 + 2F-85 dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    return model, data, bindings


def arm_positions(data: mujoco.MjData, bindings: RobotBindings) -> np.ndarray:
    return data.qpos[bindings.qpos_addresses].copy()


def initialize_arm(data: mujoco.MjData, bindings: RobotBindings, q: np.ndarray = HOME_Q) -> None:
    q = np.asarray(q, dtype=float)
    if q.shape != (6,):
        raise ValueError("UR5 joint configuration must contain six values")
    data.qpos[bindings.qpos_addresses] = q
    data.ctrl[bindings.actuator_ids] = q
