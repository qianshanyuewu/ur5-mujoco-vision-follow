"""Combine the UR5 MJCF with the vendored MuJoCo Menagerie 2F-85 model."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARM_XML = ROOT / "assets/ur5/ur5.xml"
GRIPPER_XML = ROOT / "assets/robotiq_2f85/2f85.xml"
OUTPUT_XML = ROOT / "assets/ur5/ur5_2f85.xml"

# The official UR description mounts end effectors at ``tool0``. Its
# wrist_3_link -> flange transform (0, -pi/2, -pi/2), followed by the
# flange -> tool0 transform (pi/2, 0, pi/2), composes to identity. Therefore
# the Menagerie 2F-85 base_mount is attached at the wrist_3_link origin with
# no extra rotation. MuJoCo quaternions use w, x, y, z order.
TOOL0_QUAT = "1 0 0 0"


def _append_gripper_assets(arm_root: ET.Element, gripper_root: ET.Element) -> None:
    arm_assets = arm_root.find("asset")
    source_assets = gripper_root.find("asset")
    assert arm_assets is not None and source_assets is not None

    mesh_names: dict[str, str] = {}
    material_names: dict[str, str] = {}
    for original in source_assets:
        element = copy.deepcopy(original)
        if element.tag == "mesh":
            old_name = element.get("name") or Path(element.get("file", "mesh")).stem
            new_name = f"g2f85_{old_name}"
            mesh_names[old_name] = new_name
            element.set("name", new_name)
            element.set("file", f"2f85/{Path(element.get('file', '')).name}")
        elif element.tag == "material":
            old_name = element.get("name", "material")
            new_name = f"g2f85_{old_name}"
            material_names[old_name] = new_name
            element.set("name", new_name)
        arm_assets.append(element)

    def remap_references(element: ET.Element) -> None:
        if "mesh" in element.attrib:
            element.set("mesh", mesh_names[element.get("mesh", "")])
        if "material" in element.attrib:
            old_material = element.get("material", "")
            if old_material in material_names:
                element.set("material", material_names[old_material])
        for child in element:
            remap_references(child)

    # Copy class defaults as a direct child of the arm defaults. This preserves
    # the upstream millimetre-to-metre mesh scale and all gripper joint settings.
    arm_defaults = arm_root.find("default")
    gripper_defaults = gripper_root.find("default")
    assert arm_defaults is not None and gripper_defaults is not None
    for child in gripper_defaults:
        arm_defaults.append(copy.deepcopy(child))

    gripper_world = gripper_root.find("worldbody")
    assert gripper_world is not None
    gripper_base = gripper_world.find("body[@name='base_mount']")
    assert gripper_base is not None
    copied_base = copy.deepcopy(gripper_base)
    remap_references(copied_base)
    base_body = copied_base.find("body[@name='base']")
    assert base_body is not None
    pinch = base_body.find("site[@name='pinch']")
    assert pinch is not None
    tcp = copy.deepcopy(pinch)
    tcp.set("name", "tcp")
    base_body.append(tcp)

    ur5_world = arm_root.find("worldbody")
    assert ur5_world is not None
    wrist3 = ur5_world.find(".//body[@name='wrist_3_link']")
    assert wrist3 is not None
    flange_site = wrist3.find("site[@name='tcp']")
    assert flange_site is not None
    flange_site.set("name", "flange")
    adapter = ET.SubElement(wrist3, "body", {"name": "robotiq_mount", "quat": TOOL0_QUAT})
    adapter.append(copied_base)

    # Copy the contact exclusions, tendon, finger linkage constraints and
    # actuator from the unmodified upstream gripper model.
    for tag in ("contact", "tendon", "equality"):
        source = gripper_root.find(tag)
        if source is not None:
            existing = arm_root.find(tag)
            if existing is None:
                existing = ET.SubElement(arm_root, tag)
            for child in source:
                existing.append(copy.deepcopy(child))

    # Adjacent UR links overlap by a few tens of micrometres in the supplied
    # collision meshes. Letting those pairs collide adds large friction forces
    # directly across their hinge joints and can effectively lock a joint.
    contact = arm_root.find("contact")
    if contact is None:
        contact = ET.SubElement(arm_root, "contact")
    for body1, body2 in (
        ("base_link_inertia", "shoulder_link"),
        ("shoulder_link", "upper_arm_link"),
        ("upper_arm_link", "forearm_link"),
        ("forearm_link", "wrist_1_link"),
        ("wrist_1_link", "wrist_2_link"),
        ("wrist_2_link", "wrist_3_link"),
        ("wrist_3_link", "robotiq_mount"),
        ("robotiq_mount", "base_mount"),
        ("base_mount", "base"),
    ):
        ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})

    arm_actuators = arm_root.find("actuator")
    gripper_actuators = gripper_root.find("actuator")
    assert arm_actuators is not None and gripper_actuators is not None
    for actuator in gripper_actuators:
        copied = copy.deepcopy(actuator)
        remap_references(copied)
        arm_actuators.append(copied)

    option = arm_root.find("option")
    assert option is not None
    option.set("impratio", "10")

    # The arm-only home keyframe is not valid after adding gripper DoFs. The
    # runner initializes joints by name and controls the gripper actuator.
    keyframe = arm_root.find("keyframe")
    if keyframe is not None:
        arm_root.remove(keyframe)


def build() -> Path:
    arm_root = ET.parse(ARM_XML).getroot()
    gripper_root = ET.parse(GRIPPER_XML).getroot()
    arm_root.set("model", "UR5 with Robotiq 2F-85")
    _append_gripper_assets(arm_root, gripper_root)
    ET.indent(arm_root, space="  ")
    OUTPUT_XML.write_text(ET.tostring(arm_root, encoding="unicode") + "\n", encoding="utf-8")
    return OUTPUT_XML


if __name__ == "__main__":
    print(build())
