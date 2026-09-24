# UR5 模型来源说明

[English](README.md) · 简体中文

`ur5.xml` 是本项目的 MJCF 模型，依据 Universal Robots 的 `Universal_Robots_ROS2_Description` 仓库中原版 UR5 的运动学和物理参数建立。参数对照了 `config/ur5/default_kinematics.yaml`、`joint_limits.yaml`、`physical_parameters.yaml`、`visual_parameters.yaml` 和 `urdf/ur_macro.xacro`。零位及非零关节配置下的运动学坐标系均与官方描述一致。六个连杆的碰撞网格来自该仓库的 `config/ur5`，同时用作可视网格和接触网格。

UR5 来源提交：`89bbe795f38a7ab00fb66fe8831dfff79dc99edf`，许可证为 BSD-3-Clause，见 [`UR_DESCRIPTION_LICENSE`](UR_DESCRIPTION_LICENSE)。

本项目最初检查过 `parhamkebria/UR5`，其提交为 `c215f44ef8f11554ae584dfca366b6ee722e8f36`，许可证为 Apache-2.0。该候选模型的连杆偏移、关节坐标系和末端执行器位置与当前官方 UR5 描述不一致，因此没有在本项目中分发它的模型和网格。

2F-85 模型由未经修改的 MuJoCo Menagerie 模型及其资源装配而成。夹爪的 `base_mount` 安装在 UR 官方 `tool0` 坐标系：根据 URDF 定义，`tool0` 相对 `wrist_3_link` 的合成旋转为单位旋转，平移为零。`tcp` 站点与上游模型中两指之间的 `pinch` 站点重合。MuJoCo Menagerie 来源提交：`c96a32d28fb5da84da38c1da4d749e7a13212855`，许可证为 BSD-2-Clause，见 [2F-85 许可证](../robotiq_2f85/LICENSE)。组合模型 `ur5_2f85.xml` 可通过 `tools/assemble_model.py` 重新生成。
