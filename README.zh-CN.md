# UR5 + Robotiq 2F-85 TCP 跟踪原型

[English](README.md) · 简体中文

这是一个以 Mac 为主的 MuJoCo 仿真原型。阻尼最小二乘雅可比控制器会跟随程序生成的笛卡尔目标。TCP 使用 2F-85 两指之间的 `pinch` 中心；机械臂腕部姿态保持为启动时的方向。模型世界坐标系对应 UR ROS 描述中的 `base_link`；位置单位为米，角度单位为弧度，MuJoCo 四元数顺序为 `w x y z`。

UR5 的运动学链和连杆参数已对照 [Universal Robots 官方 UR5 描述](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)核验。最初检查的公开 UR5 MuJoCo 候选模型，其连杆坐标系和 TCP 定义与官方描述不同，因此没有直接采用。夹爪模型来自 MuJoCo Menagerie，并安装在 UR5 官方 `tool0` 坐标系。模型来源、提交版本和许可证见[中文模型说明](assets/ur5/README.zh-CN.md)。

## 环境安装

本项目面向 Apple Silicon，固定使用 Python 3.13、MuJoCo 3.12.0 和 NumPy 2.5.3：

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 运行

无界面运行 60 秒直线跟踪，并将 CSV 日志写入 `logs/`：

```sh
.venv/bin/python -m ur5_mujoco --trajectory line --seconds 60
```

在 macOS 上通过交互式查看器运行圆形跟踪：

```sh
.venv/bin/mjpython -m ur5_mujoco --trajectory circle --seconds 60 --viewer
```

2F-85 默认打开。使用 `--gripper closed` 可闭合夹爪；`--speed` 设置目标速度，单位为米/秒；`--radius` 设置直线往返幅度或圆轨迹半径，单位为米；`--output PATH` 指定 CSV 日志路径。

## 仿真与控制

- 物理步长：2 ms；控制器频率：100 Hz。
- 默认目标速度：0.02 m/s；UR5 关节速度上限：0.3 rad/s。
- 启动稳定时间：0.75 秒。
- 控制器使用重力偏置力前馈，避免机械臂因重力偏离初始化姿态；相邻连杆之间排除了自碰撞，因为其碰撞网格在关节处存在轻微重叠。
- 雅可比控制命令带有关节速度和加速度限制，并在接近关节范围时进行保护。目标姿态保持不变。
- CSV 记录目标与实际 TCP 位置、位置误差、姿态误差、六个 UR5 关节的位置与速度，以及夹爪命令。命令行会报告位置误差 RMS、峰值误差和实测及命令的最大关节速度。

## 验证

```sh
.venv/bin/python -m pip install -r requirements-vision.txt
.venv/bin/python -m unittest discover -s tests -v
```

自动检查包括模型加载、六个 UR5 关节的独立控制、官方 UR5 法兰正运动学、启动时保持 Home 姿态，以及 60 秒直线和圆形跟踪。跟踪验收阈值为：TCP 位置误差 RMS ≤ 10 mm、峰值 ≤ 25 mm、实测关节速度 ≤ 0.3 rad/s。

## 后续阶段

当前集成路线为“Mac 摄像头手动框选物品 → CSRT 实时跟踪 → MuJoCo 固定平面跟随”。连接代码已实现，实时组合联调尚未执行；自动发现手持物品仍未验证，列为后续能力，不阻塞本轮手动跟踪集成。阶段安排与验收条件见[实施计划](docs/stage3-implementation-plan.zh-CN.md)；早期技术依据见[视觉跟踪调研](docs/stage3-vision-research.zh-CN.md)。

### 阶段 3：实时视觉原型

在 macOS 上安装视觉依赖并下载官方手部模型：

```sh
.venv/bin/python -m pip install -r requirements-vision.txt
.venv/bin/python tools/download_hand_landmarker.py
```

启动实时摄像头窗口。按 `q` 退出；画面中恰好检测到一只手时按 `s`，可拖框选中手持物品作为备用初始化，然后由 CSRT 跟踪：

```sh
.venv/bin/python -m ur5_mujoco.vision
```

摄像头访问失败时，请在 macOS 隐私设置中允许运行命令的终端访问摄像头。无窗口验证可用 `.venv/bin/python -m ur5_mujoco.vision --frames 60 --no-display`。当前自动持物发现尚未通过验证；手部检测、摄像头采集和手动初始化后的目标跟踪是分开的能力，不能把手动框选结果当作自动识别。

### 摄像头到 MuJoCo 平面跟随

在 macOS Finder 中双击项目根目录的 `启动UR5摄像头跟随.command`，即可启动摄像头和机械臂跟随模式。摄像头窗口按 `s` 拖框选中物品，按空格开始或暂停跟随，按 `q` 退出。启动文件使用项目现有的 `.venv`；首次安装时请先完成环境和视觉依赖安装，并下载手部模型。

先启动**标记预览模式**：MuJoCo 窗口会一直显示 UR5，摄像头跟踪窗口单独打开；此模式只移动仿真中的物品标记，机械臂保持在启动姿态。macOS 使用 `mjpython` 启动被动查看器（见 [MuJoCo Python 文档](https://mujoco.readthedocs.io/en/3.3.5/python.html#passive-viewer)）：

```sh
.venv/bin/mjpython -m ur5_mujoco --camera-follow
```

在摄像头窗口中，确保恰好检测到一只手，按 `s` 拖框选择物品。确认选框后，黄色物品标记会立即出现在夹爪前方默认 0.30 m 处，并随框选物体的左右、上下移动；此预览模式下机械臂保持不动。距离可通过 `--object-distance` 设置。按 `q` 结束会话。确认映射方向后，再显式允许仿真机械臂运动：

```sh
.venv/bin/mjpython -m ur5_mujoco --camera-follow --move-arm
```

启用 `--move-arm` 后，按空格开始/暂停机械臂跟随；机械臂跟随标记时，会保持夹爪到标记的设定距离。短暂追踪暂停时，标记保持在最后位置并等待恢复；目标持续丢失或图像超过恢复窗口后，机械臂停止跟随并隐藏标记。可用 `--seconds 60` 限定会话时长；默认运行到关闭窗口。图像左右映射到机器人基座 X 轴、上下映射到 Z 轴；相机位移映射灵敏度提高到原来的约 2.5 倍，最大行程仍限制在锚点两侧 50 mm 和 40 mm。物品标记球现在半径为 30 mm，使用高对比洋红色。Mac 摄像头没有深度测量；0.30 m 是标记平面的假定距离，不代表测得的物品深度。

UR5 的 Home 朝向已相对原姿态反转 180°，夹爪水平朝向 -Y。机械臂跟随使用更快的控制响应：TCP 位置增益为 5，线速度上限为 0.08 m/s，关节速度上限为 0.45 rad/s，关节加速度上限为 2.0 rad/s²。跟随输入使用三帧中值和 1 mm 目标死区；检测框在 0.4 秒内基本稳定时，固定跟随目标，持续移动超过阈值后恢复跟随，以减少物品停下时机械臂末端的细小晃动。
