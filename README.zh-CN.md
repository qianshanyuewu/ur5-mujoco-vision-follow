# UR5 + Robotiq 2F-85 MuJoCo 仿真

基于 macOS 的摄像头目标跟随演示，机械臂仅在 MuJoCo 中运行。

## 演示

![UR5 摄像头跟随演示](https://github.com/user-attachments/assets/d2f2fda0-b852-4656-a3ef-df0d6fc547ae)

## 运行

需要 macOS 和 Python 3.13。

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-vision.txt
.venv/bin/python tools/download_hand_landmarker.py
.venv/bin/mjpython -m ur5_mujoco --camera-follow --move-arm
```

按 `s` 框选目标，空格暂停或继续，按 `q` 退出。
