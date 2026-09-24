# UR5 + Robotiq 2F-85 in MuJoCo

A macOS demo of camera-based object following with a simulated UR5 and Robotiq 2F-85 gripper.

## Demo

![UR5 camera-follow demo](https://github.com/user-attachments/assets/d2f2fda0-b852-4656-a3ef-df0d6fc547ae)

## Run

Requires macOS and Python 3.13.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-vision.txt
.venv/bin/python tools/download_hand_landmarker.py
.venv/bin/mjpython -m ur5_mujoco --camera-follow --move-arm
```

Press `s` to select the object, Space to pause or resume, and `q` to quit.
