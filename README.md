# UR5 + Robotiq 2F-85 TCP following

[简体中文](README.zh-CN.md) · English

Mac-first MuJoCo prototype. A damped least-squares Jacobian controller follows
generated Cartesian targets with the 2F-85 `tcp` site at the center between the
fingers. The wrist orientation remains fixed at its startup pose. The model's
world frame is UR's ROS `base_link` frame; positions are metres, angles are
radians, and MuJoCo quaternions use `w x y z` order.

The UR5 kinematic chain and link parameters are checked against the official
Universal Robots UR5 description. The initial public UR5 MuJoCo candidate was
not used because its link frames and TCP site differed. The gripper comes from
MuJoCo Menagerie and is mounted at the official UR5 `tool0` frame. See
[`assets/ur5/README.md`](assets/ur5/README.md) for source commits and licences.

## Setup

Python 3.13 and MuJoCo 3.12.0 are pinned for this Apple Silicon project:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

Run headless line tracking and write a 60-second CSV log:

```sh
.venv/bin/python -m ur5_mujoco --trajectory line --seconds 60
```

Run circle tracking with the interactive viewer on macOS:

```sh
.venv/bin/mjpython -m ur5_mujoco --trajectory circle --seconds 60 --viewer
```

The 2F-85 opens by default. Use `--gripper closed` to close it, `--speed` to
change target speed (m/s), `--radius` to set line amplitude/circle radius (m),
and `--output PATH` to choose the CSV path. Default logs go to `logs/`.

The simulation uses a 2 ms physics step, 100 Hz controller update, 0.02 m/s
trajectory speed, 0.3 rad/s arm joint-speed cap, and 0.75 s startup settling
period. The controller includes bias-force feedforward so gravity does not
pull the arm away from its initialized pose. Adjacent links are excluded from
self-collision because their collision meshes overlap slightly at the joints.
Jacobian commands are rate-limited and joint limits are guarded. The CSV
contains target/actual TCP position, pose error, six arm joint positions and
velocities, and gripper command. The CLI prints RMS and peak position error
plus maximum measured and commanded joint speeds.

## Verify

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The checks cover model loading, six independently actuated UR5 joints, the
official UR5 forward-kinematics frames, and 60-second line/circle tracking. The
tracking acceptance limits are ≤10 mm RMS TCP position error, ≤25 mm peak
error, and ≤0.3 rad/s measured joint speed.

## Camera-follow integration

The selected prototype path is Mac camera manual object selection, CSRT
tracking, then planar TCP following in MuJoCo. Manual selection has been
confirmed on the live Mac camera, and the simulation controller has passed
trajectory checks independently; camera-follow code is implemented, but the
combined live session has not yet been run. Automatic held-object discovery
remains future work. See the
[implementation plan](docs/stage3-implementation-plan.zh-CN.md) for integration
steps and acceptance criteria.

Start marker-only preview with the MuJoCo viewer continuously open:

```sh
.venv/bin/mjpython -m ur5_mujoco --camera-follow
```

In the camera window, press `s` to select the object. As soon as the box is
confirmed, a yellow object marker appears 0.30 m in front of the gripper and
follows the object's image motion. The arm stays at its starting pose in this
preview mode. Set the assumed stand-off distance with `--object-distance`. After
confirming the mapping, enable simulated arm motion explicitly:

```sh
.venv/bin/mjpython -m ur5_mujoco --camera-follow --move-arm
```

With `--move-arm`, press space to start or pause arm following; the gripper
maintains the configured distance from the object marker. During a brief
tracking pause, the marker holds its last position while tracking recovers. A
prolonged loss hides the marker and pauses arm following. The session runs
until a window is closed; pass `--seconds 60` to set a time limit.

The UR5 home heading is reversed by 180 degrees, with the gripper pointing
horizontally along -Y. Arm-follow mode uses a faster controller profile: TCP
position gain 5, linear speed limit 0.08 m/s, joint speed limit 0.45 rad/s, and
joint acceleration limit 2.0 rad/s². A three-frame median, 1 mm target
deadband, and 0.4-second stillness gate hold the follow target when the tracked
object is nearly still, then resume following after movement is detected.
