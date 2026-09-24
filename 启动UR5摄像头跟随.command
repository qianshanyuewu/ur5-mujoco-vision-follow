#!/bin/bash

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
MJPYTHON="$PROJECT_ROOT/.venv/bin/mjpython"
HAND_MODEL="$PROJECT_ROOT/assets/models/hand_landmarker.task"

pause_before_close() {
    if [ -t 0 ]; then
        read -r -p "按回车关闭此终端窗口... " _
    fi
}

fail() {
    echo "启动失败：$1" >&2
    pause_before_close
    exit 1
}

cd "$PROJECT_ROOT" || fail "无法进入项目目录：$PROJECT_ROOT"

if [ ! -x "$PYTHON" ] || [ ! -x "$MJPYTHON" ]; then
    fail "找不到项目虚拟环境。请先按 README.zh-CN.md 的“环境安装”步骤创建 .venv 并安装依赖。"
fi

if ! "$PYTHON" -c 'import cv2, mediapipe' >/dev/null 2>&1; then
    fail "视觉依赖未安装。请运行：.venv/bin/python -m pip install -r requirements-vision.txt"
fi

if [ ! -f "$HAND_MODEL" ]; then
    fail "缺少手部检测模型。请运行：.venv/bin/python tools/download_hand_landmarker.py"
fi

echo "正在启动 UR5 + Robotiq 2F-85 摄像头跟随仿真..."
echo "摄像头窗口：按 s 拖框选择物品；按空格开始或暂停机械臂跟随；按 q 退出。"
echo "如果摄像头不可用，请在 macOS 隐私设置中允许 Terminal 访问摄像头。"
echo

"$MJPYTHON" -m ur5_mujoco \
    --camera-follow \
    --move-arm \
    --object-distance 0.30 \
    "$@"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo "仿真已退出，错误码：$STATUS" >&2
fi
pause_before_close
exit "$STATUS"
