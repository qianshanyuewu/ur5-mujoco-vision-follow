# 阶段 3：任意小物品的视觉跟踪调研

日期：2026-09-24。适用范围：Mac 普通二维摄像头观察人手附近的一个小物品，向现有 UR5 + 2F-85 MuJoCo 仿真提供目标观测。D435 三维定位属于后续实验室阶段。

> 本文保留自动手持物品发现方向的早期调研记录。当前已确定先用 Mac 摄像头手动框选、CSRT 跟踪并接入 MuJoCo 平面跟随；自动发现能力不作为本轮前置条件。实施范围和状态以[当前计划](stage3-implementation-plan.zh-CN.md)为准，本文关于自动检测模型的比较留作后续参考。

## 结论

第一版采用**人工框选目标 + 单实例跟踪 + 可见性状态机**。使用者在物品清楚可见时框选它，系统只跟踪这个实例。任意小物品没有预先限定类别，通用类别检测器不能保证覆盖；MediaPipe Object Detector 的输出本身也是模型类别、分数与框，适合在类别确定后作为辅助检测器。[MediaPipe Object Detector 文档](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/python)

先用 OpenCV CSRT 实现可复现实验基线。它接受初始框，随后返回跟踪成功标志和更新后的框；接口也允许设置初始掩膜。但返回的布尔值不等于经过校准的置信概率，不能单独用来决定机器人是否继续跟随。[OpenCV TrackerCSRT 文档](https://docs.opencv.org/4.13.0/d2/da2/classcv_1_1TrackerCSRT.html)

同时评估 MediaPipe Hand Landmarker，只把手的位置作为关联与遮挡判断线索。它提供实时流模式和异步回调，但手部关键点不会直接给出物品边界，也无法单凭二维接近关系证明物品被拿在手里。[MediaPipe Hand Landmarker 文档](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)

如果 CSRT 在代表性小物品上频繁漂移，再评估可由框/点击提示并传播分割掩膜的 SAM 2。其公开性能数据不能直接推断本机 Mac 的实时速度，需在实际设备上测帧率和端到端延迟。[SAM 2 官方仓库](https://github.com/facebookresearch/sam2)

## 识别与遮挡策略

| 场景 | 阶段 3 行为 | 下游目标处理 |
| --- | --- | --- |
| 物品清楚可见 | 用户框选，建立唯一 `track_id` 和初始外观/尺寸 | 可使用新鲜观测 |
| 部分被手指遮挡 | 结合可见物品区域、历史位置、尺寸变化及手部线索更新 | 质量下降时减速或保持 |
| 完全遮挡 | 短时运动预测只用于限定再次搜索区域；标记为 `OCCLUDED` | 保持最后安全目标，不把预测当实测值 |
| 重现且身份明确 | 在预测区域内核对外观和运动连续性后恢复 `TRACKED` | 恢复跟随 |
| 遮挡过久、出现相似物或离开画面 | 标记为 `LOST`，等待人工重新框选 | 停止更新目标 |

状态为 `WAIT_SELECT → TRACKED → OCCLUDED → LOST`，允许确认身份后从 `OCCLUDED` 回到 `TRACKED`。完整遮挡期间没有视觉证据可保证重现的是原物；背景里有相同物品时，系统应选择失锁。手被挡住、两只手交叉、物品被放下或由另一只手接过，也必须作为测试场景。遮挡相关阈值从录制数据调定，并记录每次状态切换原因。

小物品的像素尺寸往往比模型选型更关键。先记录物品最小可见宽度、摄像头到工作区距离、运动模糊、反光、背景对比和照明变化；若目标只有少量像素，放大推理图像不会恢复丢失的细节。界面应显示原始画面、所选框、跟踪框/掩膜、状态和观测时间，让误跟踪可被直接发现。

## 实时数据通道

Mac 摄像头由独立采集模块读取；OpenCV `VideoCapture` 可从摄像头逐帧读取。[OpenCV VideoCapture 文档](https://docs.opencv.org/4.13.0/d8/dfe/classcv_1_1VideoCapture.html) 采集时立即打单调时钟时间戳，推理只取最新帧，队列有界并丢弃过时帧。控制器仍以现有 100 Hz 频率运行，消费最近一条**新鲜、可见、身份明确**的观测，不要求视觉也达到 100 Hz。异步视觉组件可能跳过帧，不能把回调频率等同于摄像头帧率；MediaPipe 实时模式也允许在繁忙时丢帧。[MediaPipe Hand Landmarker API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker)

建议的观测接口（此阶段仍为图像坐标）：

```text
VisionObservation {
  source_id, frame_id, capture_time_monotonic,
  image_width, image_height,
  track_id, status,                 # WAIT_SELECT/TRACKED/OCCLUDED/LOST
  center_uv, bbox_xywh, mask?,      # 像素坐标；缺失时不伪造测量
  quality, quality_reasons          # 经验质量指标，不冒充概率
}
```

图像坐标与机械臂基座坐标必须分开。日志至少记录采集、推理完成和控制消费时间，以计算端到端延迟与观测年龄；同时记录丢帧数、状态、框位置/尺寸、身份切换、手部线索及人工重选事件。建议把 150–250 ms 作为“观测过期”的**初始实验范围**，最终阈值取决于实测延迟、目标速度和允许的机械臂跟随误差。

物品中心是视觉观测点，2F-85 的 `pinch` 是机械臂 TCP；二者之间的抓取偏移和姿态尚未定义。阶段 3 只传观测，后续目标生成模块再决定 TCP 要到哪里。若视觉处理搬到另一台电脑，还需统一时钟或估计时钟偏移，不能直接相减两台机器的单调时间戳。

## 二维摄像头到 D435 的迁移

阶段 3 输出的物品身份、二维位置、可见性状态和时间戳，可保持为共同接口。Mac 摄像头适配器输出 RGB 帧；后续 Ubuntu 上的 D435 适配器输出 RGB、**与 RGB 对齐的深度**、内参和时间戳。外观跟踪与手部检测可以复用处理结构，但视场、分辨率、曝光和色彩变化需要重新评估。

阶段 4 的普通摄像头演示，需要先标定固定工作平面，把像素位置映射到该平面上的基座 `x, y`，并使用指定 `z`。这是二维投影实验；手在前后方向移动时，单目画面不能直接提供真实三维位置。

D435 阶段应改为：从物品可见区域内采样有效对齐深度 → 用内参反投影到相机三维坐标 → 通过标定的 `T_base_camera` 转换到 UR5 `base_link` → 检查工作空间、速度限制并生成 2F-85 `pinch` TCP 目标。RealSense SDK 文档明确区分像素到三维的反投影、相机流间对齐及不同相机坐标系。[RealSense 投影文档](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/) 对齐和遮挡仍可能造成深度空洞或前后景混入，因此深度应从多个有效像素做稳健统计，记录有效像素比例和离散度。若只有包含手指的目标框、无法分离物品可见区域，或没有可靠深度，就保持目标。[RealSense 遮挡与对齐说明](https://dev.realsenseai.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras/)

相机固定方式一旦改变，外参必须重标定。还需实测目标在预定工作距离上的深度有效率，尤其是很小、反光或缺少纹理的物品。RealSense 官方 macOS 安装说明指出其 macOS 支持有限；实验室 Ubuntu 环境应作为 D435 集成的主要验证环境。[librealsense macOS 安装说明](https://github.com/realsenseai/librealsense/blob/master/doc/installation_osx.md) [librealsense Ubuntu 安装说明](https://github.com/realsenseai/librealsense/blob/master/doc/installation.md)

## 实验与阶段 3 验收

1. 用实际 Mac 摄像头录制可重放视频：至少覆盖几种不同颜色、材质和尺寸的小物品；包括同类背景物、双手交叉、部分及完整遮挡、快速移动、物品放下、画面出入和照明变化。记录实际物品尺寸、画面像素宽度、帧率及拍摄距离。
2. 为每段视频标注目标身份、可见区间和中心点；按**同一套数据**比较 CSRT、手部辅助策略，以及需要时的 SAM 2。分别统计可见帧定位误差、身份误切换次数、遮挡后恢复率与恢复时间、失锁上报时间、端到端延迟 P95 和有效输出帧率。
3. 阶段 3 的关键验收是“失去证据就明确失锁”，并在含相似物的测试中实现零自动身份误切换。再根据实际画面大小和机械臂目标误差预算，确定像素误差与观测年龄阈值。连续运行 60 秒，检查画面、日志与状态切换一致。
4. 阶段 4 另测平面标定误差、工作区边界、失锁保持和目标平滑；D435 阶段另测有效深度比例、相机到基座标定误差和三维定位误差。阶段 1–2 的 TCP 跟踪误差指标不能直接替代视觉误差指标。

## 实施顺序

1. 采集、回放、人工框选和观测日志。
2. CSRT 单实例跟踪与显式遮挡/失锁状态；做代表性数据基线。
3. 加手部线索，专测手指遮挡和背景中相似物。
4. 若基线无法达到项目数据集要求，再实测 SAM 2 的准确度和 Mac 延迟。
5. 经由独立坐标映射模块接入 MuJoCo；以后将摄像头适配器换为 D435 并增加深度与外参标定。
