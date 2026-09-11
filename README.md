# 养鸡场机器人视觉检测

本项目是一个面向养鸡场巡检场景的 ROS 2/Python 视觉检测工作空间。系统读取本地视频、摄像头或实时图像，使用 YOLO 检测鸡笼、鸡、鸡蛋等目标，结合 PaddleOCR 识别编号，并可通过 WebSocket 向后端上传检测结果和画面。

## 主要功能

- 多路视频源轮询和画面处理。
- 鸡笼、鸡、鸡蛋等目标检测与跟踪。
- 鸡笼编号 OCR 识别及结果缓存。
- 异常/统计数据组织与图片编码上传。
- WebSocket 实时画面和检测结果传输。
- 保留 ROS 2 发布者、订阅者示例及节点入口。

## 目录结构

```text
养鸡场机器人相关代码/
├── README.md
├── requirements.txt
└── ckegg_ws/
    └── src/
        └── ckegg_detect/
            ├── ckegg_detect/     # Python 节点
            ├── models/           # YOLO 模型
            ├── videos/           # 示例视频
            ├── version_save/     # 历史版本
            ├── package.xml
            └── setup.py
```

## 推荐环境

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- NVIDIA Jetson（可选，用于 GPU 加速）

## 安装

```bash
sudo apt update
sudo apt install python3-colcon-common-extensions python3-rosdep

cd "养鸡场机器人相关代码"
python3 -m pip install -r requirements.txt
cd ckegg_ws
rosdep install --from-paths src --ignore-src -r -y
```

Jetson 用户应根据 JetPack 版本安装 NVIDIA 提供的 PyTorch wheel；使用 GPU 版 PaddlePaddle 时，也应按 CUDA 版本选择对应安装包。

## 配置

主程序当前在 `ckegg_detect/ckegg_detect/ckegg_detection_publisher.py` 中配置视频源、模型路径和 WebSocket 地址。首次运行前请修改以下内容：

- `video_sources`：本地视频、摄像头或网络流地址。
- `cage_model` / `item_model`：模型文件的实际路径。
- `server_url`：后端 WebSocket 地址。
- 各目标的置信度阈值和是否启用本地显示。

源码中存在针对开发设备的 `/home/jetson/...` 绝对路径，换机运行时必须调整。

## 编译与运行

```bash
cd "养鸡场机器人相关代码/ckegg_ws"
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

ros2 run ckegg_detect ckegg_detection_publisher
```

也可运行基础 ROS 2 通信示例：

```bash
ros2 run ckegg_detect talker
ros2 run ckegg_detect listener
```

如果当前检测程序采用了不继承 ROS 2 `Node` 的独立运行版本，也可以从包目录直接执行对应 Python 文件。

## 模型与数据

`models/` 和 `videos/` 可能包含大文件。上传 GitHub 前建议使用 Git LFS，或只保留小型示例并在 README 中提供模型下载方式。请勿提交真实生产环境的服务器地址、账号、密钥或敏感养殖数据。

