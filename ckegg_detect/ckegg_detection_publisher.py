#!/usr/bin/env python3
import sys
import os
import time
import json
import base64
import threading
import queue
import re
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge

# 深度学习与图像处理库
sys.path.append('/home/jetson/anaconda3/envs/ckegg/lib/python3.10/site-packages/')
try:
    import torch
    from ultralytics import YOLO
    import paddle
    from paddleocr import PaddleOCR
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
    import websocket
except ImportError as e:
    print(f"❌ 依赖库缺失: {e}")
    raise


class ChickenFarmNode:
    def __init__(self):
        # super().__init__('ckegg_detection_publisher')  # 注释掉ROS2父类初始化

        # 暂时注释掉ROS2参数声明，直接设置默认值
        # self.declare_parameters(
        #     namespace='',
        #     parameters=[
        #         ('source', ''),
        #         ('image_topic', '/camera/image'),
        #         ('cage_model', '/home/jetson/ckegg_ws/src/ckegg_detect/models/cage_best.pt'),
        #         ('item_model', '/home/jetson/ckegg_ws/src/ckegg_detect/models/items_best.pt'),
        #         ('server_url', 'ws://192.168.43.7:8080/LeogEgg/websocket'),
        #         ('upload_enabled', True),
        #         ('video_loop', False),
        #         ('disappear_timeout', 2.0),
        #         ('chicken_conf', 0.4),
        #         ('egg_conf', 0.45),
        #         ('number_conf', 0.4),
        #         ('video_stream_quality', 55),
        #         ('video_stream_width', 640),
        #         ('upload_image_width', 800),
        #         ('save_root', ''),
        #         ('local_display', True),
        #     ],
        # )

        # 直接设置参数值 - 支持四个视频源
        self.video_sources = [
            '/home/jetson/ckegg_ws/src/ckegg_detect/videos/2.mp4',  # 一层画面
            '/home/jetson/ckegg_ws/src/ckegg_detect/videos/6.mp4',  # 二层画面  
            '',  # 三层画面
            '',   # 四层画面
        ]
        self.layer_labels = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
        
        # 兼容性：保留单个source变量（使用第一个视频源）
        self.source = self.video_sources[0]
        self.image_topic = '/camera/color/image_raw'
        self.server_url = 'ws://192.168.43.7:8080/LeogEgg/websocket'
        self.upload_enabled = True
        self.video_loop = False
        self.disappear_threshold = 2.0
        
        # 多视频模式标志
        self.multi_video_mode = len([v for v in self.video_sources if v]) > 1
        print(f"🎥 多视频模式: {'启用' if self.multi_video_mode else '禁用'}")
        if self.multi_video_mode:
            valid_videos = [f"{self.layer_labels[i]}: {v}" for i, v in enumerate(self.video_sources) if v]
            print(f"   有效视频源: {', '.join(valid_videos)}")

        self.conf_thresholds = {
            'chicken': 0.4,
            'egg': 0.45,
            'number': 0.4,
            'cage': 0.5,
        }

        self.video_stream_quality = 55
        self.video_stream_width = 640
        self.upload_image_width = 800
        self.enable_local_display = True
        self._display_window = 'ChickenFarm Detection'
        self._window_initialized = False
        self.display_queue = None
        self.display_thread = None

        cage_model = '/home/jetson/ckegg_ws/src/ckegg_detect/models/cage_best.pt'
        item_model = '/home/jetson/ckegg_ws/src/ckegg_detect/models/items_best.pt'
        self.model_paths = {'cage': cage_model, 'item': item_model}

        # self.bridge = CvBridge()  # 注释掉CVBridge
        self.running = True
        self.frame_count = 0
        self._last_frame_ts = 0.0
        self._last_process_log_ts = 0.0
        self._process_log_interval = 2.0
        self._last_stats_log_ts = 0.0
        self._stats_log_interval = 1.0
        self._last_timer_warn_ts = 0.0
        self._timer_warn_interval = 3.0

        self.cage_registry = {}

        self._cage_tracks = {}
        self._next_track_id = 1
        self._track_iou_threshold = 0.35
        self._track_timeout = float(self.disappear_threshold)

        self.ocr_cache = {}
        self.ocr_cache_timeout = 5.0
        self.ocr_queue = queue.Queue(maxsize=50)
        self.ocr_result_queue = queue.Queue()
        self.ocr_skip_frames = 1
        self.pending_ocr = {}
        self.pending_ocr_order = deque()
        self.pending_ocr_set = set()
        self.ocr_request_interval = 0.6
        self.ocr_pending_stale_timeout = 10.0

        self.video_ws_queue = queue.Queue(maxsize=4)
        self.upload_ws_queue = queue.Queue()
        self._video_ws = None
        self._upload_ws = None
        self._ws_fail_count = 0
        self._ws_last_offline_log = 0.0
        self._ws_fail_limit = 10

        self._uploaded_cage_ids = set()
        self._uploaded_cage_lock = threading.Lock()

        self._last_valid_cage_ids = deque(maxlen=3)
        self._id_jump_tolerance = 2

        self._ws_enabled = bool(self.upload_enabled and self.server_url)
        self._ws_disabled = False

        # 暂时注释掉保存目录设置，直接使用默认值
        # if self.get_parameter('save_root').value:
        #     save_root = str(self.get_parameter('save_root').value)
        # else:
        save_root = os.path.join(os.path.expanduser('/home/jetson/ckegg_ws/src/ckegg_detect'), 'ckegg_data')

        self.save_dirs = {
            'count': os.path.join(save_root, 'count_data'),
            'image': os.path.join(save_root, 'image_data'),
            'debug_ocr': os.path.join(save_root, 'debug_ocr_crops'),
        }
        for d in self.save_dirs.values():
            os.makedirs(d, exist_ok=True)

        self._validate_paths()
        self.init_models()
        self.setup_font()
        self.start_threads()
        self._setup_input_source()

        # self._log_runtime_env()  # 注释掉运行环境日志

        # print('✅ ckegg 检测节点已启动')  # 注释掉ROS2日志，改用普通print
        print('✅ ckegg 检测节点已启动 - 使用本地视频模式')

    def create_timer(self, period, callback):
        """模拟ROS2的create_timer方法，但在这里不使用"""
        # 这个方法在非ROS2模式下不需要实现
        pass

    def destroy_node(self):
        """清理资源"""
        self.running = False

        try:
            # 清理所有视频源的registry
            if hasattr(self, 'multi_cage_registries'):
                for registry in self.multi_cage_registries:
                    if registry:
                        self.cage_registry = registry
                        self.flush_registry()
            else:
                self.flush_registry()
        except Exception:
            pass

        try:
            # 清理多个视频捕获对象
            if hasattr(self, 'caps'):
                for cap in self.caps:
                    if cap is not None:
                        cap.release()
            # 兼容性：清理单个cap
            elif hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
        except Exception:
            pass

        self.close_display_window()

        self._close_ws('_video_ws')
        self._close_ws('_upload_ws')
        # super().destroy_node()  # 注释掉ROS2父类方法

    def _stable_cage_key(self, x1: int, y1: int, x2: int, y2: int) -> str:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        return f"{cx//25}_{cy//25}_{w//50}_{h//50}"

    def _bbox_iou(self, a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0.0:
            return 0.0
        a_area = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        b_area = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        denom = a_area + b_area - inter
        return inter / denom if denom > 0.0 else 0.0

    def _cleanup_tracks(self, current_time: float):
        if not self._cage_tracks:
            return
        stale = [tid for tid, tr in self._cage_tracks.items() if current_time - tr.get('last_seen', 0.0) > self._track_timeout]
        for tid in stale:
            self._cage_tracks.pop(tid, None)

    def _assign_track_id(self, bbox, current_time: float) -> int:
        self._cleanup_tracks(current_time)

        best_tid = None
        best_iou = 0.0
        for tid, tr in self._cage_tracks.items():
            iou = self._bbox_iou(bbox, tr.get('bbox', bbox))
            if iou > best_iou:
                best_iou = iou
                best_tid = tid

        if best_tid is not None and best_iou >= self._track_iou_threshold:
            tr = self._cage_tracks[best_tid]
            tr['bbox'] = bbox
            tr['last_seen'] = current_time
            return int(best_tid)

        tid = int(self._next_track_id)
        self._next_track_id += 1
        self._cage_tracks[tid] = {
            'bbox': bbox,
            'last_seen': current_time,
            'cage_id': None,
        }
        return tid

    def _log_runtime_env(self):
        try:
            disp = os.environ.get('DISPLAY', '')
            wayland = os.environ.get('WAYLAND_DISPLAY', '')
            sess = os.environ.get('XDG_SESSION_TYPE', '')
            print(f"RuntimeEnv DISPLAY='{disp}' WAYLAND_DISPLAY='{wayland}' XDG_SESSION_TYPE='{sess}'")
            if self.enable_local_display and (not disp and not wayland):
                self.enable_local_display = False
                print('⚠️ 检测到无 DISPLAY/WAYLAND 环境，已自动关闭本地窗口显示。将仅在终端输出统计信息。')
        except Exception:
            pass

    def _validate_paths(self):
        for k in ['cage', 'item']:
            p = self.model_paths.get(k)
            if not p:
                raise RuntimeError(f"模型路径参数未设置: {k}_model")
            if not os.path.exists(p):
                raise FileNotFoundError(f"模型文件不存在: {p}")

    def _setup_input_source(self):
        """设置多个视频输入源"""
        self.caps = []  # 存储多个VideoCapture对象
        self.video_valid = []  # 记录每个视频源是否有效
        
        for i, video_source in enumerate(self.video_sources):
            cap = None
            is_valid = False
            
            # 检查视频源是否存在
            if os.path.exists(video_source):
                print(f"🎥 加载视频源 {i+1} ({self.layer_labels[i]}): {video_source}")
                cap = cv2.VideoCapture(video_source)
                if cap.isOpened():
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    if fps <= 0.0 or np.isnan(fps):
                        fps = 30.0
                    print(f"   ✅ 视频打开成功，FPS: {fps}")
                    is_valid = True
                else:
                    print(f"   ❌ 无法打开视频文件")
                    cap.release()
                    cap = None
            else:
                print(f"⚠️ 视频文件不存在 ({self.layer_labels[i]}): {video_source}")
            
            self.caps.append(cap)
            self.video_valid.append(is_valid)
        
        # 检查是否有至少一个有效的视频源
        if not any(self.video_valid):
            print("❌ 没有找到有效的视频源，程序无法运行")
            raise RuntimeError("没有有效的视频源")
        
        # 计算统一的帧率（使用第一个有效视频的FPS）
        for i, cap in enumerate(self.caps):
            if cap is not None and self.video_valid[i]:
                self.fps = float(cap.get(cv2.CAP_PROP_FPS))
                if self.fps <= 0.0 or np.isnan(self.fps):
                    self.fps = 30.0
                break
        else:
            self.fps = 30.0
        
        print(f"🎬 使用统一帧率: {self.fps} FPS")
        
        # 启动多视频循环处理
        self.run_multi_video_loop()

    def init_models(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"🔄 加载模型中 (设备: {device})")

        self.cage_model = YOLO(self.model_paths['cage'])
        self.cage_model.to(device)

        self.item_model = YOLO(self.model_paths['item'])
        self.item_model.to(device)

        try:
            if paddle.device.is_compiled_with_cuda() and device.startswith('cuda'):
                paddle.device.set_device('gpu')
            else:
                paddle.device.set_device('cpu')
        except Exception:
            pass

        import logging
        logging.getLogger('ppocr').setLevel(logging.WARNING)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=device.startswith('cuda'), show_log=False)

    def setup_font(self):
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        ]
        self.font = ImageFont.load_default()
        for path in font_paths:
            if os.path.exists(path):
                try:
                    self.font = ImageFont.truetype(path, 20)
                    break
                except Exception:
                    continue

    def start_threads(self):
        t_ocr = threading.Thread(target=self.ocr_worker, daemon=True)
        t_ocr.start()

        if self._ws_enabled:
            try:
                test_ws = websocket.create_connection(self.server_url, timeout=0.8)
                try:
                    test_ws.close()
                except Exception:
                    pass
            except Exception:
                self._disable_ws_upload('启动时无法连接上位机 WebSocket')

        if self._ws_enabled:
            t_video = threading.Thread(target=self.video_stream_worker, daemon=True)
            t_video.start()
            t_upload = threading.Thread(target=self.upload_worker, daemon=True)
            t_upload.start()
            print(f"🌐 WebSocket 上传启用: {self.server_url}")

        if self.enable_local_display:
            print('🖥️ 本地显示启用')

    def run_multi_video_loop(self):
        """多视频循环处理方法，同时处理四个视频源"""
        frame_delay = 1.0 / self.fps
        print(f"🎬 开始多视频处理循环，帧间隔: {frame_delay:.3f}s")
        
        while self.running:
            frames = []  # 存储每个视频源的当前帧
            processed_frames = []  # 存储处理后的帧
            
            # 读取所有视频源的帧
            all_videos_ended = True
            for i, (cap, is_valid) in enumerate(zip(self.caps, self.video_valid)):
                if is_valid and cap is not None:
                    ret, frame = cap.read()
                    if ret:
                        all_videos_ended = False
                        # 处理这一帧
                        processed_frame = self.process_frame_with_detection(frame, i)
                        frames.append(frame)
                        processed_frames.append(processed_frame)
                    else:
                        # 视频结束，处理循环或黑屏
                        if self.video_loop:
                            print(f"🔄 {self.layer_labels[i]} 视频结束，重新开始")
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if ret:
                                all_videos_ended = False
                                processed_frame = self.process_frame_with_detection(frame, i)
                                frames.append(frame)
                                processed_frames.append(processed_frame)
                            else:
                                # 创建黑屏
                                black_frame = self.create_black_frame()
                                frames.append(black_frame)
                                processed_frames.append(self.add_layer_label(black_frame.copy(), i))
                        else:
                            print(f"⚠️ {self.layer_labels[i]} 视频结束")
                            # 创建黑屏
                            black_frame = self.create_black_frame()
                            frames.append(black_frame)
                            processed_frames.append(self.add_layer_label(black_frame.copy(), i))
                else:
                    # 无效视频源，创建黑屏
                    black_frame = self.create_black_frame()
                    frames.append(black_frame)
                    processed_frames.append(self.add_layer_label(black_frame.copy(), i))
            
            # 检查是否所有视频都结束了
            if all_videos_ended and not self.video_loop:
                print('🛑 所有视频播放结束，准备退出')
                break
            
            # 创建四画面合成显示
            print(f"🔄 开始合成四画面，处理帧数: {len(processed_frames)}")
            combined_frame = self.create_four_panel_display(processed_frames)
            
            # 显示合成画面
            if self.enable_local_display:
                print("📺 准备显示合成画面")
                self.render_multi_display_frame(combined_frame)
            
            # 发送到WebSocket（如果有需要）
            if self._ws_enabled:
                try:
                    self.video_ws_queue.put_nowait(combined_frame)
                except queue.Full:
                    pass
            
            # 控制帧率
            time.sleep(frame_delay)
        
        # 清理资源
        for cap in self.caps:
            if cap is not None:
                cap.release()
        
        print("🎬 多视频处理循环结束")

    def create_black_frame(self):
        """创建黑屏帧 - 动态获取视频尺寸"""
        # 尝试从第一个有效的视频源获取尺寸
        frame_size = None
        
        # 检查是否有caps属性
        if hasattr(self, 'caps') and self.caps:
            for i, (cap, is_valid) in enumerate(zip(self.caps, self.video_valid)):
                if is_valid and cap is not None:
                    try:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if width > 0 and height > 0:
                            frame_size = (height, width)
                            break
                    except:
                        continue
        
        # 如果无法获取尺寸，使用默认尺寸
        if frame_size is None:
            frame_size = (1080, 1920)  # (height, width)
        
        return np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)

    def add_layer_label(self, frame, layer_index):
        """在画面左上角添加层级标签"""
        if layer_index < len(self.layer_labels):
            label = self.layer_labels[layer_index]
            
            # 使用OpenCV渲染英文标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # 计算文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 绘制背景矩形
            margin = 5
            bg_x1, bg_y1 = 10 - margin, 10 - margin
            bg_x2, bg_y2 = 10 + text_width + margin, 10 + text_height + margin
            
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 255), 2)
            
            # 绘制文本
            cv2.putText(frame, label, (10, text_height + 10), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        
        return frame

    def create_four_panel_display(self, frames):
        """创建2x2四画面合成显示"""
        # 过滤掉None值，并用黑屏替换
        valid_frames = []
        for i, frame in enumerate(frames):
            if frame is None:
                print(f"⚠️ 帧 {i} 为None，使用黑屏替换")
                valid_frames.append(self.create_black_frame())
            else:
                valid_frames.append(frame)
        
        # 确保有4个帧（不足的用黑屏填充）
        while len(valid_frames) < 4:
            print(f"⚠️ 帧数量不足，当前: {len(valid_frames)}，添加黑屏")
            black_frame = self.create_black_frame()
            valid_frames.append(black_frame)
        
        # 获取单个画面的尺寸（使用第一个有效帧的尺寸）
        h, w = valid_frames[0].shape[:2]
        print(f"📐 使用基准尺寸: {w}x{h}")
        
        # 统一所有帧的尺寸
        unified_frames = []
        for i, frame in enumerate(valid_frames[:4]):
            if frame.shape[:2] != (h, w):
                print(f"🔄 调整帧 {i} 尺寸从 {frame.shape[:2]} 到 ({h}, {w})")
                frame = cv2.resize(frame, (w, h))
            unified_frames.append(frame)
        
        # 创建合成画面 (2x2布局)
        combined_h = h * 2
        combined_w = w * 2
        combined_frame = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
        print(f"🖼️ 创建合成画面: {combined_w}x{combined_h}")
        
        # 放置四个画面
        positions = [
            (0, 0),           # 左上：第一层
            (w, 0),           # 右上：第二层  
            (0, h),           # 左下：第三层
            (w, h)            # 右下：第四层
        ]
        
        for i, (frame, (x, y)) in enumerate(zip(unified_frames, positions)):
            # 放置到合成画面中
            combined_frame[y:y+h, x:x+w] = frame
            print(f"✅ 放置画面 {i+1} 到位置 ({x}, {y})")
        
        print(f"🎯 合成画面完成: {combined_frame.shape}")
        return combined_frame

    def process_frame_with_detection(self, frame, video_index):
        """处理单个视频帧的检测识别"""
        if frame is None:
            return None
        
        try:
            # 调用原有的process_frame方法进行检测
            # 但需要为每个视频源维护独立的cage_registry
            original_registry = self.cage_registry
            
            # 为当前视频源创建独立的registry（如果还没有的话）
            if not hasattr(self, 'multi_cage_registries'):
                self.multi_cage_registries = [{} for _ in range(4)]
            
            self.cage_registry = self.multi_cage_registries[video_index]
            
            # 处理帧
            processed_frame = self.process_frame(frame)
            
            # 恢复原始registry
            self.cage_registry = original_registry
            
            # 添加层级标签
            return self.add_layer_label(processed_frame, video_index)
        except Exception as e:
            print(f"处理视频 {video_index} 时出错: {e}")
            # 如果处理失败，返回带标签的黑屏
            black_frame = self.create_black_frame()
            return self.add_layer_label(black_frame, video_index)

    # def timer_callback(self):  # 注释掉原来的ROS2 timer回调
    #     if self.cap is None:
    #         return
    #     ret, frame = self.cap.read()
    #     if not ret:
    #         self.get_logger().info('⚠️ 视频流结束，正在结算剩余数据')
    #         self.flush_registry()

    #         if self.video_loop:
    #             self.get_logger().info('🔄 视频循环播放')
    #             self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #             self.cage_registry.clear()
    #             return

    #         self.get_logger().info('🛑 播放结束，准备退出')
    #         try:
    #             self.cap.release()
    #         except Exception:
    #             pass
    #         self.destroy_node()
    #         sys.exit(0)

    #     self._last_frame_ts = time.time()
    #     self.process_frame(frame)

    # def image_callback(self, msg: Image):  # 注释掉ROS2图像回调
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #         self.process_frame(cv_image)
    #     except Exception as e:
    #         self.get_logger().error(f"图像转换失败: {e}")

    def process_frame(self, frame):
        if frame is None:
            return

        self.frame_count += 1
        current_time = time.time()

        if current_time - self._last_process_log_ts > self._process_log_interval:
            self._last_process_log_ts = current_time

        self.update_ocr_results(current_time)

        detections = []
        updated_cage_ids = []
        frame_stats_map = {}

        h0, w0 = frame.shape[:2]
        infer_frame = frame
        sx, sy = 1.0, 1.0
        try:
            target_w = 640
            if w0 > target_w:
                ratio = float(target_w) / float(w0)
                infer_frame = cv2.resize(frame, (target_w, int(h0 * ratio)), interpolation=cv2.INTER_AREA)
                sx = float(w0) / float(infer_frame.shape[1])
                sy = float(h0) / float(infer_frame.shape[0])
        except Exception:
            infer_frame = frame
            sx, sy = 1.0, 1.0

        t0 = time.time()
        try:
            cage_results = self.cage_model(
                infer_frame,
                verbose=False,
                conf=self.conf_thresholds['cage'],
                max_det=10,
            )
        except Exception as e:
            print(f"cage_model 推理异常: {e}")
            return
        t1 = time.time()
        cage_idx = 0
        for cage_res in cage_results:
            try:
                cage_boxes = list(cage_res.boxes)
            except Exception as e:
                print(f"cage 后处理读取 boxes 异常: {e}")
                cage_boxes = []

            for c_box in cage_boxes:
                cage_idx += 1
                cx1, cy1, cx2, cy2 = map(float, c_box.xyxy[0].cpu().numpy())
                c_conf = float(c_box.conf[0])

                cx1, cy1, cx2, cy2 = cx1 * sx, cy1 * sy, cx2 * sx, cy2 * sy
                cx1, cy1, cx2, cy2 = int(cx1), int(cy1), int(cx2), int(cy2)

                h, w = frame.shape[:2]
                cx1, cy1 = max(0, cx1), max(0, cy1)
                cx2, cy2 = min(w, cx2), min(h, cy2)
                if cx2 - cx1 < 10 or cy2 - cy1 < 10:
                    continue

                track_id = self._assign_track_id([cx1, cy1, cx2, cy2], current_time)
                cage_key = self._stable_cage_key(cx1, cy1, cx2, cy2)

                cage_roi = frame[cy1:cy2, cx1:cx2]

                roi_h, roi_w = cage_roi.shape[:2]
                item_infer_roi = cage_roi
                rsx, rsy = 1.0, 1.0
                try:
                    target_roi_w = 640
                    if roi_w > target_roi_w:
                        rratio = float(target_roi_w) / float(roi_w)
                        item_infer_roi = cv2.resize(
                            cage_roi,
                            (target_roi_w, int(roi_h * rratio)),
                            interpolation=cv2.INTER_AREA,
                        )
                        rsx = float(roi_w) / float(item_infer_roi.shape[1])
                        rsy = float(roi_h) / float(item_infer_roi.shape[0])
                except Exception:
                    item_infer_roi = cage_roi
                    rsx, rsy = 1.0, 1.0

                t2 = time.time()
                try:
                    item_results = self.item_model(
                        item_infer_roi,
                        verbose=False,
                        conf=0.3,
                        max_det=50,
                    )
                except Exception as e:
                    print(f"item_model 推理异常: {e}")
                    continue
                t3 = time.time()

                items_in_cage = []
                c_cnt, e_cnt = 0, 0
                cage_id_text = None

                tr = self._cage_tracks.get(track_id)
                if tr and tr.get('cage_id'):
                    cage_id_text = str(tr.get('cage_id'))

                item_box_total = 0
                number_detections = []  # 收集所有number检测结果
                
                for item_res in item_results:
                    try:
                        names = item_res.names
                    except Exception as e:
                        print(f"item 后处理读取 names 异常: {e}")
                        names = {}

                    try:
                        item_boxes = list(item_res.boxes)
                    except Exception as e:
                        print(f"item 后处理读取 boxes 异常: {e}")
                        item_boxes = []

                    item_box_total += len(item_boxes)

                    for i_box in item_boxes:
                        cls_id = int(i_box.cls[0])
                        cls_name = names[cls_id]
                        conf = float(i_box.conf[0])

                        if cls_name in self.conf_thresholds and conf < self.conf_thresholds[cls_name]:
                            continue

                        if cls_name == 'chicken':
                            c_cnt += 1
                        elif cls_name == 'egg':
                            e_cnt += 1

                        rx1, ry1, rx2, ry2 = map(float, i_box.xyxy[0].cpu().numpy())
                        rx1, ry1, rx2, ry2 = rx1 * rsx, ry1 * rsy, rx2 * rsx, ry2 * rsy
                        rx1, ry1, rx2, ry2 = int(rx1), int(ry1), int(rx2), int(ry2)
                        ax1, ay1, ax2, ay2 = cx1 + rx1, cy1 + ry1, cx1 + rx2, cy1 + ry2

                        det_obj = {
                            'type': cls_name,
                            'bbox': [ax1, ay1, ax2, ay2],
                            'conf': conf,
                            'text': None,
                        }

                        if cls_name == 'number':
                            # 收集number检测结果，稍后统一处理
                            number_detections.append({
                                'bbox': [ax1, ay1, ax2, ay2],
                                'conf': conf,
                                'roi_img': frame[ay1:ay2, ax1:ax2].copy()
                            })
                        else:
                            items_in_cage.append(det_obj)

                # 统一处理该鸡笼区域内的所有number检测结果
                if number_detections:
                    # 按x坐标排序，确保数字顺序正确
                    number_detections.sort(key=lambda x: x['bbox'][0])
                    
                    # 合并所有number的ROI图像进行OCR识别
                    combined_text = self.process_multiple_numbers(number_detections, track_id, current_time)
                    
                    if combined_text:
                        norm_id = self.normalize_cage_id(combined_text)
                        cage_id_text = norm_id if norm_id else combined_text
                        if norm_id and tr is not None:
                            tr['cage_id'] = norm_id
                    
                    # 为每个number检测创建显示对象
                    for num_det in number_detections:
                        det_obj = {
                            'type': 'number',
                            'bbox': num_det['bbox'],
                            'conf': num_det['conf'],
                            'text': combined_text if combined_text else 'WAITING',
                        }
                        items_in_cage.append(det_obj)

                final_chicken, final_egg = c_cnt, e_cnt
                if cage_id_text:
                    if cage_id_text not in self.cage_registry:
                        self.cage_registry[cage_id_text] = {
                            'max_chicken': c_cnt,
                            'max_egg': e_cnt,
                            'last_seen': current_time,
                            'best_raw_img': frame.copy(),
                            'best_inf_img': None,
                        }
                        updated_cage_ids.append(cage_id_text)
                        self._last_valid_cage_ids.append(cage_id_text)
                    else:
                        reg = self.cage_registry[cage_id_text]
                        reg['last_seen'] = current_time
                        if c_cnt + e_cnt > reg['max_chicken'] + reg['max_egg']:
                            reg.update({
                                'max_chicken': max(c_cnt, reg['max_chicken']),
                                'max_egg': max(e_cnt, reg['max_egg']),
                                'best_raw_img': frame.copy(),
                            })
                            updated_cage_ids.append(cage_id_text)
                        final_chicken = reg['max_chicken']
                        final_egg = reg['max_egg']
                    frame_stats_map[cage_id_text] = {
                        'cage_id': cage_id_text,
                        'chickens': final_chicken,
                        'eggs': final_egg,
                    }

                for it in items_in_cage:
                    if it['type'] == 'number':
                        it['stats_payload'] = {'chickens': final_chicken, 'eggs': final_egg}

                detections.append({'type': 'cage', 'bbox': [cx1, cy1, cx2, cy2], 'conf': c_conf})
                detections.extend(items_in_cage)

        self.dispatch_ocr_tasks(current_time)

        t_draw0 = time.time()
        try:
            processed_frame = self.draw_detections(frame.copy(), detections)
        except Exception as e:
            print(f"draw_detections 失败: {e}")
            processed_frame = frame.copy()
        t_draw1 = time.time()

        for c_id in updated_cage_ids:
            if c_id in self.cage_registry:
                self.cage_registry[c_id]['best_inf_img'] = processed_frame.copy()

        self.check_and_save_disappeared(current_time)

        if current_time - self._last_stats_log_ts > self._stats_log_interval:
            self._last_stats_log_ts = current_time
            try:
                if frame_stats_map:
                    top = sorted(frame_stats_map.values(), key=lambda x: str(x.get('cage_id', '')))[:10]
                    msg = ' | '.join([f"ID:{d['cage_id']} C:{d['chickens']} E:{d['eggs']}" for d in top])
                    print(f"STATS {msg}")
                else:
                    print('STATS (none)')
            except Exception:
                pass

        # 单视频模式下显示处理后的帧，多视频模式下不显示（由run_multi_video_loop统一显示）
        if self.enable_local_display and not self.multi_video_mode:
            try:
                self.render_display_frame(processed_frame, list(frame_stats_map.values()))
            except Exception as e:
                self.enable_local_display = False
                print(f'本地显示失败，已自动关闭: {e}')

        # 单视频模式下上传处理后的帧，多视频模式下只上传合成画面（由run_multi_video_loop统一上传）
        if self._ws_enabled and not self.multi_video_mode:
            try:
                self.video_ws_queue.put_nowait(processed_frame)
            except queue.Full:
                pass

        # 在多视频模式下返回处理后的帧，单视频模式下不返回
        if self.multi_video_mode:
            return processed_frame

    def update_ocr_results(self, current_time):
        while not self.ocr_result_queue.empty():
            try:
                k, (t, c) = self.ocr_result_queue.get_nowait()
                self.ocr_cache[k] = {'text': t, 'conf': c, 'ts': current_time} if t else {
                    'text': None,
                    'conf': 0,
                    'ts': current_time - 1.5,
                }
                if t:
                    if k in self.pending_ocr:
                        del self.pending_ocr[k]
                    self.pending_ocr_set.discard(k)
            except Exception:
                break

        self.ocr_cache = {
            k: v for k, v in self.ocr_cache.items()
            if current_time - v.get('ts', 0.0) < self.ocr_cache_timeout
        }

    def _mark_ocr_pending(self, g_key, roi_img, current_time):
        if roi_img is None or roi_img.size == 0:
            return

        if g_key not in self.pending_ocr:
            self.pending_ocr[g_key] = {'img': roi_img, 'last_seen': current_time, 'last_req': 0.0}
        else:
            self.pending_ocr[g_key]['img'] = roi_img
            self.pending_ocr[g_key]['last_seen'] = current_time

        if g_key not in self.pending_ocr_set:
            self.pending_ocr_order.append(g_key)
            self.pending_ocr_set.add(g_key)

    def dispatch_ocr_tasks(self, current_time):
        if not self.pending_ocr_order:
            return

        if self.frame_count % self.ocr_skip_frames != 0:
            return

        max_dispatch = 12
        dispatched = 0
        max_checks = min(len(self.pending_ocr_order), 60)
        checks = 0
        while (
            checks < max_checks
            and dispatched < max_dispatch
            and (not self.ocr_queue.full())
            and self.pending_ocr_order
        ):
            checks += 1
            k = self.pending_ocr_order.popleft()
            if k not in self.pending_ocr:
                self.pending_ocr_set.discard(k)
                continue

            pending = self.pending_ocr[k]
            if current_time - pending['last_seen'] > self.ocr_pending_stale_timeout:
                del self.pending_ocr[k]
                self.pending_ocr_set.discard(k)
                continue

            cached = self.ocr_cache.get(k)
            if cached and cached.get('text'):
                del self.pending_ocr[k]
                self.pending_ocr_set.discard(k)
                continue

            if current_time - pending['last_req'] < self.ocr_request_interval:
                self.pending_ocr_order.append(k)
                continue

            try:
                self.ocr_queue.put_nowait((pending['img'], k))
                pending['last_req'] = current_time
                self.pending_ocr_order.append(k)
                dispatched += 1
            except queue.Full:
                self.pending_ocr_order.appendleft(k)
                break

    def process_multiple_numbers(self, number_detections, track_id, current_time):
        """
        处理同一鸡笼区域内的多个数字检测结果，合并识别为一个完整编号
        """
        if not number_detections:
            return None
            
        # 生成缓存键
        g_key = f"track_{track_id}_numbers"
        
        # 检查缓存
        cached = self.ocr_cache.get(g_key)
        cached_text = cached.get('text') if cached else None
        if cached_text:
            return cached_text
        
        # 合并所有数字的ROI图像
        combined_roi = self.combine_number_rois(number_detections)
        if combined_roi is None:
            return None
            
        # 标记OCR待处理
        self._mark_ocr_pending(g_key, combined_roi, current_time)
        return None

    def combine_number_rois(self, number_detections):
        """
        合并多个数字的ROI图像为一个连续的图像
        """
        if not number_detections:
            return None
            
        try:
            # 按x坐标排序
            number_detections.sort(key=lambda x: x['bbox'][0])
            
            # 计算合并后的图像尺寸
            max_height = 0
            total_width = 0
            padding = 10  # 数字间的间距
            
            for det in number_detections:
                roi = det['roi_img']
                h, w = roi.shape[:2]
                max_height = max(max_height, h)
                total_width += w + padding
            
            # 移除最后一个padding
            total_width -= padding
            
            # 创建合并后的图像
            combined = np.ones((max_height + 60, total_width + 60, 3), dtype=np.uint8) * 255
            
            # 逐个放置数字图像
            x_offset = 30
            for det in number_detections:
                roi = det['roi_img']
                h, w = roi.shape[:2]
                
                # 垂直居中放置
                y_offset = (max_height - h) // 2 + 30
                
                # 确保不越界
                if x_offset + w <= combined.shape[1] and y_offset + h <= combined.shape[0]:
                    combined[y_offset:y_offset+h, x_offset:x_offset+w] = roi
                
                x_offset += w + padding
                
            return combined
            
        except Exception as e:
            print(f"合并数字ROI失败: {e}")
            return None

    def normalize_cage_id(self, text):
        try:
            digits = re.sub(r'[^0-9]', '', str(text))
            if digits == '':
                return None
            n = int(digits)
            if 0 <= n < 100:
                return f"{n:02d}"
            return str(n)
        except Exception:
            return None

    def _is_id_jump(self, current_id: str) -> bool:
        if not current_id or not current_id.isdigit():
            return True
        cur_num = int(current_id)
        for prev in self._last_valid_cage_ids:
            if prev and prev.isdigit():
                prev_num = int(prev)
                if abs(cur_num - prev_num) <= self._id_jump_tolerance:
                    return False
        return True

    def check_and_save_disappeared(self, current_time):
        ids_to_remove = []
        for c_id, data in self.cage_registry.items():
            if current_time - data['last_seen'] > self.disappear_threshold:
                self.save_cage_data(c_id, data)
                ids_to_remove.append(c_id)
        for c_id in ids_to_remove:
            del self.cage_registry[c_id]

    def flush_registry(self):
        if not self.cage_registry:
            return
        for c_id, data in self.cage_registry.items():
            self.save_cage_data(c_id, data)
        self.cage_registry.clear()

    def save_cage_data(self, c_id, data):
        if self._is_id_jump(c_id):
            print(f"SKIP_UPLOAD due to ID jump: {c_id}")
            return

        ts_pretty = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ts_file = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{c_id}_{ts_file}"

        txt_path = os.path.join(self.save_dirs['count'], f"{base_name}.txt")
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(
                    f"ID: {c_id}\n"
                    f"Time: {ts_pretty}\n"
                    f"Max Chickens: {data['max_chicken']}\n"
                    f"Max Eggs: {data['max_egg']}\n"
                )
        except Exception:
            pass

        img_sub = os.path.join(self.save_dirs['image'], base_name)
        os.makedirs(img_sub, exist_ok=True)
        try:
            if data.get('best_raw_img') is not None:
                cv2.imwrite(os.path.join(img_sub, 'raw.jpg'), data['best_raw_img'])
            if data.get('best_inf_img') is not None:
                cv2.imwrite(os.path.join(img_sub, 'inference.jpg'), data['best_inf_img'])
        except Exception:
            pass

        if self._ws_enabled:
            with self._uploaded_cage_lock:
                if c_id in self._uploaded_cage_ids:
                    return
                self._uploaded_cage_ids.add(c_id)

            self.upload_ws_queue.put({
                'id': c_id,
                'time': ts_pretty,
                'chicken': int(data['max_chicken']),
                'egg': int(data['max_egg']),
                'raw_img': data.get('best_raw_img'),
                'inf_img': data.get('best_inf_img'),
            })

    def _disable_ws_upload(self, reason):
        if self._ws_disabled:
            return
        self._ws_enabled = False
        self._ws_disabled = True
        self.upload_enabled = False
        try:
            print(f"⚠️ WebSocket 上传已关闭: {reason} (将仅本地运行/显示)")
        except Exception:
            pass
        self._close_ws('_video_ws')
        self._close_ws('_upload_ws')
        try:
            while not self.video_ws_queue.empty():
                self.video_ws_queue.get_nowait()
                self.video_ws_queue.task_done()
        except Exception:
            pass
        try:
            while not self.upload_ws_queue.empty():
                self.upload_ws_queue.get_nowait()
                self.upload_ws_queue.task_done()
        except Exception:
            pass

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['type']

            if label == 'cage':
                color, thickness = (0, 0, 255), 2
            elif label == 'chicken':
                color, thickness = (0, 255, 255), 2
            elif label == 'egg':
                color, thickness = (255, 255, 255), 2
            elif label == 'number':
                color, thickness = (255, 50, 50), 2
            else:
                color, thickness = (0, 255, 0), 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if label == 'number':
                t_val = det.get('text')
                if t_val and t_val != 'WAITING':
                    t_str = f"No.{t_val}"
                elif t_val == 'WAITING':
                    t_str = 'Scanning...'
                else:
                    t_str = 'Unknown'

                stats = det.get('stats_payload')
                txt = f"{t_str} | C:{stats['chickens']} E:{stats['eggs']}" if stats else t_str

                (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_top = max(0, y1 - th - baseline - 6)
                cv2.rectangle(frame, (x1, y_top), (x1 + tw + 6, y1), (150, 0, 150) if stats else color, -1)
                cv2.putText(
                    frame,
                    txt,
                    (x1 + 3, y1 - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return frame

    def display_worker(self):
        while self.running and self.enable_local_display:
            try:
                item = self.display_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            frame, stats = item
            try:
                self.render_display_frame(frame, stats)
            finally:
                self.display_queue.task_done()

        self.close_display_window()

    def render_multi_display_frame(self, combined_frame):
        """渲染多画面合成显示"""
        if not self.enable_local_display:
            return

        if not self._window_initialized:
            try:
                cv2.namedWindow(self._display_window, cv2.WINDOW_NORMAL)
                # 调整窗口大小以适应四画面显示
                window_width = 1920
                window_height = 1080
                cv2.resizeWindow(self._display_window, window_width, window_height)
                self._window_initialized = True
                print(f"🖥️ 创建多画面显示窗口: {window_width}x{window_height}")
            except Exception as e:
                self.enable_local_display = False
                print(f'无法创建本地显示窗口: {e}')
                return

        # 检查合成画面的有效性
        if combined_frame is None:
            print("⚠️ 合成画面为None，跳过显示")
            return
        
        if not isinstance(combined_frame, np.ndarray):
            print(f"⚠️ 合成画面类型错误: {type(combined_frame)}，跳过显示")
            return
        
        if combined_frame.size == 0:
            print("⚠️ 合成画面为空，跳过显示")
            return
        
        # 显示合成画面
        try:
            cv2.imshow(self._display_window, combined_frame)
        except Exception as e:
            print(f"⚠️ 显示合成画面失败: {e}")
            return
        
        # 检查按键
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print('🔕 本地显示已关闭 (按下 q/ESC)')
            self.enable_local_display = False

    def render_display_frame(self, frame, stats):
        if not self.enable_local_display:
            return

        if not self._window_initialized:
            try:
                cv2.namedWindow(self._display_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._display_window, 960, 540)
                self._window_initialized = True
            except Exception as e:
                self.enable_local_display = False
                print(f'无法创建本地显示窗口: {e}')
                return

        display_img = frame.copy()
        stats = stats or []
        panel_rows = max(len(stats), 1)
        panel_height = 28 * panel_rows + 20
        panel_width = 360

        cv2.rectangle(
            display_img,
            (10, 10),
            (10 + panel_width, 10 + panel_height),
            (20, 20, 20),
            thickness=-1,
        )
        cv2.putText(
            display_img,
            'Cage Stats',
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if stats:
            for idx, info in enumerate(stats[:8]):
                text = f"ID:{info['cage_id']}  C:{info['chickens']}  E:{info['eggs']}"
                y = 70 + idx * 26
                cv2.putText(
                    display_img,
                    text,
                    (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        else:
            cv2.putText(
                display_img,
                '尚未识别到鸡笼',
                (24, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self._display_window, display_img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print('🔕 本地显示已关闭 (按下 q/ESC)')
            self.enable_local_display = False

    def close_display_window(self):
        if not self._window_initialized:
            return
        try:
            cv2.destroyWindow(self._display_window)
        except Exception:
            pass
        self._window_initialized = False

    def ocr_worker(self):
        while self.running:
            try:
                crop_img, bbox_key = self.ocr_queue.get(timeout=0.1)
                text, conf = self.process_image_ocr(crop_img)
                self.ocr_result_queue.put((bbox_key, (text, conf)))
                self.ocr_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                continue

    def process_image_ocr(self, img):
        if img is None or img.size == 0:
            return None, 0.0
        try:
            h, w = img.shape[:2]
            scale = 3.0 if h < 80 else 1.5
            if scale > 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            result = self.ocr.ocr(img, cls=True, det=True, rec=True)
            if not result or not result[0]:
                return None, 0.0

            best_text, best_conf = '', 0.0
            for line in result[0]:
                text_raw, conf = line[1]
                text_fix = (
                    text_raw.upper()
                    .replace('O', '0')
                    .replace('D', '0')
                    .replace('I', '1')
                    .replace('L', '1')
                    .replace('Z', '2')
                    .replace('S', '5')
                    .replace('B', '8')
                )
                digits = re.sub(r'[^0-9]', '', text_fix)
                if len(digits) > 0 and conf > 0.5 and conf > best_conf:
                    best_text, best_conf = digits, float(conf)

            return (best_text, best_conf) if best_text else (None, 0.0)
        except Exception:
            return None, 0.0

    def _connect_ws(self, ws_attr, timeout):
        if (not self.server_url) or (not self._ws_enabled):
            return None

        ws = getattr(self, ws_attr, None)
        if ws:
            try:
                if ws.connected:
                    return ws
            except Exception:
                pass

        try:
            ws = websocket.create_connection(self.server_url, timeout=timeout)
            setattr(self, ws_attr, ws)
            self._ws_fail_count = 0
            return ws
        except Exception:
            setattr(self, ws_attr, None)
            self._ws_fail_count += 1
            if self._ws_fail_count >= self._ws_fail_limit:
                self._disable_ws_upload('无法连接上位机 WebSocket')
            return None

    def _close_ws(self, ws_attr):
        ws = getattr(self, ws_attr, None)
        if not ws:
            return
        try:
            ws.close()
        except Exception:
            pass
        setattr(self, ws_attr, None)

    def _send_payload(self, payload, ws_attr, timeout, allow_drop=False):
        if (not self.server_url) or (not self._ws_enabled):
            return False

        data_str = json.dumps(payload)
        attempts = 3 if not allow_drop else 1

        for _ in range(attempts):
            ws = self._connect_ws(ws_attr, timeout)
            if not ws:
                if not allow_drop:
                    time.sleep(0.3)
                continue
            try:
                ws.send(data_str)
                return True
            except Exception:
                self._close_ws(ws_attr)
                self._ws_fail_count += 1
                if self._ws_fail_count >= self._ws_fail_limit:
                    self._disable_ws_upload('上位机 WebSocket 发送失败')
                if not allow_drop:
                    time.sleep(0.2)
        return False

    def image_to_base64(self, img_np, quality=80, resize_width=None):
        if img_np is None:
            return ''
        try:
            target_img = img_np
            if resize_width and img_np.shape[1] > resize_width:
                ratio = float(resize_width) / float(img_np.shape[1])
                dim = (int(resize_width), int(img_np.shape[0] * ratio))
                target_img = cv2.resize(img_np, dim, interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', target_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception:
            return ''

    def video_stream_worker(self):
        while self.running and self._ws_enabled:
            try:
                frame = self.video_ws_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                video_b64 = self.image_to_base64(
                    frame,
                    quality=self.video_stream_quality,
                    resize_width=self.video_stream_width,
                )
                payload = {
                    'type': 'realvideo',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'image_base64': video_b64,
                }
                self._send_payload(payload, '_video_ws', timeout=0.5, allow_drop=True)
            except Exception:
                pass
            finally:
                try:
                    self.video_ws_queue.task_done()
                except Exception:
                    pass

    def upload_worker(self):
        while self.running and self._ws_enabled:
            try:
                data = self.upload_ws_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                raw_b64 = self.image_to_base64(data.get('raw_img'), quality=70, resize_width=self.upload_image_width)
                inf_b64 = self.image_to_base64(data.get('inf_img'), quality=70, resize_width=self.upload_image_width)

                payload = {
                    'type': 'locationInfo',
                    'cage_id': str(data.get('id', '')),
                    'timestamp': str(data.get('time', '')),
                    'chicken_count': str(data.get('chicken', 0)),
                    'egg_count': str(data.get('egg', 0)),
                    'image_raw_base64': raw_b64,
                    'image_inf_base64': inf_b64,
                }

                sent = self._send_payload(payload, '_upload_ws', timeout=5.0, allow_drop=False)
                if not sent:
                    if self._ws_enabled:
                        self.upload_ws_queue.put(data)
                else:
                    try:
                        print(
                            f"UPLOAD cage_id={payload.get('cage_id')} chicken={payload.get('chicken_count')} egg={payload.get('egg_count')}"
                        )
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                try:
                    self.upload_ws_queue.task_done()
                except Exception:
                    pass


def main(args=None):
    # rclpy.init(args=args)  # 注释掉ROS2初始化
    node = None
    try:
        node = ChickenFarmNode()
        # rclpy.spin(node)  # 注释掉ROS2 spin
        # 由于使用了run_video_loop，程序会在这里阻塞直到视频播放完成
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        # if rclpy.ok():
        #     rclpy.shutdown()  # 注释掉ROS2关闭


if __name__ == '__main__':
    main()
