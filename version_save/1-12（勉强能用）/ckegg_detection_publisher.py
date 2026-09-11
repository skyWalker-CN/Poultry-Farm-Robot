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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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


class ChickenFarmNode(Node):
    def __init__(self):
        super().__init__('ckegg_detection_publisher')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('source', '/home/jetson/ckegg_ws/src/ckegg_detect/videos/1.mp4'),
                ('image_topic', '/camera/color/image_raw'),
                ('cage_model', '/home/jetson/ckegg_ws/src/ckegg_detect/models/cage_best.pt'),
                ('item_model', '/home/jetson/ckegg_ws/src/ckegg_detect/models/items_best.pt'),
                ('server_url', 'ws://192.168.43.50:8080/LeogEgg/websocket'),
                ('upload_enabled', True),
                ('video_loop', False),
                ('disappear_timeout', 2.0),
                ('chicken_conf', 0.5),
                ('egg_conf', 0.45),
                ('number_conf', 0.6),
                ('video_stream_quality', 55),
                ('video_stream_width', 640),
                ('upload_image_width', 800),
                ('save_root', ''),
                ('local_display', True),
            ],
        )

        self.source = str(self.get_parameter('source').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.server_url = str(self.get_parameter('server_url').value)
        self.upload_enabled = bool(self.get_parameter('upload_enabled').value)
        self.video_loop = bool(self.get_parameter('video_loop').value)
        self.disappear_threshold = float(self.get_parameter('disappear_timeout').value)

        self._ws_enabled = bool(self.upload_enabled and self.server_url)
        self._ws_disabled = False

        self.conf_thresholds = {
            'chicken': float(self.get_parameter('chicken_conf').value),
            'egg': float(self.get_parameter('egg_conf').value),
            'number': float(self.get_parameter('number_conf').value),
            'cage': 0.5,
        }

        self.video_stream_quality = int(self.get_parameter('video_stream_quality').value)
        self.video_stream_width = int(self.get_parameter('video_stream_width').value)
        self.upload_image_width = int(self.get_parameter('upload_image_width').value)
        self.enable_local_display = bool(self.get_parameter('local_display').value)
        self._display_window = 'ChickenFarm Detection'
        self._window_initialized = False
        self.display_queue = None
        self.display_thread = None

        cage_model = str(self.get_parameter('cage_model').value)
        item_model = str(self.get_parameter('item_model').value)
        self.model_paths = {'cage': cage_model, 'item': item_model}

        self.bridge = CvBridge()
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

        self.ocr_cache = {}
        self.ocr_cache_timeout = 2.0
        self.ocr_queue = queue.Queue(maxsize=50)
        self.ocr_result_queue = queue.Queue()
        self.ocr_skip_frames = 5
        self.pending_ocr = {}
        self.pending_ocr_order = deque()
        self.pending_ocr_set = set()
        self.ocr_request_interval = 0.6
        self.ocr_pending_stale_timeout = 6.0

        self.video_ws_queue = queue.Queue(maxsize=4)
        self.upload_ws_queue = queue.Queue()
        self._video_ws = None
        self._upload_ws = None
        self._ws_fail_count = 0
        self._ws_last_offline_log = 0.0

        if self.get_parameter('save_root').value:
            save_root = str(self.get_parameter('save_root').value)
        else:
            save_root = os.path.join(os.path.expanduser('~'), 'ckegg_data')

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

        self._log_runtime_env()

        self.get_logger().info('✅ ckegg 检测节点已启动')

    def _log_runtime_env(self):
        try:
            disp = os.environ.get('DISPLAY', '')
            wayland = os.environ.get('WAYLAND_DISPLAY', '')
            sess = os.environ.get('XDG_SESSION_TYPE', '')
            self.get_logger().info(f"RuntimeEnv DISPLAY='{disp}' WAYLAND_DISPLAY='{wayland}' XDG_SESSION_TYPE='{sess}'")
            if self.enable_local_display and (not disp and not wayland):
                self.enable_local_display = False
                self.get_logger().warning('⚠️ 检测到无 DISPLAY/WAYLAND 环境，已自动关闭本地窗口显示。将仅在终端输出统计信息。')
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
        is_camera_index = self.source.isdigit() if self.source else False
        is_video_file = os.path.exists(self.source) if self.source else False

        self.cap = None
        self.timer = None
        self.sub = None

        if is_camera_index or is_video_file:
            if is_camera_index:
                self.get_logger().info(f"🎥 使用 OpenCV 摄像头 index={self.source}")
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                self.get_logger().info(f"🎥 使用 OpenCV 视频文件: {self.source}")
                self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                raise RuntimeError('无法打开视频源')

            fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0 or np.isnan(fps):
                fps = 30.0
            self.timer = self.create_timer(1.0 / fps, self.timer_callback)
            return

        self.get_logger().info(f"📡 订阅 ROS 图像话题: {self.image_topic}")
        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)

    def init_models(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"🔄 加载模型中 (设备: {device})")

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
            self.get_logger().info(f"🌐 WebSocket 上传启用: {self.server_url}")

        if self.enable_local_display:
            self.get_logger().info('🖥️ 本地显示启用')

    def timer_callback(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info('⚠️ 视频流结束，正在结算剩余数据')
            self.flush_registry()

            if self.video_loop:
                self.get_logger().info('🔄 视频循环播放')
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.cage_registry.clear()
                return

            self.get_logger().info('🛑 播放结束，准备退出')
            try:
                self.cap.release()
            except Exception:
                pass
            self.destroy_node()
            sys.exit(0)

        self._last_frame_ts = time.time()
        self.process_frame(frame)

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_frame(cv_image)
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")

    def process_frame(self, frame):
        if frame is None:
            return

        self.frame_count += 1
        current_time = time.time()

        if current_time - self._last_process_log_ts > self._process_log_interval:
            h, w = frame.shape[:2]
            self.get_logger().info(f"Processing frame={self.frame_count} size={w}x{h}")
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

        self.get_logger().info('Stage: cage_model inference...')
        t0 = time.time()
        try:
            cage_results = self.cage_model(
                infer_frame,
                verbose=False,
                conf=self.conf_thresholds['cage'],
                max_det=10,
            )
        except Exception as e:
            self.get_logger().error(f"cage_model 推理异常: {e}")
            return
        t1 = time.time()
        self.get_logger().info(f"Stage: cage_model done ({t1 - t0:.2f}s)")
        cage_idx = 0
        for cage_res in cage_results:
            try:
                cage_boxes = list(cage_res.boxes)
            except Exception as e:
                self.get_logger().error(f"cage 后处理读取 boxes 异常: {e}")
                cage_boxes = []

            self.get_logger().info(f"Stage: cage postprocess boxes={len(cage_boxes)}")
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

                self.get_logger().info('Stage: item_model inference...')
                t2 = time.time()
                try:
                    item_results = self.item_model(
                        item_infer_roi,
                        verbose=False,
                        conf=0.3,
                        max_det=50,
                    )
                except Exception as e:
                    self.get_logger().error(f"item_model 推理异常: {e}")
                    continue
                t3 = time.time()
                self.get_logger().info(f"Stage: item_model done ({t3 - t2:.2f}s)")

                self.get_logger().info(f"Stage: item postprocess begin (cage_idx={cage_idx})")

                items_in_cage = []
                c_cnt, e_cnt = 0, 0
                cage_id_text = None

                item_box_total = 0
                for item_res in item_results:
                    try:
                        names = item_res.names
                    except Exception as e:
                        self.get_logger().error(f"item 后处理读取 names 异常: {e}")
                        names = {}

                    try:
                        item_boxes = list(item_res.boxes)
                    except Exception as e:
                        self.get_logger().error(f"item 后处理读取 boxes 异常: {e}")
                        item_boxes = []

                    item_box_total += len(item_boxes)
                    self.get_logger().info(f"Stage: item postprocess boxes={len(item_boxes)} (cage_idx={cage_idx})")

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
                            g_key = f"{ax1//30}_{ay1//30}_{ax2//30}_{ay2//30}"
                            cached = self.ocr_cache.get(g_key)
                            cached_text = cached.get('text') if cached else None

                            if cached_text:
                                norm_id = self.normalize_cage_id(cached_text)
                                det_obj['text'] = norm_id if norm_id else cached_text
                                if norm_id:
                                    cage_id_text = norm_id
                            else:
                                det_obj['text'] = 'WAITING'
                                roi_img = frame[ay1:ay2, ax1:ax2].copy()
                                self._mark_ocr_pending(g_key, roi_img, current_time)

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

                self.get_logger().info(
                    f"Stage: item postprocess end (cage_idx={cage_idx}) chickens={c_cnt} eggs={e_cnt}"
                )

        self.dispatch_ocr_tasks(current_time)

        self.get_logger().info('Stage: draw_detections...')
        t_draw0 = time.time()
        try:
            processed_frame = self.draw_detections(frame.copy(), detections)
        except Exception as e:
            self.get_logger().error(f"draw_detections 失败: {e}")
            processed_frame = frame.copy()
        t_draw1 = time.time()
        self.get_logger().info(f"Stage: draw_detections done ({t_draw1 - t_draw0:.2f}s)")

        for c_id in updated_cage_ids:
            if c_id in self.cage_registry:
                self.cage_registry[c_id]['best_inf_img'] = processed_frame.copy()

        self.check_and_save_disappeared(current_time)

        if current_time - self._last_stats_log_ts > self._stats_log_interval:
            if frame_stats_map:
                brief = ', '.join([f"{v['cage_id']}:C{v['chickens']}/E{v['eggs']}" for v in list(frame_stats_map.values())[:6]])
                self.get_logger().info(f"Stats: {brief}")
            else:
                self.get_logger().info('Stats: (none)')
            self._last_stats_log_ts = current_time

        if self.enable_local_display:
            try:
                self.get_logger().info('Stage: local_display render...')
                self.render_display_frame(processed_frame, list(frame_stats_map.values()))
            except Exception as e:
                self.enable_local_display = False
                self.get_logger().error(f'本地显示失败，已自动关闭: {e}')

        if self._ws_enabled:
            try:
                self.video_ws_queue.put_nowait(processed_frame)
            except queue.Full:
                pass

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
            self.get_logger().warning(f"⚠️ WebSocket 上传已关闭: {reason} (将仅本地运行/显示)")
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
                self.get_logger().error(f'无法创建本地显示窗口: {e}')
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
            self.get_logger().info('🔕 本地显示已关闭 (按下 q/ESC)')
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
            if self._ws_fail_count >= 2:
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
                if self._ws_fail_count >= 2:
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
            except Exception:
                pass
            finally:
                try:
                    self.upload_ws_queue.task_done()
                except Exception:
                    pass

    def destroy_node(self):
        self.running = False

        try:
            self.flush_registry()
        except Exception:
            pass

        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

        self.close_display_window()

        self._close_ws('_video_ws')
        self._close_ws('_upload_ws')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ChickenFarmNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
