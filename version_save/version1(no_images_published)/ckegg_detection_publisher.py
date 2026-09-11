#!/usr/bin/env python3
"""
养鸡场智能检测ROS2节点 (适配 Jetson Orin NX)
功能：
1. 接收ROS图像话题或读取本地视频
2. 双YOLO模型检测 (鸡笼检测 -> 鸡/蛋/号码牌检测)
3. PaddleOCR 异步识别笼号
4. 统计逻辑 (记录最大数量，消失后上传)
5. WebSocket 实时推流与数据上传 (保留 test3.py 功能)
6. 发布 ROS 话题 (检测图像与统计信息)
"""

import sys
import os
import time
import json
import base64
import threading
import queue
import re
from datetime import datetime
import cv2
import numpy as np

# ROS2 相关库
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# 深度学习与图像处理库
# 添加anaconda环境路径(如果有需要，参考原文件)
sys.path.append('/home/jetson/anaconda3/envs/ckegg/lib/python3.10/site-packages/')
try:
    import torch
    from ultralytics import YOLO
    import paddle
    from paddleocr import PaddleOCR
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
    import websocket # pip install websocket-client
except ImportError as e:
    print(f"❌ 依赖库缺失: {e}")
    print("请确保安装: ultralytics, paddlepaddle-gpu, paddleocr, websocket-client, opencv-python")
    sys.exit(1)

# 添加anaconda环境路径(如果有需要，参考原文件)
#sys.path.append('/home/jetson/anaconda3/envs/ckegg/lib/python3.10/site-packages/')

class ChickenFarmNode(Node):
    def __init__(self):
        super().__init__('chicken_farm_node')
        
        self.get_logger().info("🚀 正在启动养鸡场检测节点 (Jetson Orin NX 版)...")

        # --- 1. 参数声明与获取 ---
        self.declare_parameters(
            namespace='',
            parameters=[
                ('source', '/home/jetson/ckegg_ws/src/ckegg_detect/videos/1.mp4'),                 # 视频源：'0'为相机，或者视频路径
                ('cage_model', '/home/jetson/ckegg_ws/src/ckegg_detect/models/cage_best.pt'),
                ('item_model', '/home/jetson/ckegg_ws/src/ckegg_detect/models/items_best.pt'),
                ('server_url', 'ws://192.168.43.50:8080/LeogEgg/websocket'),
                ('chicken_conf', 0.5),
                ('egg_conf', 0.45),
                ('number_conf', 0.6),
                ('video_loop', True),            # 视频文件是否循环播放
                ('disappear_timeout', 2.0),      # 目标消失多久后触发上传(秒)
                ('upload_enabled', True)         # 是否启用WebSocket上传
            ]
        )

        self.source = self.get_parameter('source').value
        self.model_paths = {
            'cage': self.get_parameter('cage_model').value,
            'item': self.get_parameter('item_model').value
        }
        self.conf_thresholds = {
            'chicken': self.get_parameter('chicken_conf').value,
            'egg': self.get_parameter('egg_conf').value,
            'number': self.get_parameter('number_conf').value,
            'cage': 0.5
        }
        self.server_url = self.get_parameter('server_url').value
        self.video_loop = self.get_parameter('video_loop').value
        self.disappear_threshold = self.get_parameter('disappear_timeout').value
        self.upload_enabled = self.get_parameter('upload_enabled').value

        # --- 2. ROS 发布者与工具初始化 ---
        self.bridge = CvBridge()
        # 发布处理后的图像 (用于Rviz显示)
        self.processed_img_pub = self.create_publisher(Image, '/chicken_farm/processed_image', 10)
        # 发布统计数据 JSON 字符串 (用于其他ROS节点)
        self.stats_pub = self.create_publisher(String, '/chicken_farm/statistics', 10)
        
        # --- 3. 核心变量初始化 ---
        self.frame_count = 0
        self.running = True
        self.cage_registry = {}  # 存储鸡笼状态 {id: {data}}
        
        # OCR 相关
        self.ocr_cache = {}
        self.ocr_cache_timeout = 2.0
        self.ocr_queue = queue.Queue(maxsize=5)
        self.ocr_result_queue = queue.Queue()
        self.ocr_skip_frames = 5  # 每隔几帧进行一次OCR尝试
        
        # WebSocket 队列
        self.video_ws_queue = queue.Queue(maxsize=4) # 实时流队列
        self.upload_ws_queue = queue.Queue()         # 数据上传队列
        
        # WebSocket 对象
        self._video_ws = None
        self._upload_ws = None
        
        # 画质配置
        self.video_stream_quality = 55
        self.video_stream_width = 640
        self.upload_image_width = 800

        # 数据存储路径
        self.save_dirs = {
            'count': "/home/jetson/ckegg_ws/src/ckegg_detect/data/count_data",
            'image': "/home/jetson/ckegg_ws/src/ckegg_detect/data/image_data",
            'debug': "/home/jetson/ckegg_ws/src/ckegg_detect/data/debug_ocr"
        }
        for d in self.save_dirs.values():
            os.makedirs(d, exist_ok=True)

        # --- 4. 模型与资源加载 ---
        self.init_models()
        self.setup_font()
        
        # --- 5. 启动后台线程 ---
        self.start_threads()

        # --- 6. 输入源处理 (相机/视频/ROS话题) ---
        # 判断是数字(相机ID)还是文件路径
        is_camera_index = self.source.isdigit()
        is_video_file = os.path.exists(self.source)

        if is_camera_index or is_video_file:
            self.get_logger().info(f"🎥 使用 OpenCV 读取输入源: {self.source}")
            if is_camera_index:
                self.cap = cv2.VideoCapture(int(self.source))
                # 尝试设置高帧率 (Jetson 上通常需要)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                self.get_logger().error("❌ 无法打开视频源")
                sys.exit(1)
                
            # 使用定时器读取帧
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps): fps = 30.0
            self.timer = self.create_timer(1.0/fps, self.timer_callback)
        else:
            self.get_logger().info("📡 等待订阅 ROS 图像话题: /camera/color/image_raw")
            self.cap = None
            self.sub = self.create_subscription(
                Image, 
                '/camera/color/image_raw', 
                self.image_callback, 
                10
            )

    def init_models(self):
        """加载 YOLO 和 OCR 模型，适配 Jetson CUDA"""
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"🔄 加载模型中 (设备: {device})...")
        
        try:
            # 加载 YOLO
            self.cage_model = YOLO(self.model_paths['cage'])
            self.cage_model.to(device)
            self.item_model = YOLO(self.model_paths['item'])
            self.item_model.to(device)
            self.get_logger().info("✅ YOLO 模型加载成功")

            # 加载 PaddleOCR
            if paddle.device.is_compiled_with_cuda():
                paddle.device.set_device('gpu')
            
            # 抑制 PaddleOCR 日志
            import logging
            logging.getLogger("ppocr").setLevel(logging.WARNING)
            
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=(device=='cuda:0'), show_log=False)
            self.get_logger().info("✅ PaddleOCR 初始化成功")
            
        except Exception as e:
            self.get_logger().error(f"❌ 模型加载失败: {e}")
            sys.exit(1)

    def setup_font(self):
        """加载字体 (Linux环境)"""
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Ubuntu常用中文
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/arphic/ukai.ttc"
        ]
        self.font = ImageFont.load_default()
        for path in font_paths:
            if os.path.exists(path):
                try:
                    self.font = ImageFont.truetype(path, 20)
                    self.get_logger().info(f"✅ 加载字体: {path}")
                    break
                except: continue

    def start_threads(self):
        """启动 OCR 和 WebSocket 线程"""
        # OCR 线程
        t_ocr = threading.Thread(target=self.ocr_worker, daemon=True)
        t_ocr.start()
        
        if self.upload_enabled and self.server_url:
            # 视频流推流线程
            t_video = threading.Thread(target=self.video_stream_worker, daemon=True)
            t_video.start()
            # 数据上传线程
            t_upload = threading.Thread(target=self.upload_worker, daemon=True)
            t_upload.start()
            self.get_logger().info(f"🌐 WebSocket 服务已启动: {self.server_url}")
        else:
            self.get_logger().warn("⚠️ WebSocket 上传未启用")

    # ------------------ 输入回调 ------------------

    def timer_callback(self):
        """处理本地视频/相机帧"""
        if self.cap is None: return
        ret, frame = self.cap.read()
        if not ret:
            if self.video_loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            else:
                self.get_logger().info("视频播放结束")
                self.cap.release()
                self.destroy_node()
                return
        self.process_frame(frame)

    def image_callback(self, msg):
        """处理 ROS 图像帧"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.process_frame(cv_image)
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")

    # ------------------ 核心处理逻辑 ------------------

    def process_frame(self, frame):
        """主处理流程：检测、逻辑、绘制、发布"""
        self.frame_count += 1
        current_time = time.time()
        
        # 1. 更新 OCR 缓存
        self.update_ocr_results(current_time)
        
        detections = []
        updated_cage_ids = []
        
        # 2. 第一级检测：鸡笼 (Cage)
        # 使用较低置信度确保召回
        cage_results = self.cage_model(frame, verbose=False, conf=0.5)
        
        for cage_res in cage_results:
            for c_box in cage_res.boxes:
                # 获取鸡笼坐标
                cx1, cy1, cx2, cy2 = map(int, c_box.xyxy[0].cpu().numpy())
                c_conf = float(c_box.conf[0])
                
                # 边界保护
                h, w = frame.shape[:2]
                cx1, cy1 = max(0, cx1), max(0, cy1)
                cx2, cy2 = min(w, cx2), min(h, cy2)
                
                if cx2 - cx1 < 10 or cy2 - cy1 < 10: continue # 过滤过小目标
                
                # 裁剪鸡笼区域进行二级检测
                cage_roi = frame[cy1:cy2, cx1:cx2]
                
                # 3. 第二级检测：内容物 (Item)
                item_results = self.item_model(cage_roi, verbose=False, conf=0.3)
                
                items_in_cage = []
                c_cnt, e_cnt = 0, 0
                cage_id_text = None
                
                for item_res in item_results:
                    names = item_res.names
                    for i_box in item_res.boxes:
                        cls_id = int(i_box.cls[0])
                        cls_name = names[cls_id]
                        conf = float(i_box.conf[0])
                        
                        # 阈值过滤
                        if cls_name in self.conf_thresholds and conf < self.conf_thresholds[cls_name]:
                            continue
                            
                        # 统计数量
                        if cls_name == 'chicken': c_cnt += 1
                        elif cls_name == 'egg': e_cnt += 1
                        
                        # 坐标映射 (ROI -> 全图)
                        rx1, ry1, rx2, ry2 = map(int, i_box.xyxy[0].cpu().numpy())
                        ax1, ay1, ax2, ay2 = cx1+rx1, cy1+ry1, cx1+rx2, cy1+ry2
                        
                        det_obj = {
                            'type': cls_name,
                            'bbox': [ax1, ay1, ax2, ay2],
                            'conf': conf,
                            'text': None
                        }
                        
                        # 4. 号码牌 OCR 处理
                        if cls_name == 'number':
                            # 生成网格对齐的 Key，增加缓存命中率
                            g_key = f"{ax1//30}_{ay1//30}_{ax2//30}_{ay2//30}"
                            
                            if g_key in self.ocr_cache:
                                det_obj['text'] = self.ocr_cache[g_key]['text']
                                if det_obj['text']:
                                    cage_id_text = det_obj['text']
                            else:
                                det_obj['text'] = "WAITING"
                                # 只有特定帧且队列未满时才提交 OCR 任务
                                if self.frame_count % self.ocr_skip_frames == 0 and not self.ocr_queue.full():
                                    roi_img = frame[ay1:ay2, ax1:ax2].copy()
                                    try:
                                        self.ocr_queue.put_nowait((roi_img, g_key))
                                    except queue.Full: pass
                        
                        items_in_cage.append(det_obj)
                
                # 5. 统计逻辑与注册表更新
                final_chicken, final_egg = c_cnt, e_cnt
                
                if cage_id_text:
                    if cage_id_text not in self.cage_registry:
                        # 新发现的鸡笼
                        self.cage_registry[cage_id_text] = {
                            'max_chicken': c_cnt,
                            'max_egg': e_cnt,
                            'last_seen': current_time,
                            'best_raw_img': frame.copy(),
                            'best_inf_img': None # 推理图稍后生成
                        }
                        updated_cage_ids.append(cage_id_text)
                    else:
                        # 已存在的鸡笼，更新统计
                        reg = self.cage_registry[cage_id_text]
                        reg['last_seen'] = current_time
                        
                        # 如果当前数量更多，更新记录
                        if c_cnt + e_cnt > reg['max_chicken'] + reg['max_egg']:
                            reg.update({
                                'max_chicken': max(c_cnt, reg['max_chicken']),
                                'max_egg': max(e_cnt, reg['max_egg']),
                                'best_raw_img': frame.copy()
                            })
                            updated_cage_ids.append(cage_id_text)
                        
                        final_chicken = reg['max_chicken']
                        final_egg = reg['max_egg']
                
                # 将统计数据附加到号码牌检测结果上，以便绘制
                for it in items_in_cage:
                    if it['type'] == 'number':
                        it['stats_payload'] = {'chickens': final_chicken, 'eggs': final_egg}
                
                detections.append({'type': 'cage', 'bbox': [cx1, cy1, cx2, cy2], 'conf': c_conf})
                detections.extend(items_in_cage)

        # 6. 绘制结果
        processed_frame = self.draw_detections(frame.copy(), detections)
        
        # 7. 更新最佳推理图
        for c_id in updated_cage_ids:
            if c_id in self.cage_registry:
                self.cage_registry[c_id]['best_inf_img'] = processed_frame.copy()
        
        # 8. 检查消失的目标并上传/保存
        self.check_and_save_disappeared(current_time)

        # 9. 发布结果
        self.publish_ros_data(processed_frame)
        
        # 10. WebSocket 视频推流 (放入队列)
        if self.upload_enabled and self.server_url:
            try:
                self.video_ws_queue.put_nowait(processed_frame)
            except queue.Full:
                pass # 丢帧

    # ------------------ 辅助逻辑函数 ------------------

    def update_ocr_results(self, current_time):
        """处理 OCR 结果队列并清理过期缓存"""
        while not self.ocr_result_queue.empty():
            try:
                k, (t, c) = self.ocr_result_queue.get_nowait()
                self.ocr_cache[k] = {
                    'text': t, 
                    'conf': c, 
                    'ts': current_time
                } if t else {
                    'text': None, 
                    'conf': 0, 
                    'ts': current_time - 1.5 # 失败结果更快过期
                }
            except: break
        
        # 清理过期缓存
        self.ocr_cache = {
            k:v for k,v in self.ocr_cache.items() 
            if current_time - v['ts'] < self.ocr_cache_timeout
        }

    def check_and_save_disappeared(self, current_time):
        """检查鸡笼是否移出画面，如果是，保存数据并上传"""
        ids_to_remove = []
        for c_id, data in self.cage_registry.items():
            if current_time - data['last_seen'] > self.disappear_threshold:
                self.save_cage_data(c_id, data)
                ids_to_remove.append(c_id)
        
        for c_id in ids_to_remove:
            del self.cage_registry[c_id]

    def save_cage_data(self, c_id, data):
        """保存数据到本地并触发上传"""
        ts_pretty = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{c_id}_{ts_file}"
        
        self.get_logger().info(f"📊 统计完成: 笼号 {c_id}, 鸡 {data['max_chicken']}, 蛋 {data['max_egg']}")

        # 1. 本地文本保存
        txt_path = os.path.join(self.save_dirs['count'], f"{base_name}.txt")
        try:
            with open(txt_path, "w", encoding='utf-8') as f:
                f.write(f"ID: {c_id}\nTime: {ts_pretty}\nMax Chickens: {data['max_chicken']}\nMax Eggs: {data['max_egg']}\n")
        except Exception as e:
            self.get_logger().error(f"本地保存失败: {e}")

        # 2. 本地图片保存
        img_sub = os.path.join(self.save_dirs['image'], base_name)
        os.makedirs(img_sub, exist_ok=True)
        try:
            if data['best_raw_img'] is not None: 
                cv2.imwrite(os.path.join(img_sub, "raw.jpg"), data['best_raw_img'])
            if data['best_inf_img'] is not None: 
                cv2.imwrite(os.path.join(img_sub, "inference.jpg"), data['best_inf_img'])
        except Exception as e:
            self.get_logger().error(f"图片保存失败: {e}")

        # 3. 发布 ROS 统计消息
        stats_msg = String()
        stats_msg.data = json.dumps({
            "id": c_id,
            "timestamp": ts_pretty,
            "chicken_count": data['max_chicken'],
            "egg_count": data['max_egg']
        })
        self.stats_pub.publish(stats_msg)

        # 4. WebSocket 上传队列
        if self.upload_enabled and self.server_url:
            self.upload_ws_queue.put({
                'id': c_id, 'time': ts_pretty, 
                'chicken': data['max_chicken'], 'egg': data['max_egg'], 
                'raw_img': data['best_raw_img'], 'inf_img': data['best_inf_img']
            })

    def draw_detections(self, frame, detections):
        """绘制边界框和中文信息"""
        img_pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['type']
            
            # 颜色与线宽定义
            if label == 'cage':
                color, width = (0, 0, 255), 2
            elif label == 'chicken':
                color, width = (0, 255, 255), 3
            elif label == 'egg':
                color, width = (255, 255, 255), 3
            elif label == 'number':
                color, width = (255, 50, 50), 3
            else:
                color, width = (0, 255, 0), 2

            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            
            # 绘制文字 (仅针对 number 绘制统计信息)
            if label == 'number':
                t_val = det['text']
                if t_val and t_val != "WAITING":
                    t_str = f"No.{t_val}"
                elif t_val == "WAITING":
                    t_str = "Scanning..."
                else:
                    t_str = "Unknown"
                
                stats = det.get('stats_payload')
                txt = f"{t_str} | C:{stats['chickens']} E:{stats['eggs']}" if stats else t_str
                bg_color = (150, 0, 150) if stats else color
                
                # 绘制文字背景
                bbox = draw.textbbox((0, 0), txt, font=self.font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                draw.rectangle([x1, y1 - text_h - 6, x1 + text_w + 4, y1], fill=bg_color)
                draw.text((x1 + 2, y1 - text_h - 6), txt, font=self.font, fill=(255, 255, 255))
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def publish_ros_data(self, frame):
        """发布 ROS 图像"""
        try:
            # 缩放图像以减少带宽 (可选)
            # resized = cv2.resize(frame, (1280, 720))
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.processed_img_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"ROS 图像发布失败: {e}")

    # ------------------ 线程 Worker 函数 ------------------

    def ocr_worker(self):
        """OCR 异步处理线程"""
        while self.running:
            try:
                task = self.ocr_queue.get(timeout=0.1)
                crop_img, bbox_key = task
                
                # OCR 识别
                text, conf = self.process_image_ocr(crop_img)
                self.ocr_result_queue.put((bbox_key, (text, conf)))
                self.ocr_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"OCR 线程错误: {e}")

    def process_image_ocr(self, img):
        """OCR 图像预处理与推理"""
        if img is None or img.size == 0: return None, 0.0
        try:
            h, w = img.shape[:2]
            scale = 3.0 if h < 80 else 1.5
            if scale > 1.0: 
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # 增加白边
            pad = 30
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            
            result = self.ocr.ocr(img, cls=True, det=True, rec=True)
            if not result or not result[0]: return None, 0.0
            
            best_text, best_conf = "", 0.0
            for line in result[0]:
                text_raw, conf = line[1]
                # 字符修正逻辑
                text_fix = text_raw.upper().replace('O','0').replace('D','0').replace('I','1')\
                                         .replace('L','1').replace('Z','2').replace('S','5').replace('B','8')
                digits = re.sub(r'[^0-9]', '', text_fix)
                if len(digits) > 0 and conf > 0.5 and conf > best_conf:
                    best_text, best_conf = digits, conf
            return (best_text, best_conf) if best_text else (None, 0.0)
        except: return None, 0.0

    # ------------------ WebSocket 通信 ------------------

    def _connect_ws(self, ws_attr):
        """建立 WebSocket 连接"""
        ws = getattr(self, ws_attr, None)
        if ws and ws.connected: return ws
        
        try:
            ws = websocket.create_connection(self.server_url, timeout=1.0)
            setattr(self, ws_attr, ws)
            return ws
        except Exception:
            setattr(self, ws_attr, None)
            return None

    def _close_ws(self, ws_attr):
        ws = getattr(self, ws_attr, None)
        if ws:
            try: ws.close()
            except: pass
        setattr(self, ws_attr, None)

    def image_to_base64(self, img_np, quality=80, resize_width=None):
        """图像转 Base64"""
        if img_np is None: return ""
        try:
            target_img = img_np
            if resize_width and img_np.shape[1] > resize_width:
                ratio = resize_width / img_np.shape[1]
                dim = (resize_width, int(img_np.shape[0] * ratio))
                target_img = cv2.resize(img_np, dim, interpolation=cv2.INTER_AREA)
            
            _, buffer = cv2.imencode('.jpg', target_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return base64.b64encode(buffer).decode('utf-8')
        except: return ""

    def video_stream_worker(self):
        """WebSocket 视频流推送"""
        while self.running:
            try:
                frame = self.video_ws_queue.get(timeout=0.1)
                
                video_b64 = self.image_to_base64(
                    frame,
                    quality=self.video_stream_quality,
                    resize_width=self.video_stream_width
                )
                
                payload = {
                    "type": "realvideo",
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "image_base64": video_b64
                }
                
                ws = self._connect_ws('_video_ws')
                if ws:
                    try:
                        ws.send(json.dumps(payload))
                    except:
                        self._close_ws('_video_ws')
            except queue.Empty: pass
            except Exception as e:
                # 避免频繁打印连接错误
                pass

    def upload_worker(self):
        """WebSocket 数据上传"""
        while self.running:
            try:
                data = self.upload_ws_queue.get(timeout=1.0)
                
                raw_b64 = self.image_to_base64(data['raw_img'], quality=70, resize_width=self.upload_image_width)
                inf_b64 = self.image_to_base64(data['inf_img'], quality=70, resize_width=self.upload_image_width)
                
                payload = {
                    "type": "locationInfo", 
                    "cage_id": str(data['id']),
                    "timestamp": str(data['time']),
                    "chicken_count": str(data['chicken']),
                    "egg_count": str(data['egg']),
                    "image_raw_base64": raw_b64,
                    "image_inf_base64": inf_b64
                }
                
                # 重试逻辑
                success = False
                for _ in range(3):
                    ws = self._connect_ws('_upload_ws')
                    if ws:
                        try:
                            ws.send(json.dumps(payload))
                            success = True
                            self.get_logger().info(f"🚀 上传成功: 笼号 {data['id']}")
                            break
                        except:
                            self._close_ws('_upload_ws')
                            time.sleep(0.5)
                
                if not success:
                    self.get_logger().error(f"❌ 上传失败: 笼号 {data['id']}")
                    
                self.upload_ws_queue.task_done()
            except queue.Empty: pass

    def destroy_node(self):
        """清理资源"""
        self.running = False
        if self.cap: self.cap.release()
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
        print("\n🛑 节点正在停止...")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
    finally:
        if node: node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
