import sys
import cv2
import time
import argparse
import os
import numpy as np
import threading
from queue import Queue, Empty, Full
import re
from datetime import datetime
import logging
import shutil
import json
import base64

# --- 网络库检查 ---
try:
    import websocket # pip install websocket-client
except ImportError:
    print("❌ 缺少 websocket-client 库")
    print("请运行: pip install websocket-client")
    sys.exit(1)

# 深度学习相关库
try:
    import torch
    from ultralytics import YOLO
    import paddle
    from paddleocr import PaddleOCR
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
    print("✅ 检测模型库加载成功")
except ImportError as e:
    print(f"⚠️  检测模型库导入失败: {e}")
    print("请先运行: pip install -r requirements.txt")
    sys.exit(1)

class ChickenFarmSystem:
    def __init__(self, args):
        self.args = args
        self.running = True
        
        # 阈值设置
        self.conf_thresholds = {
            'chicken': args.chicken_conf,
            'egg': args.egg_conf,
            'number': args.number_conf,
            'cage': 0.5 
        }
        
        self.cap = None
        
        # OCR 异步处理
        self.ocr_cache = {} 
        self.ocr_cache_timeout = 2.0 
        self.ocr_queue = Queue(maxsize=5)
        self.ocr_result_queue = Queue()
        self.ocr_threads = []
        self.frame_count = 0
        self.ocr_skip_frames = 5 
        
        # --- 数据持久化相关 ---
        self.count_data_folder = "count_data"
        self.image_data_folder = "image_data"
        self.debug_ocr_folder = "debug_ocr_crops"
        
        os.makedirs(self.count_data_folder, exist_ok=True)
        os.makedirs(self.image_data_folder, exist_ok=True)
        os.makedirs(self.debug_ocr_folder, exist_ok=True)
        
        # --- 网络上传相关 (WebSocket) ---
        self.upload_queue = Queue() # 统计数据队列
        
        # [修改点1] 队列大小设为4，适配40帧的节奏
        self.video_queue = Queue(maxsize=4)
        
        # WebSocket 对象存储
        self.server_url = self.args.server_url
        self._upload_ws = None
        self._video_ws = None
        
        # [修改点2] 画质配置: 40帧是中间值，质量设为55 (0-100)
        # 既比30帧版本清晰，又比60帧版本节省一点带宽
        self.video_stream_quality = 55  
        self.video_stream_width = 640   
        self.upload_image_width = 800
        
        if self.server_url:
            print(f"🌐 WebSocket上传已启用: 目标 {self.server_url}")
            print(f"📺 视频流配置: 目标40FPS, 宽度={self.video_stream_width}px, 质量={self.video_stream_quality}")
        
        # 全局鸡笼状态注册表
        self.cage_registry = {}
        self.disappear_threshold = 2.0 
        
        # 初始化
        self.init_models()
        self.setup_windows_font()
        self.start_threads()

    def init_models(self):
        has_cuda = torch.cuda.is_available()
        device = 'cuda:0' if has_cuda else 'cpu'
        print(f"🔄 正在加载 YOLO 模型 (目标设备: {device})...")
        
        if not os.path.exists(self.args.cage_model) or not os.path.exists(self.args.item_model):
            print("❌ 模型文件缺失")
            sys.exit(1)
            
        try:
            self.cage_model = YOLO(self.args.cage_model)
            self.cage_model.to(device)
            self.item_model = YOLO(self.args.item_model)
            self.item_model.to(device)
            print(f"✅ 双 YOLO 模型加载成功")
        except Exception as e:
            print(f"❌ YOLO 加载失败: {e}")
            sys.exit(1)

        print("🔄 正在初始化 PaddleOCR...")
        try:
            if paddle.device.is_compiled_with_cuda():
                paddle.device.set_device('gpu')
            else:
                paddle.device.set_device('cpu')
        except: pass

        try:
            logging.getLogger("ppocr").setLevel(logging.WARNING)
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=has_cuda, show_log=False)
            print("✅ PaddleOCR 初始化成功")
        except Exception as e:
            print(f"⚠️  标准初始化失败: {e}")
            sys.exit(1)

    def setup_windows_font(self):
        font_paths = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arial.ttf"]
        self.font = ImageFont.load_default()
        for path in font_paths:
            if os.path.exists(path):
                try:
                    self.font = ImageFont.truetype(path, 20)
                    break
                except: continue

    def start_threads(self):
        """启动所有辅助线程"""
        # OCR 线程
        t_ocr = threading.Thread(target=self.ocr_worker, daemon=True)
        t_ocr.start()
        self.ocr_threads.append(t_ocr)

        if self.server_url:
            threading.Thread(target=self.upload_worker, daemon=True).start()
            threading.Thread(target=self.video_stream_worker, daemon=True).start()

    def _connect_ws(self, ws_attr, timeout):
        """建立 WebSocket 连接"""
        if not self.server_url:
            return None
        ws = getattr(self, ws_attr, None)
        
        if ws:
            try:
                if ws.connected:
                    return ws
            except:
                pass
            
        try:
            ws = websocket.create_connection(self.server_url, timeout=timeout)
            setattr(self, ws_attr, ws)
            return ws
        except Exception as e:
            print(f"❌ [{ws_attr}] WebSocket 连接失败: {e}")
            setattr(self, ws_attr, None)
            return None

    def _close_ws(self, ws_attr):
        ws = getattr(self, ws_attr, None)
        if not ws:
            return
        try:
            ws.close()
        except:
            pass
        setattr(self, ws_attr, None)

    def _send_payload(self, payload, ws_attr, timeout, allow_drop=False):
        """发送 JSON 数据到 WebSocket"""
        if not self.server_url:
            return False
        
        data_str = json.dumps(payload)
        
        attempts = 3 if not allow_drop else 1
        
        for attempt in range(attempts):
            ws = self._connect_ws(ws_attr, timeout)
            if not ws:
                if not allow_drop: time.sleep(0.3)
                continue
            try:
                ws.send(data_str)
                return True
            except Exception as e:
                if not allow_drop:
                    print(f"⚠️ [{payload.get('type','payload')}] WS发送失败: {e}")
                self._close_ws(ws_attr)
                if not allow_drop: time.sleep(0.2)
        return False

    def image_to_base64(self, img_np, quality=80, resize_width=None):
        """通用 Base64 转换函数"""
        if img_np is None: return ""
        
        try:
            target_img = img_np
            # 如果指定了缩放宽度
            if resize_width and img_np.shape[1] > resize_width:
                ratio = resize_width / img_np.shape[1]
                dim = (resize_width, int(img_np.shape[0] * ratio))
                target_img = cv2.resize(img_np, dim, interpolation=cv2.INTER_AREA)
                
            _, buffer = cv2.imencode('.jpg', target_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return b64_str
        except Exception as e:
            print(f"⚠️ 图片转Base64失败: {e}")
            return ""

    # --- 实时视频流处理线程 ---
    def video_stream_worker(self):
        """
        专门处理 type: realvideo 的线程
        优化目标：40FPS
        """
        while self.running:
            try:
                # 获取最新一帧，timeout极短(10ms)，防止阻塞
                frame = self.video_queue.get(timeout=0.01)
            except Empty:
                continue

            try:
                # 1. 压缩处理 
                video_b64 = self.image_to_base64(
                    frame,
                    quality=self.video_stream_quality, # 55
                    resize_width=self.video_stream_width # 640
                )
                
                # 2. 构造 JSON Payload
                payload = {
                    "type": "realvideo",
                    "timestamp": str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    "image_base64": video_b64
                }
                
                # 发送数据，允许丢帧，timeout 0.5秒
                self._send_payload(payload, '_video_ws', timeout=0.5, allow_drop=True)
                    
            except Exception as e:
                print(f"❌ 视频流线程错误: {e}")
            finally:
                self.video_queue.task_done()

    def upload_worker(self):
        """统计数据上传线程 (locationInfo)"""
        while self.running:
            try:
                data_pack = self.upload_queue.get(timeout=1.0)
            except Empty: continue
            
            try:
                raw_b64 = self.image_to_base64(data_pack['raw_img'], quality=70, resize_width=self.upload_image_width)
                inf_b64 = self.image_to_base64(data_pack['inf_img'], quality=70, resize_width=self.upload_image_width)
                
                payload = {
                    "type": "locationInfo", 
                    "cage_id": str(data_pack['id']),
                    "timestamp": str(data_pack['time']),
                    "chicken_count": str(data_pack['chicken']),
                    "egg_count": str(data_pack['egg']),
                    "image_raw_base64": raw_b64,
                    "image_inf_base64": inf_b64
                }
                
                sent = self._send_payload(payload, '_upload_ws', timeout=5.0, allow_drop=False)
                if sent:
                    print(f"🚀 [上传成功] 鸡笼 No.{data_pack['id']} 数据已发送")
                else:
                    print(f"❌ [上传失败] 多次尝试仍未成功，稍后重试鸡笼 {data_pack['id']}")
                    self.upload_queue.put(data_pack)
            except Exception as e:
                print(f"❌ 上传线程内部错误: {e}")
            finally:
                self.upload_queue.task_done()

    def ocr_worker(self):
        while self.running:
            try:
                task = self.ocr_queue.get(timeout=0.1)
                crop_img, bbox_key = task
                if len(os.listdir(self.debug_ocr_folder)) < 20:
                    cv2.imwrite(f"{self.debug_ocr_folder}/ocr_{bbox_key}_{int(time.time()*1000)}.jpg", crop_img)
                text, conf = self.process_image_ocr(crop_img)
                self.ocr_result_queue.put((bbox_key, (text, conf)))
                self.ocr_queue.task_done()
            except Empty: continue
            except: continue

    def process_image_ocr(self, img):
        if img is None or img.size == 0: return None, 0.0
        try:
            h, w = img.shape[:2]
            scale = 3.0 if h < 80 else 1.5
            if scale > 1.0: img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            pad = 30
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            result = self.ocr.ocr(img, cls=True, det=True, rec=True)
            if not result or not result[0]: return None, 0.0
            best_text, best_conf = "", 0.0
            for line in result[0]:
                text_raw, conf = line[1]
                text_fix = text_raw.upper().replace('O','0').replace('D','0').replace('I','1').replace('L','1').replace('Z','2').replace('S','5').replace('B','8')
                digits = re.sub(r'[^0-9]', '', text_fix)
                if len(digits) > 0 and conf > 0.5 and conf > best_conf:
                    best_text, best_conf = digits, conf
            return (best_text, best_conf) if best_text else (None, 0.0)
        except: return None, 0.0

    def run(self):
        if self.args.video_path:
            if not os.path.exists(self.args.video_path): return
            self.cap = cv2.VideoCapture(self.args.video_path)
        else:
            self.cap = cv2.VideoCapture(0)
            
            # --- [修改点3] 摄像头设置 ---
            # 设置 MJPG 格式以支持高帧率
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            # 请求 40 FPS
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            
            # 读取并打印实际生效的帧率 (硬件通常会贴靠到 30 或 60)
            real_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"📷 摄像头初始化: 请求 40 FPS | 实际获取 {real_fps} FPS")

        if not self.cap.isOpened(): return
        print("\nCommands:\n [q] 退出程序\n [s] 手动截图\n")

        while self.cap.isOpened() and self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                if self.args.video_loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else: break

            # 处理帧
            processed_frame, updated_cage_ids = self.process_frame(frame)
            
            # 将处理后的帧推送到视频流队列
            if self.server_url:
                try:
                    # 使用 put_nowait，确保主检测循环不被网络阻塞
                    self.video_queue.put_nowait(processed_frame.copy())
                except Full:
                    # 队列满时丢帧
                    pass 
            
            # 记录最佳推理图像
            current_time = time.time()
            for c_id in updated_cage_ids:
                if c_id in self.cage_registry:
                    self.cage_registry[c_id]['best_inf_img'] = processed_frame.copy()
            
            self.check_and_save_disappeared(current_time)

            fps = 1.0 / (time.time() - start_time)
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if not self.args.no_display:
                cv2.imshow('Chicken Farm Detection System', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): self.running = False; break
            elif key == ord('s'): 
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"{self.image_data_folder}/manual_{ts}.jpg", frame)
                print(f"📸 手动截图已保存: manual_{ts}.jpg")

        self.cleanup()

    def check_and_save_disappeared(self, current_time):
        ids_to_remove = []
        for c_id, data in self.cage_registry.items():
            if current_time - data['last_seen'] > self.disappear_threshold:
                self.save_cage_data(c_id, data)
                ids_to_remove.append(c_id)
        for c_id in ids_to_remove: del self.cage_registry[c_id]
        if ids_to_remove: print(f"💾 {len(ids_to_remove)} 个鸡笼数据已保存")

    def save_cage_data(self, c_id, data):
        ts_pretty = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{c_id}_{ts_file}"
        
        # 本地保存
        txt_path = os.path.join(self.count_data_folder, f"{base_name}.txt")
        try:
            with open(txt_path, "w", encoding='utf-8') as f:
                f.write(f"ID: {c_id}\nTime: {ts_pretty}\nMax Chickens: {data['max_chicken']}\nMax Eggs: {data['max_egg']}\n")
        except: pass

        img_sub = os.path.join(self.image_data_folder, base_name)
        os.makedirs(img_sub, exist_ok=True)
        try:
            if data['best_raw_img'] is not None: cv2.imwrite(os.path.join(img_sub, "raw.jpg"), data['best_raw_img'])
            if data['best_inf_img'] is not None: cv2.imwrite(os.path.join(img_sub, "inference.jpg"), data['best_inf_img'])
        except: pass

        # 网络发送
        if self.server_url:
            self.upload_queue.put({
                'id': c_id, 'time': ts_pretty, 
                'chicken': data['max_chicken'], 'egg': data['max_egg'], 
                'raw_img': data['best_raw_img'], 'inf_img': data['best_inf_img']
            })

    def process_frame(self, frame):
        self.frame_count += 1
        while not self.ocr_result_queue.empty():
            try:
                k, (t, c) = self.ocr_result_queue.get_nowait()
                self.ocr_cache[k] = {'text': t, 'conf': c, 'ts': time.time()} if t else {'text': None, 'conf': 0, 'ts': time.time()-1.5}
            except: break
        now = time.time()
        self.ocr_cache = {k:v for k,v in self.ocr_cache.items() if now - v['ts'] < self.ocr_cache_timeout}

        detections = []
        updated_cage_ids = []
        
        cage_results = self.cage_model(frame, verbose=False, conf=0.5)
        for cage_res in cage_results:
            for c_box in cage_res.boxes:
                cx1, cy1, cx2, cy2 = map(int, c_box.xyxy[0].cpu().numpy())
                c_conf = float(c_box.conf[0])
                h, w = frame.shape[:2]
                cx1, cy1, cx2, cy2 = max(0, cx1), max(0, cy1), min(w, cx2), min(h, cy2)
                if cx2 - cx1 < 10 or cy2 - cy1 < 10: continue
                
                cage_roi = frame[cy1:cy2, cx1:cx2]
                item_results = self.item_model(cage_roi, verbose=False, conf=0.3)
                
                items_in_cage = []
                c_cnt, e_cnt, cage_id = 0, 0, None
                
                for item_res in item_results:
                    names = item_res.names
                    for i_box in item_res.boxes:
                        cls_name = names[int(i_box.cls[0])]
                        if cls_name not in self.conf_thresholds or float(i_box.conf[0]) < self.conf_thresholds[cls_name]: continue
                        
                        if cls_name == 'chicken': c_cnt += 1
                        elif cls_name == 'egg': e_cnt += 1
                        
                        rx1, ry1, rx2, ry2 = map(int, i_box.xyxy[0].cpu().numpy())
                        ax1, ay1, ax2, ay2 = cx1+rx1, cy1+ry1, cx1+rx2, cy1+ry2
                        det = {'type': cls_name, 'bbox': [ax1, ay1, ax2, ay2], 'conf': float(i_box.conf[0]), 'text': None}
                        
                        if cls_name == 'number':
                            g_key = f"{ax1//30}_{ay1//30}_{ax2//30}_{ay2//30}"
                            if g_key in self.ocr_cache: 
                                det['text'] = self.ocr_cache[g_key]['text']
                                if det['text']: cage_id = det['text']
                            else:
                                det['text'] = "WAITING"
                                if self.frame_count % self.ocr_skip_frames == 0 and not self.ocr_queue.full():
                                    self.ocr_queue.put((frame[ay1:ay2, ax1:ax2].copy(), g_key))
                        items_in_cage.append(det)
                
                final_chicken, final_egg = c_cnt, e_cnt
                if cage_id:
                    if cage_id not in self.cage_registry:
                        self.cage_registry[cage_id] = {'max_chicken': c_cnt, 'max_egg': e_cnt, 'last_seen': now, 'best_raw_img': frame.copy(), 'best_inf_img': None}
                        updated_cage_ids.append(cage_id)
                    else:
                        reg = self.cage_registry[cage_id]
                        reg['last_seen'] = now
                        if c_cnt + e_cnt > reg['max_chicken'] + reg['max_egg']:
                            reg.update({'max_chicken': max(c_cnt, reg['max_chicken']), 'max_egg': max(e_cnt, reg['max_egg']), 'best_raw_img': frame.copy()})
                            updated_cage_ids.append(cage_id)
                        final_chicken, final_egg = reg['max_chicken'], reg['max_egg']
                
                for it in items_in_cage:
                    if it['type'] == 'number': it['stats_payload'] = {'chickens': final_chicken, 'eggs': final_egg}
                
                detections.append({'type': 'cage', 'bbox': [cx1, cy1, cx2, cy2], 'conf': c_conf})
                detections.extend(items_in_cage)

        return self.draw_detections(frame, detections), updated_cage_ids

    def draw_detections(self, frame, detections):
        img_pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            l = det['type']
            c, w = ((0,0,255),2) if l=='cage' else ((0,255,255),3) if l=='chicken' else ((255,255,255),3) if l=='egg' else ((255,50,50),3) if l=='number' else ((0,255,0),2)
            draw.rectangle([x1, y1, x2, y2], outline=c, width=w)
            
            txt = ""
            if l == 'number':
                t_val = det['text']
                t_str = f"No.{t_val}" if t_val and t_val!="WAITING" else "Scanning..." if t_val=="WAITING" else "Unknown"
                st = det.get('stats_payload')
                txt = f"{t_str} | C:{st['chickens']} E:{st['eggs']}" if st else t_str
                bg = (150,0,150) if st else c
                
                tb = draw.textbbox((0,0), txt, font=self.font)
                draw.rectangle([x1, y1-(tb[3]-tb[1])-6, x1+(tb[2]-tb[0])+4, y1], fill=bg)
                draw.text((x1+2, y1-(tb[3]-tb[1])-6), txt, font=self.font, fill=(255,255,255))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def cleanup(self):
        print("💾 正在保存剩余数据...")
        for c_id, data in self.cage_registry.items(): self.save_cage_data(c_id, data)
        self.running = False
        if self.cap: self.cap.release()
        self._close_ws('_video_ws')
        self._close_ws('_upload_ws')
        cv2.destroyAllWindows()
        print("✅ 系统已退出")

def parse_args():
    parser = argparse.ArgumentParser(description="养鸡场检测系统 (WebSocket实时视频流版)")
    parser.add_argument('--cage-model', type=str, default='models/cage_best.pt')
    parser.add_argument('--item-model', type=str, default='models/items_best.pt')
    parser.add_argument('--video-path', type=str, default='videos/1.mp4')
    parser.add_argument('--video-loop', action='store_true')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--chicken-conf', type=float, default=0.5)
    parser.add_argument('--egg-conf', type=float, default=0.45)
    parser.add_argument('--number-conf', type=float, default=0.6)
    parser.add_argument('--server-url', type=str, default='ws://192.168.0.95:8080/LeogEgg/websocket', help="WebSocket服务器地址")
    return parser.parse_args()

if __name__ == "__main__":
    system = ChickenFarmSystem(parse_args())
    system.run()