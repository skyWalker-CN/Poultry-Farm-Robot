#!/usr/bin/env python3
"""
测试中文字体渲染
"""
import sys
import os
import cv2
import numpy as np

def test_chinese_font():
    """测试中文字体渲染"""
    print("🔍 测试中文字体渲染...")
    
    # 测试PIL是否可用
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("✅ PIL库可用")
        
        # 创建测试图像
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        
        # 测试中文字体
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试加载字体
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        ]
        
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 20)
                print(f"✅ 找到字体: {font_path}")
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
            print("⚠️ 使用默认字体")
        
        # 测试中文渲染
        test_text = "一层画面"
        draw.text((10, 10), test_text, font=font, fill=(0, 255, 255))
        
        # 转换回OpenCV格式
        result_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        print("✅ 中文渲染测试成功")
        
        return True
        
    except ImportError:
        print("❌ PIL库不可用")
        print("💡 请安装PIL库: pip install Pillow")
        return False
    except Exception as e:
        print(f"❌ 中文渲染测试失败: {e}")
        return False

def test_opencv_english():
    """测试OpenCV英文渲染"""
    print("\n🔍 测试OpenCV英文渲染...")
    
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    
    # 使用OpenCV渲染英文
    label = "Layer 1"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    cv2.rectangle(frame, (10, 10), (text_width + 20, text_height + 20), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (text_width + 20, text_height + 20), (0, 255, 255), 2)
    cv2.putText(frame, label, (15, text_height + 15), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    
    print("✅ OpenCV英文渲染测试成功")
    return True

if __name__ == '__main__':
    chinese_ok = test_chinese_font()
    test_opencv_english()
    
    if not chinese_ok:
        print("\n💡 建议:")
        print("1. 安装PIL库: pip install Pillow")
        print("2. 或者使用英文标签: Layer 1, Layer 2, Layer 3, Layer 4")
