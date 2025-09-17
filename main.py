import google.generativeai as genai
import time
import os
from tqdm import tqdm
import threading
from datetime import datetime, timedelta
import argparse
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class GeminiVideoAnalyzer:
    def __init__(self, api_key, model_name='gemini-2.5-pro'):
        """初始化Gemini视频分析器"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.upload_progress = 0
        self.analysis_progress = 0
        self.current_status = "准备中..."
        
    def _format_file_size(self, size_bytes):
        """格式化文件大小显示"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.1f} MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.1f} GB"
    
    def _get_file_info(self, file_path):
        """获取文件信息"""
        size = os.path.getsize(file_path)
        return {
            'name': os.path.basename(file_path),
            'size': size,
            'size_str': self._format_file_size(size)
        }
    
    def _upload_with_progress(self, file_path, file_type="video"):
        """带进度显示的文件上传"""
        file_info = self._get_file_info(file_path)
        
        print(f"\n📁 开始上传{file_type}文件...")
        print(f"   文件名: {file_info['name']}")
        print(f"   文件大小: {file_info['size_str']}")
        
        # 创建进度条
        progress_bar = tqdm(
            total=100, 
            desc=f"上传{file_type}", 
            unit="%",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        # 模拟上传进度（因为Google API没有提供真实进度回调）
        upload_start = time.time()
        
        def simulate_upload_progress():
            for i in range(100):
                time.sleep(0.1)  # 调整这个值来模拟上传速度
                progress_bar.update(1)
                self.upload_progress = i + 1
        
        # 启动进度模拟线程
        progress_thread = threading.Thread(target=simulate_upload_progress)
        progress_thread.start()
        
        try:
            # 实际上传文件
            uploaded_file = genai.upload_file(path=file_path)
            
            # 等待进度条完成
            progress_thread.join()
            progress_bar.close()
            
            upload_time = time.time() - upload_start
            print(f"✅ {file_type}上传完成！耗时: {upload_time:.1f}秒")
            print(f"   文件URI: {uploaded_file.uri}")
            
            return uploaded_file
            
        except Exception as e:
            progress_bar.close()
            print(f"❌ {file_type}上传失败: {str(e)}")
            raise
    
    def _wait_for_file_processing(self, uploaded_file, file_type="video"):
        """等待文件处理完成"""
        print(f"\n⏳ 等待{file_type}文件处理...")
        
        processing_start = time.time()
        spinner_chars = "|/-\\"
        spinner_idx = 0
        
        while uploaded_file.state.name == "PROCESSING":
            print(f"\r🔄 {spinner_chars[spinner_idx]} 文件处理中... (已等待 {int(time.time() - processing_start)}秒)", end="")
            spinner_idx = (spinner_idx + 1) % len(spinner_chars)
            time.sleep(1)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        print(f"\r✅ 文件处理完成！总耗时: {int(time.time() - processing_start)}秒" + " " * 20)
        return uploaded_file
    
    def analyze_person_in_video(self, video_path, reference_photo_path):
        """完整的视频人物分析流程"""
        
        print("🎬 Gemini 2.5 Pro 视频人物识别分析")
        print("=" * 50)
        
        total_start_time = time.time()
        
        try:
            # 步骤1：上传参考照片
            self.current_status = "上传参考照片中..."
            reference_file = self._upload_with_progress(reference_photo_path, "参考照片")
            
            # 步骤2：上传视频文件
            self.current_status = "上传视频文件中..."
            video_file = self._upload_with_progress(video_path, "视频")
            
            # 步骤3：等待文件处理
            self.current_status = "处理参考照片中..."
            reference_file = self._wait_for_file_processing(reference_file, "参考照片")
            
            self.current_status = "处理视频文件中..."
            video_file = self._wait_for_file_processing(video_file, "视频")
            
            # 步骤4：构建分析提示
            self.current_status = "构建分析提示中..."
            print(f"\n📝 构建分析提示...")
            
            prompt = f"""
            我需要你详细分析这个直播视频，判断参考照片中的人物是否以真人形式出现在视频中。

            **核心任务：基于生物特征的身份确认 + 截图证据支撑**
            **重要：仅关注不可变的生物特征，排除所有可变外在因素**
            **必要：每个结论都必须提供时间戳截图作为可视化证据**

            **第一步：参考照片生物特征提取**
            请仅分析以下固有特征（忽略发型、服装、眼镜、饰品、妆容等）：
            1. **骨骼结构特征**：
               - 头颅形状和比例（长宽比、前额高度）
               - 脸型轮廓（方形/圆形/椭圆形/心形）
               - 颧骨高度和突出程度
               - 下颌线条和下巴形状

            2. **五官固有特征**：
               - 鼻梁形状（直/弯曲）、鼻翼宽度、鼻头形状
               - 嘴唇厚度比例、嘴角形状、人中深度
               - 眉骨结构、眉间距离
               - 耳廓形状和耳垂特征（如果可见）

            3. **面部比例关系**：
               - 三庭五眼比例
               - 眼距与面宽比例
               - 鼻长与面长比例

            **第二步：视频活体检测与生物特征对比**
            逐段分析视频时，必须同时验证：

            A. **活体真实性指标**：
               - 自然的面部表情变化和微表情
               - 眨眼频率和自然程度
               - 说话时口型与声音的同步性
               - 头部的三维转动和角度变化
               - 面部肌肉的自然运动

            B. **防伪检测**：
               - 检查是否存在平面照片特征：边缘反光、不自然的阴影
               - 观察人物与环境的深度关系是否真实
               - 验证光线在面部的自然反射效果
               - 检测是否有屏幕显示痕迹（像素点、刷新频率）

            C. **生物特征匹配**：
               - 对比上述提取的骨骼结构特征
               - 验证五官固有形状是否一致
               - 计算面部比例关系的匹配度

            **第三步：证据收集（强制要求）**
            **重要：无论结论如何，都必须提供时间戳截图作为证据**

            A. **如果判断为同一人**：
               - 提供3-5个最相似的时间戳（最清晰的正面角度）
               - 每个时间戳说明证明的具体生物特征
               - 生物特征相似度评分（1-10分）
               - 活体检测置信度（1-10分）

            B. **如果判断为不同人**：
               - 提供3-5个视频中主要人物的代表性时间戳
               - 每个时间戳说明与参考照片的关键差异
               - 重点展示不匹配的生物特征证据

            C. **如果判断不确定**：
               - 提供导致不确定的关键时间戳
               - 说明哪些特征相似，哪些特征不同
               - 解释为什么无法得出明确结论

            **第四步：时间戳证据说明（必填）**
            对每个提供的时间戳，必须说明：
            1. **选择理由**：为什么选择这个时间点
            2. **特征展示**：该截图显示的关键生物特征
            3. **对比结论**：与参考照片的具体对比结果
            4. **证据价值**：该截图对最终结论的支撑作用

            **第五步：综合判断**
            基于截图证据得出最终结论：
            - 明确的判断结论（是/否/不确定）
            - 整体置信度（1-10分）
            - 关键证据总结

            **防伪警告级别**：
            - 低风险：自然动作，真实交互
            - 中风险：部分动作略显僵硬
            - 高风险：疑似照片或屏幕显示

            **输出格式要求：**
            在分析结论后，请单独列出：
            证据时间戳: [时间戳1, 时间戳2, 时间戳3...]
            活体检测评分: X/10
            生物特征评分: X/10
            伪造风险评分: X/10 (1-3分为低风险，4-6分为中风险，7-10分为高风险)

            **时间戳精度要求：**
            - 请提供帧级精确时间戳，格式：MM:SS.FF 或 HH:MM:SS.FF
            - 其中FF表示该秒内的帧数（例如：00:09.15 表示第9秒的第15帧）
            - 这样可以精确定位到具体帧，提供最准确的截图证据
            - 如果无法确定具体帧数，可使用 MM:SS.00 格式

            请开始你的专业生物识别分析。
            """
            
            # 步骤5：开始AI分析
            self.current_status = "AI分析中..."
            print(f"\n🤖 开始AI分析...")
            print("   这可能需要几分钟时间，请耐心等待...")
            
            analysis_start = time.time()
            
            # 创建分析进度指示器
            analysis_progress = tqdm(
                total=100,
                desc="AI分析进度",
                bar_format='{l_bar}{bar}| {elapsed} | {desc}'
            )
            
            def simulate_analysis_progress():
                """模拟分析进度"""
                # 前30%较快
                for i in range(30):
                    time.sleep(2)
                    analysis_progress.update(1)
                
                # 中间40%较慢（实际分析阶段）
                for i in range(40):
                    time.sleep(5)
                    analysis_progress.update(1)
                
                # 最后30%逐渐完成
                for i in range(30):
                    time.sleep(3)
                    analysis_progress.update(1)
            
            # 启动进度模拟
            progress_thread = threading.Thread(target=simulate_analysis_progress)
            progress_thread.start()
            
            # 实际API调用
            response = self.model.generate_content([
                prompt,
                reference_file,
                video_file
            ])

            # 获取token使用统计
            token_count = response.usage_metadata if hasattr(response, 'usage_metadata') else None
            
            # 等待进度条完成
            progress_thread.join()
            analysis_progress.close()
            
            analysis_time = time.time() - analysis_start
            total_time = time.time() - total_start_time
            
            # 步骤6：显示结果
            print(f"\n✅ 分析完成！")
            print(f"   分析耗时: {analysis_time/60:.1f}分钟")
            print(f"   总耗时: {total_time/60:.1f}分钟")

            # 显示token使用统计
            if token_count:
                print(f"\n📊 Token使用统计:")
                print(f"   输入token: {token_count.prompt_token_count:,}")
                print(f"   输出token: {token_count.candidates_token_count:,}")
                print(f"   总token: {token_count.total_token_count:,}")

            print("\n" + "=" * 50)
            print("📊 分析结果:")
            print("=" * 50)
            print(response.text)

            # 提取分析数据并生成证据截图
            analysis_data = self._extract_analysis_data_from_response(response.text)
            timestamps = analysis_data['timestamps']
            screenshot_files = []
            comparison_file = None

            # 显示评分信息
            if any([analysis_data['liveness_score'], analysis_data['biometric_score'], analysis_data['spoofing_risk_score']]):
                print(f"\n📊 专业评分结果:")
                if analysis_data['liveness_score'] is not None:
                    score_emoji = "✅" if analysis_data['liveness_score'] >= 8 else "⚠️" if analysis_data['liveness_score'] >= 6 else "❌"
                    print(f"   {score_emoji} 活体检测评分: {analysis_data['liveness_score']}/10")
                if analysis_data['biometric_score'] is not None:
                    score_emoji = "✅" if analysis_data['biometric_score'] >= 8 else "⚠️" if analysis_data['biometric_score'] >= 6 else "❌"
                    print(f"   {score_emoji} 生物特征评分: {analysis_data['biometric_score']}/10")
                if analysis_data['spoofing_risk_score'] is not None:
                    if analysis_data['spoofing_risk_score'] <= 3:
                        risk_emoji = "🟢"
                        risk_level = "低风险"
                    elif analysis_data['spoofing_risk_score'] <= 6:
                        risk_emoji = "🟡"
                        risk_level = "中风险"
                    else:
                        risk_emoji = "🔴"
                        risk_level = "高风险"
                    print(f"   {risk_emoji} 伪造风险评分: {analysis_data['spoofing_risk_score']}/10 ({risk_level})")

            if timestamps:
                screenshot_files = self._extract_video_frames(video_path, timestamps)
                if screenshot_files:
                    print(f"\n📷 证据截图已保存:")
                    for file in screenshot_files:
                        print(f"   🖼️  {file}")

                    # 生成对比图
                    comparison_file = self._create_comparison_image(reference_photo_path, screenshot_files)
                    if comparison_file:
                        print(f"   🖼️  {comparison_file}")

            # 清理上传的文件
            self._cleanup_files([reference_file, video_file])
            
            return {
                'success': True,
                'analysis': response.text,
                'analysis_time': analysis_time,
                'total_time': total_time,
                'token_usage': {
                    'prompt_tokens': token_count.prompt_token_count if token_count else 0,
                    'output_tokens': token_count.candidates_token_count if token_count else 0,
                    'total_tokens': token_count.total_token_count if token_count else 0
                } if token_count else None,
                'timestamps': timestamps,
                'screenshot_files': screenshot_files,
                'comparison_file': comparison_file,
                'liveness_score': analysis_data['liveness_score'],
                'biometric_score': analysis_data['biometric_score'],
                'spoofing_risk_score': analysis_data['spoofing_risk_score']
            }
            
        except Exception as e:
            print(f"\n❌ 分析过程中出现错误: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_analysis_data_from_response(self, response_text):
        """从AI响应中提取时间戳和评分信息"""
        result = {
            'timestamps': [],
            'liveness_score': None,
            'biometric_score': None,
            'spoofing_risk_score': None
        }

        # 提取时间戳（支持中英文格式和帧精度）
        timestamp_patterns = [
            r'证据时间戳:\s*\[(.*?)\]',
            r'EVIDENCE_TIMESTAMPS:\s*\[(.*?)\]'
        ]
        for pattern in timestamp_patterns:
            timestamp_match = re.search(pattern, response_text, re.IGNORECASE)
            if timestamp_match:
                timestamps_str = timestamp_match.group(1)
                # 支持帧精度格式：MM:SS.FF 或 HH:MM:SS.FF，以及旧格式
                time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,2})?)'
                result['timestamps'] = re.findall(time_pattern, timestamps_str)
                break

        # 提取活体检测评分（支持中英文格式）
        liveness_patterns = [
            r'活体检测评分:\s*(\d+)/10',
            r'LIVENESS_SCORE:\s*(\d+)/10'
        ]
        for pattern in liveness_patterns:
            liveness_match = re.search(pattern, response_text, re.IGNORECASE)
            if liveness_match:
                result['liveness_score'] = int(liveness_match.group(1))
                break

        # 提取生物特征评分（支持中英文格式）
        biometric_patterns = [
            r'生物特征评分:\s*(\d+)/10',
            r'BIOMETRIC_SCORE:\s*(\d+)/10'
        ]
        for pattern in biometric_patterns:
            biometric_match = re.search(pattern, response_text, re.IGNORECASE)
            if biometric_match:
                result['biometric_score'] = int(biometric_match.group(1))
                break

        # 提取伪造风险评分（新的数字格式）
        spoofing_score_pattern = r'伪造风险评分:\s*(\d+)/10'
        spoofing_score_match = re.search(spoofing_score_pattern, response_text, re.IGNORECASE)
        if spoofing_score_match:
            result['spoofing_risk_score'] = int(spoofing_score_match.group(1))

        return result

    def _convert_timestamp_to_frame_info(self, timestamp, fps):
        """将时间戳转换为帧信息"""
        # 分离时间和帧数
        if '.' in timestamp:
            time_part, frame_part = timestamp.split('.')
            frame_offset = int(frame_part)
        else:
            time_part = timestamp
            frame_offset = 0

        # 解析时间部分
        parts = time_part.split(':')
        if len(parts) == 2:  # MM:SS格式
            minutes, seconds = int(parts[0]), int(parts[1])
            total_seconds = minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS格式
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            total_seconds = hours * 3600 + minutes * 60 + seconds
        else:
            total_seconds = 0

        # 计算精确帧号
        base_frame = int(total_seconds * fps)
        exact_frame = base_frame + frame_offset

        return {
            'total_seconds': total_seconds,
            'frame_offset': frame_offset,
            'exact_frame': exact_frame,
            'timestamp_with_frame': f"{time_part}.{frame_offset:02d}"
        }

    def _convert_timestamp_to_seconds(self, timestamp):
        """将时间戳转换为秒数（保持向后兼容）"""
        # 分离时间和帧数
        if '.' in timestamp:
            time_part, _ = timestamp.split('.')
        else:
            time_part = timestamp

        parts = time_part.split(':')
        if len(parts) == 2:  # MM:SS格式
            minutes, seconds = int(parts[0]), int(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS格式
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        return 0

    def _extract_video_frames(self, video_path, timestamps, output_dir="screenshots"):
        """根据时间戳提取视频帧（支持帧级精度）"""
        if not timestamps:
            return []

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📸 提取证据截图...")
        print(f"   目标时间戳: {timestamps}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        extracted_files = []

        print(f"   视频信息: {fps:.2f} FPS, 总帧数: {total_frames}")

        for i, timestamp in enumerate(timestamps):
            # 获取精确帧信息
            frame_info = self._convert_timestamp_to_frame_info(timestamp, fps)
            exact_frame = frame_info['exact_frame']

            # 确保帧号在有效范围内
            if exact_frame >= total_frames:
                exact_frame = total_frames - 1
                print(f"   ⚠️  时间戳 {timestamp} 超出视频范围，使用最后一帧")

            # 精确定位到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, exact_frame)
            ret, frame = cap.read()

            if ret:
                # 验证实际提取的帧号
                actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                # 生成文件名（替换特殊字符）
                safe_timestamp = timestamp.replace(':', '-').replace('.', '_')
                filename = f"evidence_{i+1}_{safe_timestamp}_frame{exact_frame}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame)
                extracted_files.append(filepath)

                print(f"   ✅ 已提取: {filename}")
                print(f"       目标时间戳: {timestamp} (第{exact_frame}帧)")
                print(f"       实际提取帧: 第{actual_frame}帧")
                if abs(actual_frame - exact_frame) > 1:
                    print(f"       ⚠️  帧偏移: {actual_frame - exact_frame}帧")
            else:
                print(f"   ❌ 提取失败: 时间戳 {timestamp} (第{exact_frame}帧)")

        cap.release()
        return extracted_files

    def _create_comparison_image(self, reference_photo_path, screenshot_files, output_dir="screenshots"):
        """创建参考照片与截图的对比图"""
        if not screenshot_files:
            return None

        try:
            # 读取参考照片
            ref_img = Image.open(reference_photo_path)

            # 设置图片尺寸
            img_width = 300
            img_height = 300

            # 调整参考照片大小
            ref_img = ref_img.resize((img_width, img_height), Image.Resampling.LANCZOS)

            # 计算总画布尺寸
            cols = min(4, len(screenshot_files) + 1)  # 最多4列
            rows = (len(screenshot_files) + 1 + cols - 1) // cols  # 计算所需行数

            canvas_width = cols * img_width + (cols + 1) * 20  # 间距
            canvas_height = rows * (img_height + 40) + 40  # 标题空间

            # 创建画布
            canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
            draw = ImageDraw.Draw(canvas)

            # 尝试加载字体
            try:
                font = ImageFont.truetype("Arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            # 放置参考照片
            canvas.paste(ref_img, (20, 40))
            draw.text((20, 10), "参考照片", fill='black', font=font)

            # 放置截图
            for i, screenshot_file in enumerate(screenshot_files):
                try:
                    screenshot = Image.open(screenshot_file)
                    screenshot = screenshot.resize((img_width, img_height), Image.Resampling.LANCZOS)

                    # 计算位置
                    col = (i + 1) % cols
                    row = (i + 1) // cols
                    if col == 0:
                        col = cols
                        row -= 1

                    x = col * (img_width + 20)
                    y = row * (img_height + 40) + 40

                    canvas.paste(screenshot, (x, y))

                    # 添加时间戳和帧号标签
                    filename = os.path.basename(screenshot_file)
                    # 解析新格式的文件名: evidence_1_00-09_15_frame15.jpg
                    parts = filename.split('_')
                    if len(parts) >= 4 and 'frame' in parts[-1]:
                        # 新格式: 有帧号信息
                        timestamp_part = '_'.join(parts[2:-1])  # 00-09_15
                        frame_part = parts[-1].replace('.jpg', '')  # frame15
                        timestamp = timestamp_part.replace('-', ':').replace('_', '.')  # 00:09.15
                        frame_num = frame_part.replace('frame', '')
                        label = f"时间: {timestamp} (第{frame_num}帧)"
                    else:
                        # 旧格式: 只有时间戳
                        timestamp = parts[2].replace('-', ':').replace('.jpg', '')
                        label = f"时间: {timestamp}"

                    draw.text((x, y - 25), label, fill='black', font=font)

                except Exception as e:
                    print(f"   ⚠️  处理截图失败: {screenshot_file}, 错误: {e}")

            # 保存对比图
            comparison_path = os.path.join(output_dir, "comparison_evidence.jpg")
            canvas.save(comparison_path, "JPEG", quality=90)

            print(f"   📋 对比图已生成: {comparison_path}")
            return comparison_path

        except Exception as e:
            print(f"   ❌ 生成对比图失败: {e}")
            return None

    def _cleanup_files(self, files):
        """清理上传的文件"""
        print(f"\n🧹 清理临时文件...")
        for file in files:
            try:
                genai.delete_file(file.name)
                print(f"   ✅ 已删除: {file.name}")
            except:
                print(f"   ⚠️  删除失败: {file.name}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用Gemini API进行视频人物识别分析')
    parser.add_argument('--model', type=str, default='gemini-2.5-pro',
                       help='Gemini模型名称 (默认: gemini-2.5-pro)')
    parser.add_argument('--video', type=str, required=True,
                       help='视频文件路径')
    parser.add_argument('--image', type=str, required=True,
                       help='参考图片文件路径')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 配置你的API密钥
    API_KEY = "AIzaSyCramKPwcDFmB-6I8Tn6FVioayosJJUeXo"

    # 创建分析器
    analyzer = GeminiVideoAnalyzer(API_KEY, args.model)

    # 分析视频
    result = analyzer.analyze_person_in_video(
        video_path=args.video,
        reference_photo_path=args.image
    )
    
    if result['success']:
        print(f"\n🎉 分析成功完成！")
        # 可以进一步处理分析结果
    else:
        print(f"\n😞 分析失败: {result['error']}")

if __name__ == "__main__":
    main()