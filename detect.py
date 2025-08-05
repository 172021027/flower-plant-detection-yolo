#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
花草检测YOLO推理脚本

基于训练好的YOLOv8模型进行花草检测推理。

使用方法:
    # 单张图片检测
    python detect.py --source image.jpg --model models/best_flower_model.pt
    
    # 批量图片检测
    python detect.py --source images/ --model models/best_flower_model.pt
    
    # 视频检测
    python detect.py --source video.mp4 --model models/best_flower_model.pt
    
    # 实时摄像头检测
    python detect.py --source 0 --model models/best_flower_model.pt
"""

import os
import sys
import argparse
import logging
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("错误: 请安装ultralytics和torch库")
    print("运行: pip install ultralytics torch")
    sys.exit(1)

from src.models.yolo_detector import FlowerDetector
from src.utils.visualization import DetectionVisualizer
from src import FLOWER_CLASSES, FLOWER_CLASSES_EN


class FlowerDetectionApp:
    """花草检测应用程序类"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45, device: str = 'auto'):
        """
        初始化检测应用
        
        Args:
            model_path: 模型文件路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            device: 计算设备 ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = self._setup_device(device)
        
        # 初始化检测器和可视化器
        self.detector = None
        self.visualizer = DetectionVisualizer()
        
        # 设置日志
        self._setup_logging()
        
        # 加载模型
        self._load_model()
        
    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda:0'
                print(f"🔥 检测到CUDA，使用GPU加速: {device}")
            else:
                device = 'cpu'
                print("💻 使用CPU进行推理")
        else:
            print(f"🎯 使用指定设备: {device}")
            
        return device
        
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_model(self):
        """加载YOLO模型"""
        try:
            if not os.path.exists(self.model_path):
                self.logger.error(f"模型文件不存在: {self.model_path}")
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
                
            self.logger.info(f"正在加载模型: {self.model_path}")
            
            # 创建花草检测器
            self.detector = FlowerDetector(
                model_path=self.model_path,
                device=self.device,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold
            )
            
            self.logger.info("✅ 模型加载成功!")
            
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            raise
            
    def detect_image(self, image_path: str, save_path: Optional[str] = None, 
                    show: bool = False) -> Dict[str, Any]:
        """
        检测单张图片
        
        Args:
            image_path: 图片路径
            save_path: 保存结果的路径
            show: 是否显示结果
            
        Returns:
            检测结果字典
        """
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
                
            self.logger.info(f"🔍 正在检测图片: {image_path}")
            
            # 执行检测
            start_time = time.time()
            results = self.detector.detect(image)
            inference_time = time.time() - start_time
            
            # 处理检测结果
            detections = self._process_results(results)
            
            # 可视化结果
            if detections['detections']:
                annotated_image = self.visualizer.draw_detections(
                    image, detections['detections']
                )
            else:
                annotated_image = image.copy()
                
            # 添加信息文本
            info_text = f"检测到 {len(detections['detections'])} 个花草对象"
            info_text += f" | 推理时间: {inference_time:.3f}s"
            self.visualizer.add_info_text(annotated_image, info_text)
            
            # 保存结果
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, annotated_image)
                self.logger.info(f"💾 结果已保存: {save_path}")
                
            # 显示结果
            if show:
                self._show_image("花草检测结果", annotated_image)
                
            # 打印检测统计
            self._print_detection_stats(detections['detections'], inference_time)
            
            return {
                'image_path': image_path,
                'detections': detections['detections'],
                'inference_time': inference_time,
                'annotated_image': annotated_image
            }
            
        except Exception as e:
            self.logger.error(f"图片检测失败 {image_path}: {e}")
            raise
            
    def detect_batch(self, source_dir: str, output_dir: str, 
                    image_extensions: List[str] = None) -> List[Dict[str, Any]]:
        """
        批量检测图片
        
        Args:
            source_dir: 源图片目录
            output_dir: 输出目录
            image_extensions: 支持的图片格式
            
        Returns:
            所有检测结果列表
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            
        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(source_dir).glob(f"*{ext}"))
            image_files.extend(Path(source_dir).glob(f"*{ext.upper()}"))
            
        if not image_files:
            self.logger.warning(f"在目录 {source_dir} 中未找到图片文件")
            return []
            
        self.logger.info(f"🗂️ 找到 {len(image_files)} 张图片，开始批量检测...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        total_time = 0
        
        for i, image_path in enumerate(image_files, 1):
            try:
                # 生成输出路径
                output_path = os.path.join(
                    output_dir, 
                    f"detected_{image_path.name}"
                )
                
                print(f"\n[{i}/{len(image_files)}] 检测: {image_path.name}")
                
                # 检测图片
                result = self.detect_image(
                    str(image_path), 
                    output_path, 
                    show=False
                )
                
                results.append(result)
                total_time += result['inference_time']
                
            except Exception as e:
                self.logger.error(f"批量检测失败 {image_path}: {e}")
                continue
                
        # 打印批量检测统计
        self._print_batch_stats(results, total_time)
        
        return results
        
    def detect_video(self, video_path: str, output_path: Optional[str] = None,
                    display: bool = True, fps_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        检测视频
        
        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径
            display: 是否实时显示
            fps_limit: 限制处理帧率
            
        Returns:
            检测结果统计
        """
        try:
            # 打开视频
            if video_path == '0':  # 摄像头
                cap = cv2.VideoCapture(0)
                self.logger.info("📹 开始实时摄像头检测 (按 'q' 退出)")
            else:
                cap = cv2.VideoCapture(video_path)
                self.logger.info(f"🎥 正在检测视频: {video_path}")
                
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
                
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if video_path != '0':
                self.logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
                
            # 设置输出视频
            writer = None
            if output_path and video_path != '0':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
            # 处理视频帧
            frame_count = 0
            detection_stats = {'total_detections': 0, 'frame_count': 0}
            start_time = time.time()
            
            # 计算帧间隔（用于限制FPS）
            frame_interval = 1
            if fps_limit and fps > fps_limit:
                frame_interval = fps // fps_limit
                
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # 跳帧处理（用于限制FPS）
                if frame_count % frame_interval != 0:
                    continue
                    
                try:
                    # 执行检测
                    results = self.detector.detect(frame)
                    detections = self._process_results(results)
                    
                    # 可视化结果
                    if detections['detections']:
                        annotated_frame = self.visualizer.draw_detections(
                            frame, detections['detections']
                        )
                        detection_stats['total_detections'] += len(detections['detections'])
                    else:
                        annotated_frame = frame.copy()
                        
                    # 添加帧信息
                    if video_path != '0':
                        info_text = f"帧: {frame_count}/{total_frames}"
                    else:
                        info_text = f"实时检测 - 帧: {frame_count}"
                        
                    info_text += f" | 检测: {len(detections['detections'])} 个对象"
                    self.visualizer.add_info_text(annotated_frame, info_text)
                    
                    detection_stats['frame_count'] += 1
                    
                    # 保存视频帧
                    if writer:
                        writer.write(annotated_frame)
                        
                    # 显示结果
                    if display:
                        cv2.imshow('花草检测 - 视频', annotated_frame)
                        
                        # 检查退出键
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                            
                except Exception as e:
                    self.logger.error(f"处理第{frame_count}帧时出错: {e}")
                    continue
                    
            # 清理资源
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            
            # 打印视频检测统计
            self._print_video_stats(detection_stats, total_time)
            
            if output_path and video_path != '0':
                self.logger.info(f"💾 视频结果已保存: {output_path}")
                
            return detection_stats
            
        except Exception as e:
            self.logger.error(f"视频检测失败: {e}")
            raise
            
    def _process_results(self, results) -> Dict[str, Any]:
        """处理YOLO检测结果"""
        detections = []
        
        if results and len(results) > 0:
            result = results[0]  # 取第一个结果
            
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    # 获取置信度和类别
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # 获取类别名称
                    class_name_cn = FLOWER_CLASSES.get(class_id, "未知")
                    class_name_en = FLOWER_CLASSES_EN.get(class_id, "Unknown")
                    
                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name_cn': class_name_cn,
                        'class_name_en': class_name_en
                    }
                    
                    detections.append(detection)
                    
        return {'detections': detections}
        
    def _show_image(self, window_name: str, image: np.ndarray):
        """显示图片"""
        # 调整图片大小以适应屏幕
        h, w = image.shape[:2]
        max_size = 1000
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            
        cv2.imshow(window_name, image)
        print("按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def _print_detection_stats(self, detections: List[Dict], inference_time: float):
        """打印检测统计信息"""
        if not detections:
            print("🔍 未检测到任何花草对象")
            return
            
        print(f"\n🌸 检测结果统计:")
        print(f"  📊 总计检测到: {len(detections)} 个对象")
        print(f"  ⏱️  推理时间: {inference_time:.3f}秒")
        
        # 统计各类别数量
        class_counts = {}
        for det in detections:
            class_name = det['class_name_cn']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        print(f"  🏷️  类别分布:")
        for class_name, count in class_counts.items():
            print(f"    - {class_name}: {count} 个")
            
        # 显示置信度信息
        confidences = [det['confidence'] for det in detections]
        avg_conf = np.mean(confidences)
        max_conf = np.max(confidences)
        min_conf = np.min(confidences)
        
        print(f"  🎯 置信度统计:")
        print(f"    - 平均: {avg_conf:.3f}")
        print(f"    - 最高: {max_conf:.3f}")
        print(f"    - 最低: {min_conf:.3f}")
        
    def _print_batch_stats(self, results: List[Dict], total_time: float):
        """打印批量检测统计"""
        if not results:
            return
            
        total_detections = sum(len(r['detections']) for r in results)
        avg_time = total_time / len(results)
        
        print(f"\n📊 批量检测统计:")
        print(f"  🖼️  处理图片: {len(results)} 张")
        print(f"  🌸 总计检测: {total_detections} 个对象")
        print(f"  ⏱️  总时间: {total_time:.2f}秒")
        print(f"  📈 平均每张: {avg_time:.3f}秒")
        print(f"  🚀 处理速度: {len(results)/total_time:.1f} 张/秒")
        
    def _print_video_stats(self, stats: Dict, total_time: float):
        """打印视频检测统计"""
        processed_frames = stats['frame_count']
        total_detections = stats['total_detections']
        
        if processed_frames > 0:
            fps = processed_frames / total_time
            avg_detections = total_detections / processed_frames
            
            print(f"\n🎥 视频检测统计:")
            print(f"  🎞️  处理帧数: {processed_frames}")
            print(f"  🌸 总计检测: {total_detections} 个对象")
            print(f"  ⏱️  总时间: {total_time:.2f}秒")
            print(f"  🚀 处理速度: {fps:.1f} FPS")
            print(f"  📈 平均每帧: {avg_detections:.1f} 个对象")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="花草检测YOLO推理脚本")
    
    parser.add_argument(
        '--source', '-s',
        type=str, 
        required=True,
        help='输入源 (图片路径/目录/视频文件/摄像头编号0)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str, 
        default='models/best_flower_model.pt',
        help='模型文件路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='输出目录'
    )
    
    parser.add_argument(
        '--conf', 
        type=float, 
        default=0.25,
        help='置信度阈值'
    )
    
    parser.add_argument(
        '--iou', 
        type=float, 
        default=0.45,
        help='NMS IoU阈值'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        help='计算设备 (auto/cpu/cuda/cuda:0)'
    )
    
    parser.add_argument(
        '--show', 
        action='store_true',
        help='显示检测结果'
    )
    
    parser.add_argument(
        '--save', 
        action='store_true',
        help='保存检测结果'
    )
    
    parser.add_argument(
        '--fps-limit', 
        type=int,
        help='限制视频处理帧率'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    print("🌸 花草检测YOLO推理程序 🌸")
    print("="*50)
    
    try:
        # 创建检测应用
        app = FlowerDetectionApp(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device
        )
        
        # 判断输入类型并执行相应检测
        source = args.source
        
        if source.isdigit():  # 摄像头
            app.detect_video(
                video_path=source,
                output_path=None,
                display=True,
                fps_limit=args.fps_limit
            )
            
        elif os.path.isfile(source):
            # 判断是图片还是视频
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            if any(source.lower().endswith(ext) for ext in video_extensions):
                # 视频文件
                output_path = None
                if args.save:
                    output_path = os.path.join(args.output, f"detected_{os.path.basename(source)}")
                    
                app.detect_video(
                    video_path=source,
                    output_path=output_path,
                    display=args.show,
                    fps_limit=args.fps_limit
                )
            else:
                # 图片文件
                output_path = None
                if args.save:
                    os.makedirs(args.output, exist_ok=True)
                    output_path = os.path.join(args.output, f"detected_{os.path.basename(source)}")
                    
                app.detect_image(
                    image_path=source,
                    save_path=output_path,
                    show=args.show
                )
                
        elif os.path.isdir(source):
            # 图片目录 - 批量检测
            if args.save:
                app.detect_batch(source, args.output)
            else:
                print("批量检测需要使用 --save 参数保存结果")
                
        else:
            raise ValueError(f"无效的输入源: {source}")
            
        print("\n🎉 检测完成! 🎉")
        
    except KeyboardInterrupt:
        print("\n❌ 检测被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 检测失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()