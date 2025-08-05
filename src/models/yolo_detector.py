#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO花草检测器类

封装YOLOv8模型，提供专门的花草检测功能。
"""

import os
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 请安装ultralytics库")
    print("运行: pip install ultralytics")
    raise

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowerDetector:
    """花草检测器类"""
    
    def __init__(self, model_path: str, device: str = 'auto',
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        初始化花草检测器
        
        Args:
            model_path: 模型文件路径
            device: 计算设备 ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 花草类别定义
        self.class_names_cn = {
            0: "玫瑰",      # Rose
            1: "向日葵",    # Sunflower
            2: "郁金香",    # Tulip
            3: "雏菊",      # Daisy
            4: "百合",      # Lily
            5: "康乃馨",    # Carnation
            6: "兰花",      # Orchid
            7: "牡丹"       # Peony
        }
        
        self.class_names_en = {
            0: "Rose",
            1: "Sunflower", 
            2: "Tulip",
            3: "Daisy",
            4: "Lily",
            5: "Carnation",
            6: "Orchid", 
            7: "Peony"
        }
        
        # 加载模型
        self.model = self._load_model()
        
        # 预热模型
        self._warmup_model()
        
    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda:0'
                logger.info(f"自动选择GPU设备: {device}")
            else:
                device = 'cpu'
                logger.info("自动选择CPU设备")
        else:
            logger.info(f"使用指定设备: {device}")
            
        return device
        
    def _load_model(self) -> YOLO:
        """加载YOLO模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
                
            logger.info(f"正在加载花草检测模型: {self.model_path}")
            
            # 加载模型
            model = YOLO(self.model_path)
            
            # 设置设备
            model.to(self.device)
            
            logger.info("✅ 花草检测模型加载成功")
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
            
    def _warmup_model(self, input_size: Tuple[int, int] = (640, 640)):
        """预热模型以获得更准确的性能测量"""
        try:
            logger.info("正在预热模型...")
            
            # 创建随机输入进行预热
            dummy_image = np.random.randint(0, 255, (input_size[1], input_size[0], 3), dtype=np.uint8)
            
            # 预热运行
            for _ in range(3):
                _ = self.model(dummy_image, verbose=False)
                
            logger.info("✅ 模型预热完成")
            
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
            
    def detect(self, image: Union[str, np.ndarray], 
              return_crops: bool = False) -> List[Dict[str, Any]]:
        """
        检测花草对象
        
        Args:
            image: 输入图像 (文件路径或numpy数组)
            return_crops: 是否返回裁剪的检测区域
            
        Returns:
            检测结果列表
        """
        try:
            # 加载图像
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"图像文件不存在: {image}")
                img = cv2.imread(image)
                if img is None:
                    raise ValueError(f"无法读取图像: {image}")
            else:
                img = image.copy()
                
            # 执行检测
            results = self.model(
                img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # 处理检测结果
            detections = self._process_results(results, img, return_crops)
            
            return detections
            
        except Exception as e:
            logger.error(f"检测过程中出现错误: {e}")
            raise
            
    def detect_batch(self, images: List[Union[str, np.ndarray]], 
                    batch_size: int = 4) -> List[List[Dict[str, Any]]]:
        """
        批量检测花草对象
        
        Args:
            images: 图像列表 (文件路径或numpy数组)
            batch_size: 批处理大小
            
        Returns:
            批量检测结果列表
        """
        try:
            logger.info(f"开始批量检测 {len(images)} 张图像，批次大小: {batch_size}")
            
            all_results = []
            
            # 分批处理
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                logger.info(f"处理批次 {i//batch_size + 1}/{(len(images)-1)//batch_size + 1}")
                
                batch_results = []
                for image in batch_images:
                    try:
                        detections = self.detect(image)
                        batch_results.append(detections)
                    except Exception as e:
                        logger.error(f"批次检测失败: {e}")
                        batch_results.append([])
                        
                all_results.extend(batch_results)
                
            logger.info("✅ 批量检测完成")
            return all_results
            
        except Exception as e:
            logger.error(f"批量检测过程中出现错误: {e}")
            raise
            
    def _process_results(self, results, original_image: np.ndarray, 
                        return_crops: bool = False) -> List[Dict[str, Any]]:
        """处理YOLO检测结果"""
        detections = []
        
        if results and len(results) > 0:
            result = results[0]  # 取第一个结果
            
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    
                    # 获取置信度和类别
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # 获取类别名称
                    class_name_cn = self.class_names_cn.get(class_id, "未知")
                    class_name_en = self.class_names_en.get(class_id, "Unknown")
                    
                    # 计算边界框中心点和尺寸
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    detection = {
                        'id': i,
                        'bbox': [x1, y1, x2, y2],
                        'bbox_center': [center_x, center_y],
                        'bbox_size': [width, height],
                        'bbox_area': area,
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name_cn': class_name_cn,
                        'class_name_en': class_name_en
                    }
                    
                    # 添加裁剪区域（如果需要）
                    if return_crops:
                        try:
                            x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
                            x2_int, y2_int = min(original_image.shape[1], int(x2)), min(original_image.shape[0], int(y2))
                            
                            if x2_int > x1_int and y2_int > y1_int:
                                crop = original_image[y1_int:y2_int, x1_int:x2_int]
                                detection['crop'] = crop
                            else:
                                detection['crop'] = None
                        except Exception as e:
                            logger.warning(f"裁剪检测区域失败: {e}")
                            detection['crop'] = None
                    
                    detections.append(detection)
                    
        return detections
        
    def filter_detections(self, detections: List[Dict[str, Any]], 
                         min_confidence: float = None,
                         target_classes: List[int] = None,
                         min_area: float = None) -> List[Dict[str, Any]]:
        """
        过滤检测结果
        
        Args:
            detections: 检测结果列表
            min_confidence: 最小置信度阈值
            target_classes: 目标类别ID列表
            min_area: 最小区域面积
            
        Returns:
            过滤后的检测结果
        """
        filtered_detections = []
        
        for detection in detections:
            # 置信度过滤
            if min_confidence is not None and detection['confidence'] < min_confidence:
                continue
                
            # 类别过滤
            if target_classes is not None and detection['class_id'] not in target_classes:
                continue
                
            # 面积过滤
            if min_area is not None and detection['bbox_area'] < min_area:
                continue
                
            filtered_detections.append(detection)
            
        return filtered_detections
        
    def get_detection_statistics(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取检测结果统计信息
        
        Args:
            detections: 检测结果列表
            
        Returns:
            统计信息字典
        """
        if not detections:
            return {
                'total_detections': 0,
                'class_distribution': {},
                'confidence_stats': {},
                'bbox_stats': {}
            }
            
        # 统计类别分布
        class_counts = {}
        for detection in detections:
            class_name = detection['class_name_cn']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        # 统计置信度
        confidences = [d['confidence'] for d in detections]
        confidence_stats = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'median': float(np.median(confidences))
        }
        
        # 统计边界框信息
        areas = [d['bbox_area'] for d in detections]
        widths = [d['bbox_size'][0] for d in detections]
        heights = [d['bbox_size'][1] for d in detections]
        
        bbox_stats = {
            'area_stats': {
                'mean': float(np.mean(areas)),
                'std': float(np.std(areas)),
                'min': float(np.min(areas)),
                'max': float(np.max(areas))
            },
            'width_stats': {
                'mean': float(np.mean(widths)),
                'std': float(np.std(widths)),
                'min': float(np.min(widths)),
                'max': float(np.max(widths))
            },
            'height_stats': {
                'mean': float(np.mean(heights)),
                'std': float(np.std(heights)),
                'min': float(np.min(heights)),
                'max': float(np.max(heights))
            }
        }
        
        return {
            'total_detections': len(detections),
            'class_distribution': class_counts,
            'confidence_stats': confidence_stats,
            'bbox_stats': bbox_stats
        }
        
    def benchmark_inference(self, image_size: Tuple[int, int] = (640, 640),
                          num_runs: int = 100) -> Dict[str, float]:
        """
        推理性能基准测试
        
        Args:
            image_size: 测试图像尺寸
            num_runs: 运行次数
            
        Returns:
            性能统计信息
        """
        logger.info(f"开始推理性能测试，图像尺寸: {image_size}, 运行次数: {num_runs}")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # 预热
        for _ in range(10):
            _ = self.detect(test_image)
            
        # 性能测试
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            _ = self.detect(test_image)
            end_time = time.time()
            
            inference_time = end_time - start_time
            times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                logger.info(f"完成 {i + 1}/{num_runs} 次测试")
                
        # 计算统计信息
        times = np.array(times)
        stats = {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'fps': 1.0 / float(np.mean(times)),
            'throughput': num_runs / float(np.sum(times))
        }
        
        logger.info(f"✅ 性能测试完成 - 平均FPS: {stats['fps']:.1f}")
        return stats
        
    def save_detection_results(self, detections: List[Dict[str, Any]], 
                             output_path: str, format: str = 'json') -> None:
        """
        保存检测结果
        
        Args:
            detections: 检测结果列表
            output_path: 输出文件路径
            format: 保存格式 ('json', 'csv', 'txt')
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if format.lower() == 'json':
                import json
                
                # 移除不能序列化的numpy数组
                serializable_detections = []
                for detection in detections:
                    det_copy = detection.copy()
                    if 'crop' in det_copy:
                        del det_copy['crop']  # 移除图像数据
                    serializable_detections.append(det_copy)
                    
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_detections, f, ensure_ascii=False, indent=2)
                    
            elif format.lower() == 'csv':
                import pandas as pd
                
                # 转换为DataFrame
                data = []
                for detection in detections:
                    row = {
                        'id': detection['id'],
                        'x1': detection['bbox'][0],
                        'y1': detection['bbox'][1],
                        'x2': detection['bbox'][2],
                        'y2': detection['bbox'][3],
                        'confidence': detection['confidence'],
                        'class_id': detection['class_id'],
                        'class_name_cn': detection['class_name_cn'],
                        'class_name_en': detection['class_name_en'],
                        'width': detection['bbox_size'][0],
                        'height': detection['bbox_size'][1],
                        'area': detection['bbox_area']
                    }
                    data.append(row)
                    
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                
            elif format.lower() == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for detection in detections:
                        line = f"{detection['class_id']} {detection['confidence']:.6f} "
                        line += f"{detection['bbox'][0]:.2f} {detection['bbox'][1]:.2f} "
                        line += f"{detection['bbox'][2]:.2f} {detection['bbox'][3]:.2f}\n"
                        f.write(line)
                        
            else:
                raise ValueError(f"不支持的保存格式: {format}")
                
            logger.info(f"检测结果已保存: {output_path}")
            
        except Exception as e:
            logger.error(f"保存检测结果失败: {e}")
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'model_path': self.model_path,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'num_classes': len(self.class_names_cn),
            'class_names_cn': self.class_names_cn,
            'class_names_en': self.class_names_en
        }
        
        # 尝试获取模型详细信息
        try:
            if hasattr(self.model, 'model'):
                model = self.model.model
                if hasattr(model, 'yaml'):
                    info['model_yaml'] = model.yaml
                    
        except Exception as e:
            logger.warning(f"获取模型详细信息失败: {e}")
            
        return info


if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) > 2:
        model_path = sys.argv[1]
        image_path = sys.argv[2]
        
        # 创建检测器
        detector = FlowerDetector(model_path)
        
        # 获取模型信息
        info = detector.get_model_info()
        print("📊 模型信息:")
        for key, value in info.items():
            if key not in ['class_names_cn', 'class_names_en']:
                print(f"  {key}: {value}")
                
        # 执行检测
        print(f"\n🔍 检测图像: {image_path}")
        detections = detector.detect(image_path, return_crops=True)
        
        # 显示结果
        if detections:
            print(f"✅ 检测到 {len(detections)} 个花草对象:")
            for det in detections:
                print(f"  - {det['class_name_cn']} (置信度: {det['confidence']:.3f})")
        else:
            print("❌ 未检测到任何花草对象")
            
        # 获取统计信息
        stats = detector.get_detection_statistics(detections)
        print(f"\n📈 检测统计:")
        print(f"  总数: {stats['total_detections']}")
        print(f"  类别分布: {stats['class_distribution']}")
        
    else:
        print("用法: python yolo_detector.py <model_path> <image_path>")