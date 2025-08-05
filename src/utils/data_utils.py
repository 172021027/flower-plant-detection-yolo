#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理工具模块

提供数据集管理、处理、验证等功能。
"""

import os
import shutil
import random
import yaml
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据集管理器
        
        Args:
            config: 数据集配置字典
        """
        self.config = config
        self.classes = config['classes']['names']
        self.class_names_en = config['classes']['english_names']
        self.paths = config['paths']
        self.statistics = config.get('statistics', {})
        
    def create_directory_structure(self):
        """创建数据集目录结构"""
        logger.info("创建数据集目录结构...")
        
        # 创建主要目录
        directories = [
            self.paths['root'],
            self.paths['raw_data'],
            self.paths['processed_data'],
            self.paths['annotations'],
            self.paths['samples'],
            self.paths['train']['images'],
            self.paths['train']['labels'],
            self.paths['val']['images'],
            self.paths['val']['labels'],
            self.paths['test']['images'],
            self.paths['test']['labels']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✅ 创建目录: {directory}")
            
        logger.info("目录结构创建完成!")
        
    def create_yolo_dataset_yaml(self) -> str:
        """
        创建YOLO格式的数据集配置文件
        
        Returns:
            生成的YAML文件路径
        """
        # YOLO数据集配置
        yolo_config = {
            'path': os.path.abspath(self.paths['root']),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': list(self.classes.values())
        }
        
        # 保存配置文件
        yaml_path = os.path.join(self.paths['root'], 'dataset.yaml')
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yolo_config, f, default_flow_style=False, allow_unicode=True)
            
        logger.info(f"YOLO数据集配置已保存: {yaml_path}")
        return yaml_path
        
    def split_dataset(self, source_images_dir: str, source_labels_dir: str,
                     train_ratio: float = 0.7, val_ratio: float = 0.2, 
                     test_ratio: float = 0.1, random_seed: int = 42):
        """
        划分数据集为训练、验证和测试集
        
        Args:
            source_images_dir: 源图像目录
            source_labels_dir: 源标签目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
        """
        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("训练、验证、测试集比例之和必须等于1")
            
        logger.info(f"开始划分数据集 (训练:{train_ratio}, 验证:{val_ratio}, 测试:{test_ratio})")
        
        # 获取所有图像文件
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            image_files.extend(Path(source_images_dir).glob(f"*.{ext}"))
            image_files.extend(Path(source_images_dir).glob(f"*.{ext.upper()}"))
            
        if not image_files:
            raise ValueError(f"在目录 {source_images_dir} 中未找到图像文件")
            
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        # 随机打乱
        random.seed(random_seed)
        random.shuffle(image_files)
        
        # 计算划分点
        total_files = len(image_files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        # 划分文件列表
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        logger.info(f"数据集划分: 训练集 {len(train_files)}, 验证集 {len(val_files)}, 测试集 {len(test_files)}")
        
        # 复制文件到相应目录
        splits = [
            ('train', train_files, self.paths['train']),
            ('val', val_files, self.paths['val']),
            ('test', test_files, self.paths['test'])
        ]
        
        for split_name, files, paths in splits:
            logger.info(f"复制 {split_name} 数据...")
            
            for image_file in files:
                # 复制图像文件
                dst_image = os.path.join(paths['images'], image_file.name)
                shutil.copy2(image_file, dst_image)
                
                # 复制对应的标签文件
                label_file = Path(source_labels_dir) / f"{image_file.stem}.txt"
                if label_file.exists():
                    dst_label = os.path.join(paths['labels'], label_file.name)
                    shutil.copy2(label_file, dst_label)
                else:
                    logger.warning(f"标签文件不存在: {label_file}")
                    
        logger.info("数据集划分完成!")
        
    def generate_sample_data(self, num_samples_per_class: int = 10):
        """
        生成示例数据（用于测试）
        
        Args:
            num_samples_per_class: 每个类别的样本数量
        """
        logger.info(f"生成示例数据，每类 {num_samples_per_class} 个样本...")
        
        samples_dir = self.paths['samples']
        os.makedirs(samples_dir, exist_ok=True)
        
        # 为每个类别生成示例图像和标注
        for class_id, class_name in self.classes.items():
            logger.info(f"生成类别 {class_id}: {class_name}")
            
            for i in range(num_samples_per_class):
                # 生成示例图像 (640x640 彩色图像)
                image = self._generate_sample_image(class_id)
                
                # 保存图像
                image_name = f"{class_name}_{i:03d}.jpg"
                image_path = os.path.join(samples_dir, image_name)
                cv2.imwrite(image_path, image)
                
                # 生成对应的YOLO格式标注
                label_name = f"{class_name}_{i:03d}.txt"
                label_path = os.path.join(samples_dir, label_name)
                
                with open(label_path, 'w', encoding='utf-8') as f:
                    # 生成随机边界框
                    x_center = random.uniform(0.2, 0.8)
                    y_center = random.uniform(0.2, 0.8)
                    width = random.uniform(0.1, 0.4)
                    height = random.uniform(0.1, 0.4)
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
        logger.info(f"示例数据生成完成，保存在: {samples_dir}")
        
    def _generate_sample_image(self, class_id: int, size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        生成示例图像
        
        Args:
            class_id: 类别ID
            size: 图像尺寸
            
        Returns:
            生成的图像数组
        """
        # 基于类别ID生成不同颜色的图像
        colors = [
            (255, 192, 203),  # 玫瑰 - 粉色
            (255, 255, 0),    # 向日葵 - 黄色
            (255, 20, 147),   # 郁金香 - 深粉色
            (255, 255, 255),  # 雏菊 - 白色
            (255, 255, 224),  # 百合 - 浅黄色
            (255, 182, 193),  # 康乃馨 - 浅粉色
            (138, 43, 226),   # 兰花 - 紫色
            (255, 105, 180)   # 牡丹 - 热粉色
        ]
        
        # 创建基础图像
        image = np.full((size[1], size[0], 3), (50, 100, 50), dtype=np.uint8)  # 深绿背景
        
        # 添加花朵区域
        color = colors[class_id % len(colors)]
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = random.randint(80, 150)
        
        cv2.circle(image, (center_x, center_y), radius, color, -1)
        
        # 添加一些噪声和纹理
        noise = np.random.normal(0, 20, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
        
    def validate_annotations(self, images_dir: str, labels_dir: str) -> Dict[str, Any]:
        """
        验证标注文件
        
        Args:
            images_dir: 图像目录
            labels_dir: 标签目录
            
        Returns:
            验证结果统计
        """
        logger.info("开始验证标注文件...")
        
        stats = {
            'total_images': 0,
            'total_labels': 0,
            'matched_pairs': 0,
            'invalid_labels': [],
            'missing_labels': [],
            'class_distribution': defaultdict(int),
            'bbox_stats': {'valid': 0, 'invalid': 0}
        }
        
        # 获取所有图像文件
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            image_files.extend(Path(images_dir).glob(f"*.{ext}"))
            image_files.extend(Path(images_dir).glob(f"*.{ext.upper()}"))
            
        stats['total_images'] = len(image_files)
        
        for image_file in image_files:
            # 查找对应的标签文件
            label_file = Path(labels_dir) / f"{image_file.stem}.txt"
            
            if not label_file.exists():
                stats['missing_labels'].append(str(image_file))
                continue
                
            stats['total_labels'] += 1
            stats['matched_pairs'] += 1
            
            # 验证标签文件内容
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) != 5:
                        stats['invalid_labels'].append({
                            'file': str(label_file),
                            'line': line_num,
                            'error': f'期望5个值，得到{len(parts)}个'
                        })
                        continue
                        
                    try:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        # 验证类别ID
                        if class_id not in self.classes:
                            stats['invalid_labels'].append({
                                'file': str(label_file),
                                'line': line_num,
                                'error': f'无效的类别ID: {class_id}'
                            })
                            continue
                            
                        # 验证边界框坐标
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                               0 < width <= 1 and 0 < height <= 1):
                            stats['invalid_labels'].append({
                                'file': str(label_file),
                                'line': line_num,
                                'error': f'边界框坐标超出范围: {x_center}, {y_center}, {width}, {height}'
                            })
                            stats['bbox_stats']['invalid'] += 1
                        else:
                            stats['bbox_stats']['valid'] += 1
                            
                        stats['class_distribution'][class_id] += 1
                        
                    except ValueError as e:
                        stats['invalid_labels'].append({
                            'file': str(label_file),
                            'line': line_num,
                            'error': f'数值解析错误: {e}'
                        })
                        
            except Exception as e:
                stats['invalid_labels'].append({
                    'file': str(label_file),
                    'line': 0,
                    'error': f'文件读取错误: {e}'
                })
                
        # 打印验证结果
        self._print_validation_stats(stats)
        
        return stats
        
    def _print_validation_stats(self, stats: Dict[str, Any]):
        """打印验证统计信息"""
        print("\n📊 数据集验证结果:")
        print(f"  🖼️  总图像数: {stats['total_images']}")
        print(f"  🏷️  总标签数: {stats['total_labels']}")
        print(f"  ✅ 匹配对数: {stats['matched_pairs']}")
        print(f"  ❌ 缺失标签: {len(stats['missing_labels'])}")
        print(f"  ⚠️  无效标签: {len(stats['invalid_labels'])}")
        
        print(f"\n📈 类别分布:")
        for class_id, count in stats['class_distribution'].items():
            class_name = self.classes.get(class_id, f"未知({class_id})")
            print(f"  {class_name}: {count} 个标注")
            
        print(f"\n📦 边界框统计:")
        print(f"  ✅ 有效边界框: {stats['bbox_stats']['valid']}")
        print(f"  ❌ 无效边界框: {stats['bbox_stats']['invalid']}")
        
        if stats['invalid_labels']:
            print(f"\n⚠️  详细错误信息 (前10个):")
            for error in stats['invalid_labels'][:10]:
                print(f"  📄 {error['file']}:{error['line']} - {error['error']}")
            if len(stats['invalid_labels']) > 10:
                print(f"  ... 还有 {len(stats['invalid_labels']) - 10} 个错误")


def validate_dataset(config: Dict[str, Any]) -> bool:
    """
    验证整个数据集
    
    Args:
        config: 数据集配置
        
    Returns:
        验证是否通过
    """
    logger.info("开始验证数据集...")
    
    try:
        # 创建数据集管理器
        manager = DatasetManager(config)
        
        # 检查目录结构
        paths = config['paths']
        required_dirs = [
            paths['train']['images'],
            paths['train']['labels'],
            paths['val']['images'], 
            paths['val']['labels']
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                logger.error(f"必需目录不存在: {directory}")
                return False
                
        # 验证训练集
        logger.info("验证训练集...")
        train_stats = manager.validate_annotations(
            paths['train']['images'],
            paths['train']['labels']
        )
        
        # 验证验证集
        logger.info("验证验证集...")
        val_stats = manager.validate_annotations(
            paths['val']['images'],
            paths['val']['labels']
        )
        
        # 检查是否有足够的数据
        min_samples = 10  # 每个数据集最少样本数
        
        if train_stats['matched_pairs'] < min_samples:
            logger.error(f"训练集样本数不足: {train_stats['matched_pairs']} < {min_samples}")
            return False
            
        if val_stats['matched_pairs'] < min_samples:
            logger.error(f"验证集样本数不足: {val_stats['matched_pairs']} < {min_samples}")
            return False
            
        # 检查类别分布
        train_classes = set(train_stats['class_distribution'].keys())
        val_classes = set(val_stats['class_distribution'].keys())
        
        if not train_classes:
            logger.error("训练集中没有任何标注")
            return False
            
        if not val_classes:
            logger.error("验证集中没有任何标注")
            return False
            
        logger.info("✅ 数据集验证通过!")
        return True
        
    except Exception as e:
        logger.error(f"数据集验证失败: {e}")
        return False


def create_dataset_summary(config: Dict[str, Any], output_path: str = "dataset_summary.json"):
    """
    创建数据集摘要报告
    
    Args:
        config: 数据集配置
        output_path: 输出文件路径
    """
    logger.info("创建数据集摘要报告...")
    
    manager = DatasetManager(config)
    
    summary = {
        'dataset_info': {
            'name': config['dataset']['name'],
            'version': config['dataset']['version'], 
            'description': config['dataset']['description'],
            'created_time': str(datetime.now())
        },
        'classes': {
            'total_classes': len(manager.classes),
            'class_names': dict(manager.classes),
            'english_names': dict(manager.class_names_en)
        },
        'statistics': {}
    }
    
    # 统计各个数据集
    for split in ['train', 'val', 'test']:
        if split in config['paths']:
            images_dir = config['paths'][split]['images']
            labels_dir = config['paths'][split]['labels']
            
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                stats = manager.validate_annotations(images_dir, labels_dir)
                summary['statistics'][split] = {
                    'total_images': stats['total_images'],
                    'total_labels': stats['total_labels'],
                    'matched_pairs': stats['matched_pairs'],
                    'class_distribution': dict(stats['class_distribution']),
                    'valid_bboxes': stats['bbox_stats']['valid'],
                    'invalid_bboxes': stats['bbox_stats']['invalid']
                }
                
    # 保存摘要
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    logger.info(f"数据集摘要已保存: {output_path}")
    return summary


if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 创建数据集管理器
        manager = DatasetManager(config)
        
        # 创建目录结构
        manager.create_directory_structure()
        
        # 生成示例数据
        manager.generate_sample_data(num_samples_per_class=5)
        
        # 创建YOLO配置
        yaml_path = manager.create_yolo_dataset_yaml()
        
        # 验证数据集
        validate_dataset(config)
        
        print(f"✅ 数据集初始化完成!")
        print(f"YOLO配置文件: {yaml_path}")
    else:
        print("用法: python data_utils.py <dataset_config.yaml>")