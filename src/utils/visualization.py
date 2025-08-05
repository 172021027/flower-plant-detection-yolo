#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具模块

提供检测结果可视化、训练曲线绘制、统计图表等功能。
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionVisualizer:
    """检测结果可视化器"""
    
    def __init__(self, class_names: Optional[Dict[int, str]] = None, 
                 colors: Optional[List[Tuple[int, int, int]]] = None):
        """
        初始化可视化器
        
        Args:
            class_names: 类别名称字典
            colors: 类别颜色列表
        """
        # 默认花草类别
        self.class_names = class_names or {
            0: "玫瑰",      # Rose
            1: "向日葵",    # Sunflower
            2: "郁金香",    # Tulip
            3: "雏菊",      # Daisy
            4: "百合",      # Lily
            5: "康乃馨",    # Carnation
            6: "兰花",      # Orchid
            7: "牡丹"       # Peony
        }
        
        # 默认颜色方案
        self.colors = colors or [
            (255, 192, 203),  # 玫瑰 - 粉色
            (255, 255, 0),    # 向日葵 - 黄色
            (255, 20, 147),   # 郁金香 - 深粉色
            (255, 255, 255),  # 雏菊 - 白色
            (255, 255, 224),  # 百合 - 浅黄色
            (255, 182, 193),  # 康乃馨 - 浅粉色
            (138, 43, 226),   # 兰花 - 紫色
            (255, 105, 180)   # 牡丹 - 热粉色
        ]
        
        # 确保颜色数量与类别数量匹配
        while len(self.colors) < len(self.class_names):
            self.colors.extend(self.colors)
            
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                       thickness: int = 2, font_scale: float = 0.6) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            thickness: 边界框线条粗细
            font_scale: 字体大小
            
        Returns:
            绘制了检测结果的图像
        """
        # 复制图像以避免修改原图
        result_image = image.copy()
        
        for detection in detections:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, detection['bbox'])
            
            # 获取类别信息
            class_id = detection['class_id']
            class_name = detection.get('class_name_cn', self.class_names.get(class_id, '未知'))
            confidence = detection['confidence']
            
            # 获取颜色
            color = self.colors[class_id % len(self.colors)]
            
            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # 准备标签文本
            label = f"{class_name} {confidence:.2f}"
            
            # 计算文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # 绘制标签背景
            label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            cv2.rectangle(
                result_image,
                (x1, label_y - text_height - baseline),
                (x1 + text_width, label_y + baseline),
                color,
                -1
            )
            
            # 绘制标签文本
            text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
            cv2.putText(
                result_image,
                label,
                (x1, label_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1
            )
            
        return result_image
        
    def add_info_text(self, image: np.ndarray, text: str, 
                     position: str = 'top_left', 
                     font_scale: float = 0.7,
                     color: Tuple[int, int, int] = (255, 255, 255),
                     background_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        在图像上添加信息文本
        
        Args:
            image: 输入图像
            text: 要添加的文本
            position: 文本位置 ('top_left', 'top_right', 'bottom_left', 'bottom_right')
            font_scale: 字体大小
            color: 文本颜色
            background_color: 背景颜色
            
        Returns:
            添加了文本的图像
        """
        h, w = image.shape[:2]
        
        # 计算文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        # 确定文本位置
        margin = 10
        if position == 'top_left':
            x, y = margin, margin + text_height
        elif position == 'top_right':
            x, y = w - text_width - margin, margin + text_height
        elif position == 'bottom_left':
            x, y = margin, h - margin
        elif position == 'bottom_right':
            x, y = w - text_width - margin, h - margin
        else:
            x, y = margin, margin + text_height
            
        # 绘制背景
        cv2.rectangle(
            image,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            background_color,
            -1
        )
        
        # 绘制文本
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1
        )
        
        return image
        
    def create_detection_summary(self, detections: List[Dict[str, Any]], 
                               image_shape: Tuple[int, int],
                               save_path: Optional[str] = None) -> np.ndarray:
        """
        创建检测结果摘要图
        
        Args:
            detections: 检测结果列表
            image_shape: 图像尺寸 (height, width)
            save_path: 保存路径
            
        Returns:
            摘要图像
        """
        # 统计类别分布
        class_counts = {}
        for detection in detections:
            class_id = detection['class_id']
            class_name = detection.get('class_name_cn', self.class_names.get(class_id, '未知'))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        # 创建摘要图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('检测结果摘要', fontsize=16, fontweight='bold')
        
        # 1. 类别分布饼图
        if class_counts:
            axes[0, 0].pie(
                class_counts.values(),
                labels=class_counts.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=[tuple(c/255 for c in self.colors[i % len(self.colors)]) 
                       for i in range(len(class_counts))]
            )
            axes[0, 0].set_title('类别分布')
        else:
            axes[0, 0].text(0.5, 0.5, '无检测结果', ha='center', va='center')
            axes[0, 0].set_title('类别分布')
            
        # 2. 置信度分布直方图
        if detections:
            confidences = [d['confidence'] for d in detections]
            axes[0, 1].hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_xlabel('置信度')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].set_title('置信度分布')
            axes[0, 1].axvline(np.mean(confidences), color='red', linestyle='--', 
                              label=f'平均值: {np.mean(confidences):.3f}')
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, '无检测结果', ha='center', va='center')
            axes[0, 1].set_title('置信度分布')
            
        # 3. 边界框大小分布
        if detections:
            bbox_areas = []
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                area = (x2 - x1) * (y2 - y1)
                bbox_areas.append(area / (image_shape[0] * image_shape[1]))  # 归一化面积
                
            axes[1, 0].hist(bbox_areas, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].set_xlabel('归一化面积')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].set_title('边界框大小分布')
        else:
            axes[1, 0].text(0.5, 0.5, '无检测结果', ha='center', va='center')
            axes[1, 0].set_title('边界框大小分布')
            
        # 4. 检测统计表
        axes[1, 1].axis('off')
        
        stats_text = f"""检测统计信息:
        
总检测数量: {len(detections)}
图像尺寸: {image_shape[1]} × {image_shape[0]}
        
类别统计:"""
        
        for class_name, count in class_counts.items():
            stats_text += f"\n  {class_name}: {count}"
            
        if detections:
            confidences = [d['confidence'] for d in detections]
            stats_text += f"""
            
置信度统计:
  平均值: {np.mean(confidences):.3f}
  最大值: {np.max(confidences):.3f}
  最小值: {np.min(confidences):.3f}"""
            
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"检测摘要已保存: {save_path}")
            
        # 转换为numpy数组
        fig.canvas.draw()
        summary_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        summary_array = summary_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return summary_array


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self):
        """初始化训练可视化器"""
        pass
        
    def plot_training_curves(self, results_dir: str, save_path: Optional[str] = None) -> None:
        """
        绘制训练曲线
        
        Args:
            results_dir: 训练结果目录
            save_path: 保存路径
        """
        # 查找results.csv文件
        results_file = None
        for file_name in ['results.csv', 'train/results.csv']:
            full_path = os.path.join(results_dir, file_name)
            if os.path.exists(full_path):
                results_file = full_path
                break
                
        if not results_file:
            logger.warning(f"在 {results_dir} 中未找到training results文件")
            return
            
        try:
            import pandas as pd
            
            # 读取训练结果
            df = pd.read_csv(results_file)
            df.columns = df.columns.str.strip()  # 去除列名空格
            
            # 创建子图
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('训练过程曲线', fontsize=16, fontweight='bold')
            
            # 绘制损失曲线
            if 'train/box_loss' in df.columns:
                axes[0, 0].plot(df.index, df['train/box_loss'], label='训练', color='blue')
            if 'val/box_loss' in df.columns:
                axes[0, 0].plot(df.index, df['val/box_loss'], label='验证', color='red')
            axes[0, 0].set_title('边界框损失')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 绘制分类损失
            if 'train/cls_loss' in df.columns:
                axes[0, 1].plot(df.index, df['train/cls_loss'], label='训练', color='blue')
            if 'val/cls_loss' in df.columns:
                axes[0, 1].plot(df.index, df['val/cls_loss'], label='验证', color='red')
            axes[0, 1].set_title('分类损失')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 绘制总损失
            if 'train/total_loss' in df.columns:
                axes[0, 2].plot(df.index, df['train/total_loss'], label='训练', color='blue')
            axes[0, 2].set_title('总损失')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 绘制mAP曲线
            if 'metrics/mAP50(B)' in df.columns:
                axes[1, 0].plot(df.index, df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
            if 'metrics/mAP50-95(B)' in df.columns:
                axes[1, 0].plot(df.index, df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange')
            axes[1, 0].set_title('平均精度 (mAP)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 绘制精确率和召回率
            if 'metrics/precision(B)' in df.columns:
                axes[1, 1].plot(df.index, df['metrics/precision(B)'], label='精确率', color='purple')
            if 'metrics/recall(B)' in df.columns:
                axes[1, 1].plot(df.index, df['metrics/recall(B)'], label='召回率', color='brown')
            axes[1, 1].set_title('精确率和召回率')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('值')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 绘制学习率
            if 'lr/pg0' in df.columns:
                axes[1, 2].plot(df.index, df['lr/pg0'], label='学习率', color='red')
            axes[1, 2].set_title('学习率')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图像
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"训练曲线已保存: {save_path}")
            else:
                plt.show()
                
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"绘制训练曲线失败: {e}")
            
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                            class_names: List[str],
                            save_path: Optional[str] = None,
                            normalize: bool = True) -> None:
        """
        绘制混淆矩阵
        
        Args:
            confusion_matrix: 混淆矩阵
            class_names: 类别名称列表
            save_path: 保存路径
            normalize: 是否归一化
        """
        try:
            # 归一化处理
            if normalize:
                cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
                title = '归一化混淆矩阵'
                fmt = '.2f'
            else:
                cm = confusion_matrix
                title = '混淆矩阵'
                fmt = 'd'
                
            # 创建图形
            plt.figure(figsize=(10, 8))
            
            # 绘制热力图
            sns.heatmap(
                cm,
                annot=True,
                fmt=fmt,
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                square=True,
                cbar_kws={'shrink': 0.8}
            )
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('预测类别', fontsize=12)
            plt.ylabel('真实类别', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # 保存图像
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"混淆矩阵已保存: {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制混淆矩阵失败: {e}")


class DatasetVisualizer:
    """数据集可视化器"""
    
    def __init__(self, class_names: Dict[int, str]):
        """
        初始化数据集可视化器
        
        Args:
            class_names: 类别名称字典
        """
        self.class_names = class_names
        
    def plot_class_distribution(self, class_counts: Dict[str, int], 
                              save_path: Optional[str] = None) -> None:
        """
        绘制类别分布图
        
        Args:
            class_counts: 类别计数字典
            save_path: 保存路径
        """
        try:
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('数据集类别分布', fontsize=16, fontweight='bold')
            
            # 条形图
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            bars = ax1.bar(classes, counts, color='skyblue', edgecolor='black', alpha=0.7)
            ax1.set_title('类别计数')
            ax1.set_xlabel('类别')
            ax1.set_ylabel('样本数量')
            ax1.tick_params(axis='x', rotation=45)
            
            # 在条形图上添加数值标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{count}', ha='center', va='bottom')
            
            # 饼图
            ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
            ax2.set_title('类别比例')
            
            plt.tight_layout()
            
            # 保存图像
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"类别分布图已保存: {save_path}")
            else:
                plt.show()
                
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"绘制类别分布图失败: {e}")
            
    def visualize_sample_images(self, images_dir: str, labels_dir: str,
                              num_samples: int = 12,
                              save_path: Optional[str] = None) -> None:
        """
        可视化样本图像
        
        Args:
            images_dir: 图像目录
            labels_dir: 标签目录
            num_samples: 显示的样本数量
            save_path: 保存路径
        """
        try:
            # 获取图像文件列表
            image_files = []
            for ext in ['jpg', 'jpeg', 'png', 'bmp']:
                image_files.extend(Path(images_dir).glob(f"*.{ext}"))
                image_files.extend(Path(images_dir).glob(f"*.{ext.upper()}"))
                
            if not image_files:
                logger.warning(f"在 {images_dir} 中未找到图像文件")
                return
                
            # 随机选择样本
            import random
            random.shuffle(image_files)
            selected_files = image_files[:num_samples]
            
            # 计算子图布局
            cols = 4
            rows = (num_samples + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
            fig.suptitle('数据集样本展示', fontsize=16, fontweight='bold')
            
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, image_file in enumerate(selected_files):
                row, col = i // cols, i % cols
                
                # 读取图像
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 读取标签文件
                label_file = Path(labels_dir) / f"{image_file.stem}.txt"
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        
                    # 绘制边界框
                    h, w = image.shape[:2]
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            # 转换为像素坐标
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            
                            # 绘制边界框
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            # 添加类别标签
                            class_name = self.class_names.get(class_id, f"类别{class_id}")
                            cv2.putText(image, class_name, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # 显示图像
                axes[row, col].imshow(image)
                axes[row, col].set_title(f"{image_file.name}", fontsize=10)
                axes[row, col].axis('off')
                
            # 隐藏多余的子图
            for i in range(len(selected_files), rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].axis('off')
                
            plt.tight_layout()
            
            # 保存图像
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"样本图像已保存: {save_path}")
            else:
                plt.show()
                
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"可视化样本图像失败: {e}")


if __name__ == "__main__":
    # 示例用法
    
    # 创建检测可视化器
    visualizer = DetectionVisualizer()
    
    # 模拟检测结果
    fake_detections = [
        {
            'bbox': [100, 100, 200, 200],
            'confidence': 0.95,
            'class_id': 0,
            'class_name_cn': '玫瑰'
        },
        {
            'bbox': [300, 150, 400, 250],
            'confidence': 0.87,
            'class_id': 1,
            'class_name_cn': '向日葵'
        }
    ]
    
    # 创建模拟图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 绘制检测结果
    result_image = visualizer.draw_detections(test_image, fake_detections)
    
    # 创建检测摘要
    summary = visualizer.create_detection_summary(
        fake_detections, 
        test_image.shape[:2],
        'detection_summary.png'
    )
    
    print("✅ 可视化工具测试完成!")