#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本

提供全面的模型性能评估功能。
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 请安装ultralytics库")
    print("运行: pip install ultralytics")
    sys.exit(1)

from src.utils.model_utils import ModelManager, ModelEvaluator, compare_models
from src.utils.visualization import TrainingVisualizer, DetectionVisualizer
from src.models.yolo_detector import FlowerDetector
from src import FLOWER_CLASSES

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_single_model(model_path: str, dataset_yaml: str, 
                         output_dir: str = "evaluation_results",
                         conf_threshold: float = 0.25,
                         iou_threshold: float = 0.45,
                         create_plots: bool = True):
    """
    评估单个模型
    
    Args:
        model_path: 模型文件路径
        dataset_yaml: 数据集配置文件
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        create_plots: 是否创建可视化图表
    """
    print("🌸 花草检测模型评估工具 🌸")
    print("="*50)
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"开始评估模型: {model_path}")
        
        # 1. 加载模型和获取基本信息
        logger.info("获取模型信息...")
        manager = ModelManager(model_path)
        model_info = manager.get_model_info()
        
        print("📊 模型基本信息:")
        for key, value in model_info.items():
            if key != 'parameters':
                print(f"  {key}: {value}")
                
        # 保存模型信息
        model_info_path = os.path.join(output_dir, "model_info.json")
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        print(f"💾 模型信息已保存: {model_info_path}")
        
        # 2. 性能基准测试
        logger.info("进行性能基准测试...")
        benchmark_results = manager.benchmark_model(
            image_size=(640, 640),
            batch_sizes=[1, 4, 8],
            num_runs=50
        )
        
        print("\n🚀 性能基准测试结果:")
        for batch_size, result in benchmark_results['batch_results'].items():
            if 'error' not in result:
                print(f"  批次大小 {batch_size}: {result['fps']:.1f} FPS")
                
        # 保存基准测试结果
        benchmark_path = os.path.join(output_dir, "benchmark_results.json")
        with open(benchmark_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
        print(f"💾 基准测试结果已保存: {benchmark_path}")
        
        # 3. 数据集评估
        logger.info("在数据集上评估模型...")
        model = YOLO(model_path)
        evaluator = ModelEvaluator(model, FLOWER_CLASSES)
        
        eval_results = evaluator.evaluate_on_dataset(
            dataset_yaml,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        print("\n📈 数据集评估结果:")
        print(f"  mAP@0.5: {eval_results['map50']:.4f}")
        print(f"  mAP@0.5:0.95: {eval_results['map50_95']:.4f}")
        print(f"  精确率: {eval_results['precision']:.4f}")
        print(f"  召回率: {eval_results['recall']:.4f}")
        
        if 'class_metrics' in eval_results:
            print(f"\n🏷️ 各类别AP@0.5:")
            for class_name, metrics in eval_results['class_metrics'].items():
                print(f"  {class_name}: {metrics['ap50']:.4f}")
                
        # 保存评估结果
        eval_path = os.path.join(output_dir, "evaluation_results.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        print(f"💾 评估结果已保存: {eval_path}")
        
        # 4. 创建可视化图表
        if create_plots:
            logger.info("创建可视化图表...")
            visualizer = TrainingVisualizer()
            
            # 尝试绘制训练曲线（如果有训练日志）
            model_dir = Path(model_path).parent
            possible_dirs = [
                model_dir / "runs" / "train",
                model_dir.parent / "runs" / "train",
                Path("runs") / "train"
            ]
            
            for train_dir in possible_dirs:
                if train_dir.exists():
                    curves_path = os.path.join(output_dir, "training_curves.png")
                    visualizer.plot_training_curves(str(train_dir), curves_path)
                    break
            else:
                logger.warning("未找到训练日志，跳过训练曲线绘制")
                
            # 混淆矩阵
            try:
                confusion_matrix = evaluator.compute_confusion_matrix(dataset_yaml, conf_threshold)
                if confusion_matrix is not None:
                    confusion_path = os.path.join(output_dir, "confusion_matrix.png")
                    class_names = list(FLOWER_CLASSES.values())
                    visualizer.plot_confusion_matrix(
                        confusion_matrix, 
                        class_names,
                        confusion_path
                    )
                    print(f"📊 混淆矩阵已保存: {confusion_path}")
            except Exception as e:
                logger.warning(f"混淆矩阵生成失败: {e}")
                
        # 5. 生成评估报告
        generate_evaluation_report(
            model_info, benchmark_results, eval_results,
            os.path.join(output_dir, "evaluation_report.md")
        )
        
        print(f"\n✅ 模型评估完成，结果保存在: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"模型评估失败: {e}")
        print(f"❌ 模型评估失败: {e}")
        return False


def compare_multiple_models(model_paths: list, dataset_yaml: str,
                           output_dir: str = "comparison_results"):
    """
    比较多个模型
    
    Args:
        model_paths: 模型文件路径列表
        dataset_yaml: 数据集配置文件
        output_dir: 输出目录
    """
    print("🔍 多模型对比评估")
    print("="*50)
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"开始比较 {len(model_paths)} 个模型...")
        
        # 执行模型比较
        comparison_results = compare_models(model_paths, dataset_yaml)
        
        # 显示比较结果
        print("\n📊 模型比较结果:")
        
        valid_models = {k: v for k, v in comparison_results['models'].items() 
                       if 'error' not in v and 'metrics' in v}
        
        if valid_models:
            print(f"\n📈 性能对比:")
            print(f"{'模型名称':<20} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12} {'精确率':<8} {'召回率':<8}")
            print("-" * 60)
            
            for model_name, model_data in valid_models.items():
                metrics = model_data['metrics']
                print(f"{model_name:<20} {metrics['map50']:<10.4f} "
                      f"{metrics['map50_95']:<12.4f} {metrics['precision']:<8.4f} "
                      f"{metrics['recall']:<8.4f}")
                      
            # 显示最佳模型
            if 'comparison' in comparison_results:
                comp = comparison_results['comparison']
                if 'best_map50' in comp:
                    print(f"\n🏆 最佳模型 (mAP@0.5): {comp['best_map50']['model']} "
                          f"({comp['best_map50']['value']:.4f})")
                if 'best_map50_95' in comp:
                    print(f"🏆 最佳模型 (mAP@0.5:0.95): {comp['best_map50_95']['model']} "
                          f"({comp['best_map50_95']['value']:.4f})")
        else:
            print("❌ 没有成功评估的模型")
            
        # 保存比较结果
        comparison_path = os.path.join(output_dir, "model_comparison.json")
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 比较结果已保存: {comparison_path}")
        
        # 生成比较报告
        generate_comparison_report(
            comparison_results,
            os.path.join(output_dir, "comparison_report.md")
        )
        
        print(f"\n✅ 模型比较完成，结果保存在: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"模型比较失败: {e}")
        print(f"❌ 模型比较失败: {e}")
        return False


def test_model_on_images(model_path: str, images_dir: str, 
                        output_dir: str = "test_results",
                        conf_threshold: float = 0.25,
                        iou_threshold: float = 0.45,
                        save_crops: bool = False):
    """
    在图像上测试模型
    
    Args:
        model_path: 模型文件路径
        images_dir: 测试图像目录
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        save_crops: 是否保存检测区域裁剪
    """
    print("📸 模型图像测试")
    print("="*50)
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        if save_crops:
            crops_dir = os.path.join(output_dir, "crops")
            os.makedirs(crops_dir, exist_ok=True)
            
        # 创建检测器
        detector = FlowerDetector(
            model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # 获取图像文件
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            image_files.extend(Path(images_dir).glob(f"*.{ext}"))
            image_files.extend(Path(images_dir).glob(f"*.{ext.upper()}"))
            
        if not image_files:
            print(f"❌ 在目录 {images_dir} 中未找到图像文件")
            return False
            
        logger.info(f"找到 {len(image_files)} 张测试图像")
        
        # 创建可视化器
        visualizer = DetectionVisualizer()
        
        # 统计信息
        total_detections = 0
        all_detections = []
        
        # 处理每张图像
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 处理: {image_path.name}")
            
            try:
                # 执行检测
                detections = detector.detect(str(image_path), return_crops=save_crops)
                all_detections.extend(detections)
                total_detections += len(detections)
                
                if detections:
                    print(f"  🌸 检测到 {len(detections)} 个对象:")
                    for det in detections:
                        print(f"    - {det['class_name_cn']} (置信度: {det['confidence']:.3f})")
                else:
                    print("  📷 未检测到任何对象")
                    
                # 可视化结果
                import cv2
                image = cv2.imread(str(image_path))
                if image is not None:
                    result_image = visualizer.draw_detections(image, detections)
                    
                    # 保存结果图像
                    output_image_path = os.path.join(output_dir, f"result_{image_path.name}")
                    cv2.imwrite(output_image_path, result_image)
                    
                    # 保存裁剪区域
                    if save_crops and detections:
                        for j, det in enumerate(detections):
                            if det.get('crop') is not None:
                                crop_name = f"{image_path.stem}_{j}_{det['class_name_en']}.jpg"
                                crop_path = os.path.join(crops_dir, crop_name)
                                cv2.imwrite(crop_path, det['crop'])
                                
            except Exception as e:
                logger.error(f"处理图像 {image_path} 失败: {e}")
                continue
                
        # 生成测试报告
        print(f"\n📊 测试结果统计:")
        print(f"  处理图像: {len(image_files)} 张")
        print(f"  总检测数: {total_detections} 个")
        print(f"  平均每张: {total_detections/len(image_files):.1f} 个")
        
        # 保存检测结果
        results_path = os.path.join(output_dir, "detection_results.json")
        detector.save_detection_results(all_detections, results_path, 'json')
        
        # 获取统计信息
        stats = detector.get_detection_statistics(all_detections)
        
        stats_path = os.path.join(output_dir, "test_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            
        print(f"\n✅ 图像测试完成，结果保存在: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"图像测试失败: {e}")
        print(f"❌ 图像测试失败: {e}")
        return False


def generate_evaluation_report(model_info: dict, benchmark_results: dict, 
                             eval_results: dict, output_path: str):
    """生成评估报告"""
    
    report_content = f"""# 花草检测模型评估报告

## 评估概要
- **评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **模型路径**: {model_info.get('model_path', 'N/A')}
- **设备**: {model_info.get('device', 'N/A')}

## 模型信息
- **模型类型**: {model_info.get('model_type', 'YOLOv8')}
- **模型大小**: {model_info.get('model_size_mb', 'N/A')} MB
- **参数数量**: {model_info.get('parameters', {}).get('total', 'N/A')}
- **输入尺寸**: {model_info.get('input_size', [640, 640])}

## 性能基准测试

| 批次大小 | 平均时间(s) | FPS | 吞吐量 |
|---------|------------|-----|--------|
"""
    
    for batch_size, result in benchmark_results.get('batch_results', {}).items():
        if 'error' not in result:
            report_content += f"| {batch_size} | {result['avg_time']:.4f} | {result['fps']:.1f} | {result['throughput']:.1f} |\n"
            
    report_content += f"""
## 数据集评估结果

### 整体性能指标
- **mAP@0.5**: {eval_results.get('map50', 0):.4f}
- **mAP@0.5:0.95**: {eval_results.get('map50_95', 0):.4f}
- **精确率**: {eval_results.get('precision', 0):.4f}
- **召回率**: {eval_results.get('recall', 0):.4f}

### 各类别性能
"""
    
    if 'class_metrics' in eval_results:
        report_content += "| 类别 | AP@0.5 |\n|------|--------|\n"
        for class_name, metrics in eval_results['class_metrics'].items():
            report_content += f"| {class_name} | {metrics['ap50']:.4f} |\n"
            
    report_content += f"""
## 评估设置
- **置信度阈值**: {eval_results.get('conf_threshold', 0.25)}
- **IoU阈值**: {eval_results.get('iou_threshold', 0.45)}

## 结论
基于以上评估结果，该模型在花草检测任务上表现{'良好' if eval_results.get('map50', 0) > 0.7 else '一般' if eval_results.get('map50', 0) > 0.5 else '较差'}。

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"📄 评估报告已保存: {output_path}")


def generate_comparison_report(comparison_results: dict, output_path: str):
    """生成比较报告"""
    
    report_content = f"""# 花草检测模型比较报告

## 比较概要
- **比较时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **参与模型数量**: {len(comparison_results.get('models', {}))}

## 模型性能对比

| 模型 | mAP@0.5 | mAP@0.5:0.95 | 精确率 | 召回率 |
|------|---------|-------------|--------|--------|
"""
    
    valid_models = {k: v for k, v in comparison_results.get('models', {}).items() 
                   if 'error' not in v and 'metrics' in v}
    
    for model_name, model_data in valid_models.items():
        metrics = model_data['metrics']
        report_content += f"| {model_name} | {metrics['map50']:.4f} | {metrics['map50_95']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} |\n"
        
    if 'comparison' in comparison_results:
        comp = comparison_results['comparison']
        report_content += f"""
## 最佳模型

- **mAP@0.5 最佳**: {comp.get('best_map50', {}).get('model', 'N/A')} ({comp.get('best_map50', {}).get('value', 0):.4f})
- **mAP@0.5:0.95 最佳**: {comp.get('best_map50_95', {}).get('model', 'N/A')} ({comp.get('best_map50_95', {}).get('value', 0):.4f})
"""
    
    report_content += f"""
## 结论

根据比较结果，推荐使用性能最佳的模型进行花草检测任务。

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"📄 比较报告已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="花草检测模型评估工具")
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='模型文件路径'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        help='多个模型文件路径（用于比较）'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='data/dataset.yaml',
        help='数据集配置文件路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation_results',
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
        help='IoU阈值'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='不创建可视化图表'
    )
    
    parser.add_argument(
        '--test-images',
        type=str,
        help='测试图像目录'
    )
    
    parser.add_argument(
        '--save-crops',
        action='store_true',
        help='保存检测区域裁剪'
    )
    
    args = parser.parse_args()
    
    try:
        if args.models:
            # 多模型比较
            success = compare_multiple_models(
                args.models,
                args.dataset,
                args.output
            )
        elif args.model:
            if args.test_images:
                # 图像测试
                success = test_model_on_images(
                    args.model,
                    args.test_images,
                    args.output,
                    args.conf,
                    args.iou,
                    args.save_crops
                )
            else:
                # 单模型评估
                success = evaluate_single_model(
                    args.model,
                    args.dataset,
                    args.output,
                    args.conf,
                    args.iou,
                    not args.no_plots
                )
        else:
            print("❌ 请指定要评估的模型文件 (--model) 或多个模型文件 (--models)")
            sys.exit(1)
            
        if not success:
            sys.exit(1)
            
        print("\n🎉 模型评估工具执行完成! 🎉")
        
    except KeyboardInterrupt:
        print("\n❌ 操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()