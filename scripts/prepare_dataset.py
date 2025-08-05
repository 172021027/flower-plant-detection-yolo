#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集准备脚本

自动化数据集下载、处理、划分等操作。
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.data_utils import DatasetManager, validate_dataset, create_dataset_summary

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_dataset(config_path: str, 
                   create_sample_data: bool = False,
                   samples_per_class: int = 50,
                   validate_only: bool = False):
    """
    准备数据集
    
    Args:
        config_path: 数据集配置文件路径
        create_sample_data: 是否创建示例数据
        samples_per_class: 每个类别的示例样本数
        validate_only: 是否仅进行验证
    """
    print("🌸 花草检测数据集准备工具 🌸")
    print("="*50)
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 创建数据集管理器
        manager = DatasetManager(config)
        
        if validate_only:
            # 仅验证数据集
            logger.info("开始验证数据集...")
            is_valid = validate_dataset(config)
            
            if is_valid:
                print("✅ 数据集验证通过!")
                
                # 创建数据集摘要
                summary = create_dataset_summary(config, "dataset_summary.json")
                print(f"📊 数据集摘要已保存: dataset_summary.json")
                
            else:
                print("❌ 数据集验证失败!")
                return False
                
        else:
            # 完整的数据集准备流程
            
            # 1. 创建目录结构
            logger.info("创建数据集目录结构...")
            manager.create_directory_structure()
            
            # 2. 创建示例数据（如果需要）
            if create_sample_data:
                logger.info(f"生成示例数据，每类 {samples_per_class} 个样本...")
                manager.generate_sample_data(samples_per_class)
                
                # 将示例数据划分到训练/验证/测试集
                logger.info("划分示例数据...")
                try:
                    manager.split_dataset(
                        source_images_dir=config['paths']['samples'],
                        source_labels_dir=config['paths']['samples'],
                        train_ratio=0.7,
                        val_ratio=0.2,
                        test_ratio=0.1
                    )
                except Exception as e:
                    logger.warning(f"数据集划分失败: {e}")
                    
            # 3. 创建YOLO配置文件
            logger.info("创建YOLO数据集配置...")
            yaml_path = manager.create_yolo_dataset_yaml()
            print(f"📄 YOLO配置文件已创建: {yaml_path}")
            
            # 4. 验证数据集
            logger.info("验证数据集...")
            is_valid = validate_dataset(config)
            
            if is_valid:
                print("✅ 数据集准备完成并验证通过!")
                
                # 创建数据集摘要
                summary = create_dataset_summary(config, "dataset_summary.json")
                print(f"📊 数据集摘要已保存: dataset_summary.json")
                
            else:
                print("⚠️ 数据集准备完成，但验证未通过，请检查数据质量!")
                
        return True
        
    except Exception as e:
        logger.error(f"数据集准备失败: {e}")
        print(f"❌ 数据集准备失败: {e}")
        return False


def download_sample_images(output_dir: str, num_images_per_class: int = 10):
    """
    下载示例花草图像（模拟功能）
    
    Args:
        output_dir: 输出目录
        num_images_per_class: 每个类别下载的图像数量
    """
    logger.info("开始下载示例花草图像...")
    
    # 花草类别列表
    flower_classes = [
        "玫瑰", "向日葵", "郁金香", "雏菊", 
        "百合", "康乃馨", "兰花", "牡丹"
    ]
    
    print("⚠️  注意: 这是一个模拟的下载功能")
    print("实际使用时，您需要:")
    print("1. 从公开数据集获取图像")
    print("2. 使用网络爬虫收集图像（注意版权）")
    print("3. 使用专业的图像数据集API")
    
    # 创建目录结构
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in flower_classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"📁 创建类别目录: {class_name} (需要 {num_images_per_class} 张图片)")
        
    print(f"✅ 图像下载目录结构已创建: {output_dir}")
    print("\n💡 建议的数据来源:")
    print("- Open Images Dataset")
    print("- COCO Dataset")
    print("- ImageNet")
    print("- Flickr API")
    print("- Unsplash API")


def convert_annotations(input_dir: str, input_format: str, 
                       output_dir: str, output_format: str = "yolo"):
    """
    转换标注格式
    
    Args:
        input_dir: 输入标注目录
        input_format: 输入格式 (coco, pascal_voc, etc.)
        output_dir: 输出目录
        output_format: 输出格式 (yolo)
    """
    logger.info(f"转换标注格式: {input_format} -> {output_format}")
    
    if input_format.lower() == "coco" and output_format.lower() == "yolo":
        # COCO到YOLO格式转换
        print("🔄 COCO -> YOLO 格式转换")
        print("⚠️  注意: 这是一个示例转换功能")
        print("实际转换需要具体的COCO标注文件")
        
    elif input_format.lower() == "pascal_voc" and output_format.lower() == "yolo":
        # Pascal VOC到YOLO格式转换
        print("🔄 Pascal VOC -> YOLO 格式转换")
        print("⚠️  注意: 这是一个示例转换功能")
        print("实际转换需要具体的XML标注文件")
        
    else:
        logger.warning(f"不支持的转换: {input_format} -> {output_format}")
        return False
        
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"✅ 标注转换完成，输出目录: {output_dir}")
    return True


def augment_dataset(images_dir: str, labels_dir: str, 
                   output_images_dir: str, output_labels_dir: str,
                   augmentation_factor: int = 2):
    """
    数据增强
    
    Args:
        images_dir: 原始图像目录
        labels_dir: 原始标签目录
        output_images_dir: 输出图像目录
        output_labels_dir: 输出标签目录
        augmentation_factor: 增强倍数
    """
    logger.info(f"开始数据增强，增强倍数: {augmentation_factor}")
    
    # 创建输出目录
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    print("🔄 数据增强功能")
    print("⚠️  注意: 这是一个示例增强功能")
    print("实际增强需要使用专业的数据增强库，如:")
    print("- Albumentations")
    print("- imgaug")
    print("- Augmentor")
    
    # 这里应该实现实际的数据增强逻辑
    # 包括旋转、缩放、翻转、颜色变换等
    
    print(f"✅ 数据增强完成，输出目录:")
    print(f"  图像: {output_images_dir}")
    print(f"  标签: {output_labels_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="花草检测数据集准备工具")
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/dataset_config.yaml',
        help='数据集配置文件路径'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='创建示例数据'
    )
    
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=50,
        help='每个类别的示例样本数'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='仅验证数据集'
    )
    
    parser.add_argument(
        '--download-samples',
        type=str,
        help='下载示例图像到指定目录'
    )
    
    parser.add_argument(
        '--images-per-class',
        type=int,
        default=10,
        help='每个类别下载的图像数量'
    )
    
    parser.add_argument(
        '--convert-annotations',
        action='store_true',
        help='转换标注格式'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        help='输入目录'
    )
    
    parser.add_argument(
        '--input-format',
        type=str,
        choices=['coco', 'pascal_voc'],
        help='输入标注格式'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='输出目录'
    )
    
    parser.add_argument(
        '--augment-dataset',
        action='store_true',
        help='数据增强'
    )
    
    parser.add_argument(
        '--augmentation-factor',
        type=int,
        default=2,
        help='数据增强倍数'
    )
    
    args = parser.parse_args()
    
    try:
        if args.download_samples:
            # 下载示例图像
            download_sample_images(args.download_samples, args.images_per_class)
            
        elif args.convert_annotations:
            # 转换标注格式
            if not all([args.input_dir, args.input_format, args.output_dir]):
                print("❌ 标注转换需要指定 --input-dir, --input-format, --output-dir")
                sys.exit(1)
                
            convert_annotations(
                args.input_dir, 
                args.input_format, 
                args.output_dir
            )
            
        elif args.augment_dataset:
            # 数据增强
            if not all([args.input_dir, args.output_dir]):
                print("❌ 数据增强需要指定 --input-dir, --output-dir")
                sys.exit(1)
                
            images_dir = os.path.join(args.input_dir, 'images')
            labels_dir = os.path.join(args.input_dir, 'labels')
            output_images_dir = os.path.join(args.output_dir, 'images')
            output_labels_dir = os.path.join(args.output_dir, 'labels')
            
            augment_dataset(
                images_dir, labels_dir,
                output_images_dir, output_labels_dir,
                args.augmentation_factor
            )
            
        else:
            # 默认数据集准备流程
            success = prepare_dataset(
                args.config,
                args.create_sample,
                args.samples_per_class,
                args.validate_only
            )
            
            if not success:
                sys.exit(1)
                
        print("\n🎉 数据集准备工具执行完成! 🎉")
        
    except KeyboardInterrupt:
        print("\n❌ 操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()