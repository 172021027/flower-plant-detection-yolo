#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
花草检测YOLO训练脚本

基于YOLOv8的花草检测模型训练程序，支持8种花草类别。

使用方法:
    python train.py --config config/yolo_config.yaml --data config/dataset_config.yaml
"""

import os
import sys
import argparse
import logging
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    print("错误: 请安装ultralytics库")
    print("运行: pip install ultralytics")
    sys.exit(1)

from src.utils.data_utils import DatasetManager, validate_dataset
from src.utils.model_utils import ModelManager
from src.utils.visualization import TrainingVisualizer


class FlowerTrainer:
    """花草检测YOLO训练器类"""
    
    def __init__(self, config_path: str, dataset_config_path: str):
        """
        初始化训练器
        
        Args:
            config_path: YOLO配置文件路径
            dataset_config_path: 数据集配置文件路径
        """
        self.config = self._load_config(config_path)
        self.dataset_config = self._load_config(dataset_config_path)
        self.model = None
        self.device = self._setup_device()
        self.save_dir = None
        
        # 设置日志
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"配置文件未找到: {config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            self.logger.error(f"配置文件格式错误: {e}")
            sys.exit(1)
            
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        gpu_id = self.config.get('device', {}).get('gpu_id', 0)
        
        if gpu_id == -1 or not torch.cuda.is_available():
            device = torch.device('cpu')
            self.logger.info("使用CPU进行训练")
        else:
            device = torch.device(f'cuda:{gpu_id}')
            self.logger.info(f"使用GPU进行训练: {device}")
            
        return device
        
    def _setup_logging(self):
        """设置日志系统"""
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 配置日志
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def prepare_dataset(self) -> str:
        """准备数据集"""
        self.logger.info("开始准备数据集...")
        
        # 创建数据集管理器
        dataset_manager = DatasetManager(self.dataset_config)
        
        # 验证数据集
        is_valid = validate_dataset(self.dataset_config)
        if not is_valid:
            self.logger.error("数据集验证失败")
            sys.exit(1)
            
        # 创建YOLO格式的数据集配置文件
        dataset_yaml = dataset_manager.create_yolo_dataset_yaml()
        
        self.logger.info(f"数据集准备完成: {dataset_yaml}")
        return dataset_yaml
        
    def initialize_model(self) -> YOLO:
        """初始化YOLO模型"""
        model_config = self.config['model']
        model_name = model_config['name']
        pretrained = model_config.get('pretrained', True)
        
        self.logger.info(f"初始化YOLO模型: {model_name}")
        
        try:
            if pretrained:
                # 使用预训练模型
                model = YOLO(f"{model_name}.pt")
                self.logger.info(f"加载预训练模型: {model_name}.pt")
            else:
                # 从配置创建模型
                model = YOLO(f"{model_name}.yaml")
                self.logger.info(f"从配置创建模型: {model_name}.yaml")
                
            return model
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            sys.exit(1)
            
    def train(self, dataset_yaml: str):
        """开始训练"""
        self.logger.info("开始训练模型...")
        
        # 初始化模型
        self.model = self.initialize_model()
        
        # 获取训练参数
        training_config = self.config['training']
        device_config = self.config['device']
        logging_config = self.config['logging']
        
        # 设置训练参数
        train_args = {
            'data': dataset_yaml,
            'epochs': training_config['epochs'],
            'batch': training_config['batch_size'],
            'imgsz': self.config['image']['size'],
            'lr0': training_config['learning_rate'],
            'momentum': training_config['momentum'],
            'weight_decay': training_config['weight_decay'],
            'warmup_epochs': training_config['warmup_epochs'],
            'warmup_momentum': training_config['warmup_momentum'],
            'warmup_bias_lr': training_config['warmup_bias_lr'],
            'device': self.device,
            'workers': device_config.get('workers', 8),
            'amp': device_config.get('amp', True),
            'project': logging_config['project'],
            'name': logging_config['name'],
            'exist_ok': logging_config.get('exist_ok', False),
            'verbose': logging_config.get('verbose', True),
            'val': True,
            'plots': True,
            'save': True,
            'save_period': self.config['validation'].get('save_period', -1)
        }
        
        # 添加数据增强参数
        aug_config = self.config['augmentation']
        train_args.update({
            'hsv_h': aug_config.get('hsv_h', 0.015),
            'hsv_s': aug_config.get('hsv_s', 0.7),
            'hsv_v': aug_config.get('hsv_v', 0.4),
            'degrees': aug_config.get('degrees', 0.0),
            'translate': aug_config.get('translate', 0.1),
            'scale': aug_config.get('scale', 0.5),
            'shear': aug_config.get('shear', 0.0),
            'perspective': aug_config.get('perspective', 0.0),
            'flipud': aug_config.get('flipud', 0.0),
            'fliplr': aug_config.get('fliplr', 0.5),
            'mosaic': aug_config.get('mosaic', 1.0),
            'mixup': aug_config.get('mixup', 0.0)
        })
        
        try:
            # 开始训练
            self.logger.info("训练参数:")
            for key, value in train_args.items():
                self.logger.info(f"  {key}: {value}")
                
            results = self.model.train(**train_args)
            
            # 保存训练结果路径
            self.save_dir = results.save_dir
            
            self.logger.info(f"训练完成! 结果保存在: {self.save_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {e}")
            raise
            
    def validate_model(self):
        """验证模型性能"""
        if self.model is None:
            self.logger.error("模型未初始化，无法进行验证")
            return
            
        self.logger.info("开始验证模型...")
        
        try:
            # 验证模型
            val_results = self.model.val(
                data=self.dataset_yaml,
                device=self.device,
                plots=True,
                save_json=True
            )
            
            self.logger.info("验证完成!")
            self.logger.info(f"mAP50: {val_results.box.map50:.4f}")
            self.logger.info(f"mAP50-95: {val_results.box.map:.4f}")
            
            return val_results
            
        except Exception as e:
            self.logger.error(f"验证过程中出现错误: {e}")
            
    def save_model(self, output_path: Optional[str] = None):
        """保存训练好的模型"""
        if self.model is None:
            self.logger.error("模型未初始化，无法保存")
            return
            
        if output_path is None:
            output_path = "models/best_flower_model.pt"
            
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # 保存最佳模型
            best_model_path = self.save_dir / "weights" / "best.pt"
            if best_model_path.exists():
                import shutil
                shutil.copy2(best_model_path, output_path)
                self.logger.info(f"模型已保存到: {output_path}")
            else:
                self.logger.warning("未找到最佳模型文件")
                
        except Exception as e:
            self.logger.error(f"保存模型时出现错误: {e}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="花草检测YOLO训练脚本")
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/yolo_config.yaml',
        help='YOLO配置文件路径'
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='config/dataset_config.yaml',
        help='数据集配置文件路径'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='models/best_flower_model.pt',
        help='输出模型路径'
    )
    
    parser.add_argument(
        '--validate', 
        action='store_true',
        help='训练后进行验证'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    print("🌸 花草检测YOLO训练程序 🌸")
    print("="*50)
    
    try:
        # 创建训练器
        trainer = FlowerTrainer(args.config, args.data)
        
        # 准备数据集
        dataset_yaml = trainer.prepare_dataset()
        trainer.dataset_yaml = dataset_yaml
        
        # 开始训练
        results = trainer.train(dataset_yaml)
        
        # 验证模型 (如果指定)
        if args.validate:
            trainer.validate_model()
            
        # 保存模型
        trainer.save_model(args.output)
        
        print("\n🎉 训练完成! 🎉")
        print(f"📁 训练结果保存在: {trainer.save_dir}")
        print(f"💾 最佳模型保存在: {args.output}")
        
    except KeyboardInterrupt:
        print("\n❌ 训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()