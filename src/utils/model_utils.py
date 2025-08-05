#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工具模块

提供模型管理、评估、转换等功能。
"""

import os
import json
import time
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    print("警告: ultralytics库未安装，某些功能可能不可用")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化模型管理器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.device = self._detect_device()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def _detect_device(self) -> str:
        """检测可用的计算设备"""
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            gpu_name = torch.cuda.get_device_name()
            logger.info(f"检测到GPU: {gpu_name}")
        else:
            device = "cpu"
            logger.info("使用CPU计算")
            
        return device
        
    def load_model(self, model_path: str) -> YOLO:
        """
        加载YOLO模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的YOLO模型
        """
        try:
            logger.info(f"正在加载模型: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
                
            self.model = YOLO(model_path)
            self.model_path = model_path
            
            logger.info("✅ 模型加载成功")
            return self.model
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if self.model is None:
            raise ValueError("模型未加载")
            
        try:
            # 获取模型基本信息
            model_info = {
                'model_path': self.model_path,
                'device': self.device,
                'model_type': 'YOLOv8',
                'framework': 'PyTorch'
            }
            
            # 获取模型架构信息
            if hasattr(self.model, 'model'):
                model_info.update({
                    'parameters': self._count_parameters(),
                    'model_size_mb': self._get_model_size(),
                    'input_size': self._get_input_size(),
                    'num_classes': self._get_num_classes()
                })
                
            # 获取训练信息 (如果有)
            if hasattr(self.model, 'ckpt') and self.model.ckpt:
                ckpt = self.model.ckpt
                if 'epoch' in ckpt:
                    model_info['training_epochs'] = ckpt['epoch']
                if 'best_fitness' in ckpt:
                    model_info['best_fitness'] = float(ckpt['best_fitness'])
                    
            return model_info
            
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {'error': str(e)}
            
    def _count_parameters(self) -> Dict[str, int]:
        """统计模型参数数量"""
        if not hasattr(self.model, 'model'):
            return {}
            
        total_params = 0
        trainable_params = 0
        
        for param in self.model.model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
        
    def _get_model_size(self) -> float:
        """获取模型文件大小(MB)"""
        if self.model_path and os.path.exists(self.model_path):
            size_bytes = os.path.getsize(self.model_path)
            return round(size_bytes / (1024 * 1024), 2)
        return 0.0
        
    def _get_input_size(self) -> List[int]:
        """获取模型输入尺寸"""
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'args'):
                imgsz = getattr(self.model.model.args, 'imgsz', 640)
                if isinstance(imgsz, (list, tuple)):
                    return list(imgsz)
                else:
                    return [imgsz, imgsz]
            return [640, 640]  # 默认尺寸
        except:
            return [640, 640]
            
    def _get_num_classes(self) -> int:
        """获取类别数量"""
        try:
            if hasattr(self.model, 'model'):
                if hasattr(self.model.model, 'nc'):
                    return self.model.model.nc
                elif hasattr(self.model.model, 'yaml') and 'nc' in self.model.model.yaml:
                    return self.model.model.yaml['nc']
            return 8  # 默认花草类别数
        except:
            return 8
            
    def benchmark_model(self, image_size: Tuple[int, int] = (640, 640), 
                       batch_sizes: List[int] = [1, 4, 8, 16],
                       num_runs: int = 100) -> Dict[str, Any]:
        """
        模型性能基准测试
        
        Args:
            image_size: 输入图像尺寸
            batch_sizes: 测试的批次大小列表
            num_runs: 每个批次的运行次数
            
        Returns:
            基准测试结果
        """
        if self.model is None:
            raise ValueError("模型未加载")
            
        logger.info("开始模型性能基准测试...")
        
        results = {
            'device': self.device,
            'image_size': image_size,
            'num_runs': num_runs,
            'batch_results': {},
            'system_info': self._get_system_info()
        }
        
        for batch_size in batch_sizes:
            logger.info(f"测试批次大小: {batch_size}")
            
            try:
                # 创建随机输入
                dummy_input = torch.randn(
                    batch_size, 3, image_size[1], image_size[0]
                ).to(self.device)
                
                # 预热
                for _ in range(10):
                    with torch.no_grad():
                        _ = self.model.model(dummy_input)
                        
                # 同步GPU（如果使用）
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                # 性能测试
                times = []
                
                for _ in range(num_runs):
                    start_time = time.time()
                    
                    with torch.no_grad():
                        _ = self.model.model(dummy_input)
                        
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
                # 计算统计信息
                times = np.array(times)
                batch_result = {
                    'batch_size': batch_size,
                    'avg_time': float(np.mean(times)),
                    'std_time': float(np.std(times)),
                    'min_time': float(np.min(times)),
                    'max_time': float(np.max(times)),
                    'fps': batch_size / float(np.mean(times)),
                    'throughput': batch_size * num_runs / float(np.sum(times))
                }
                
                results['batch_results'][batch_size] = batch_result
                
                logger.info(f"批次 {batch_size}: {batch_result['fps']:.1f} FPS")
                
            except Exception as e:
                logger.error(f"批次 {batch_size} 测试失败: {e}")
                results['batch_results'][batch_size] = {'error': str(e)}
                
        return results
        
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': f"{torch.__version__}",
            'torch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_available': True,
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_names': [torch.cuda.get_device_name(i) 
                            for i in range(torch.cuda.device_count())],
                'gpu_memory_gb': [
                    round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                    for i in range(torch.cuda.device_count())
                ]
            })
        else:
            info['cuda_available'] = False
            
        return info
        
    def profile_model(self, input_size: Tuple[int, int] = (640, 640)) -> Dict[str, Any]:
        """
        分析模型结构和计算复杂度
        
        Args:
            input_size: 输入尺寸
            
        Returns:
            模型分析结果
        """
        if self.model is None:
            raise ValueError("模型未加载")
            
        logger.info("分析模型结构...")
        
        try:
            # 基本信息
            profile_info = {
                'model_summary': self.get_model_info(),
                'input_size': input_size,
                'layers': []
            }
            
            # 分析模型层
            if hasattr(self.model, 'model'):
                model = self.model.model
                
                # 获取模型结构
                layer_info = []
                total_params = 0
                
                for name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # 叶子节点
                        params = sum(p.numel() for p in module.parameters())
                        total_params += params
                        
                        layer_info.append({
                            'name': name,
                            'type': type(module).__name__,
                            'parameters': params
                        })
                        
                profile_info['layers'] = layer_info
                profile_info['total_parameters'] = total_params
                
            return profile_info
            
        except Exception as e:
            logger.error(f"模型分析失败: {e}")
            return {'error': str(e)}
            
    def convert_model(self, output_format: str, output_path: str = None) -> str:
        """
        转换模型格式
        
        Args:
            output_format: 输出格式 (onnx, tensorrt, coreml, etc.)
            output_path: 输出路径
            
        Returns:
            转换后的模型路径
        """
        if self.model is None:
            raise ValueError("模型未加载")
            
        supported_formats = ['onnx', 'tensorrt', 'coreml', 'tflite', 'edgetpu', 'tfjs']
        
        if output_format.lower() not in supported_formats:
            raise ValueError(f"不支持的格式: {output_format}. 支持的格式: {supported_formats}")
            
        logger.info(f"转换模型到 {output_format} 格式...")
        
        try:
            # 执行转换
            result = self.model.export(format=output_format)
            
            # 如果指定了输出路径，移动文件
            if output_path and result != output_path:
                import shutil
                shutil.move(result, output_path)
                result = output_path
                
            logger.info(f"✅ 模型转换完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"模型转换失败: {e}")
            raise
            
    def save_model_info(self, output_path: str = "model_info.json"):
        """
        保存模型信息到文件
        
        Args:
            output_path: 输出文件路径
        """
        if self.model is None:
            raise ValueError("模型未加载")
            
        try:
            # 收集所有信息
            info = {
                'basic_info': self.get_model_info(),
                'profile': self.profile_model(),
                'system_info': self._get_system_info()
            }
            
            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
                
            logger.info(f"模型信息已保存: {output_path}")
            
        except Exception as e:
            logger.error(f"保存模型信息失败: {e}")
            raise


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: YOLO, class_names: Dict[int, str]):
        """
        初始化评估器
        
        Args:
            model: YOLO模型
            class_names: 类别名称字典
        """
        self.model = model
        self.class_names = class_names
        
    def evaluate_on_dataset(self, dataset_yaml: str, conf_threshold: float = 0.25,
                           iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        在数据集上评估模型
        
        Args:
            dataset_yaml: 数据集配置文件
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            
        Returns:
            评估结果
        """
        logger.info("开始模型评估...")
        
        try:
            # 执行验证
            results = self.model.val(
                data=dataset_yaml,
                conf=conf_threshold,
                iou=iou_threshold,
                plots=True,
                save_json=True
            )
            
            # 提取关键指标
            metrics = {
                'map50': float(results.box.map50),
                'map50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold
            }
            
            # 按类别统计
            if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
                class_metrics = {}
                ap50_values = results.box.ap50
                
                for i, ap50 in enumerate(ap50_values):
                    if i in self.class_names:
                        class_metrics[self.class_names[i]] = {
                            'ap50': float(ap50),
                            'class_id': i
                        }
                        
                metrics['class_metrics'] = class_metrics
                
            logger.info(f"✅ 评估完成 - mAP50: {metrics['map50']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            raise
            
    def compute_confusion_matrix(self, dataset_yaml: str, conf_threshold: float = 0.25) -> np.ndarray:
        """
        计算混淆矩阵
        
        Args:
            dataset_yaml: 数据集配置文件
            conf_threshold: 置信度阈值
            
        Returns:
            混淆矩阵
        """
        logger.info("计算混淆矩阵...")
        
        try:
            # 执行验证并生成混淆矩阵
            results = self.model.val(
                data=dataset_yaml,
                conf=conf_threshold,
                plots=True,
                save_json=True
            )
            
            # 获取混淆矩阵
            if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
                confusion_matrix = results.confusion_matrix.matrix
                logger.info("✅ 混淆矩阵计算完成")
                return confusion_matrix
            else:
                logger.warning("无法获取混淆矩阵")
                return None
                
        except Exception as e:
            logger.error(f"混淆矩阵计算失败: {e}")
            raise


def compare_models(model_paths: List[str], dataset_yaml: str) -> Dict[str, Any]:
    """
    比较多个模型的性能
    
    Args:
        model_paths: 模型文件路径列表
        dataset_yaml: 数据集配置文件
        
    Returns:
        模型比较结果
    """
    logger.info(f"开始比较 {len(model_paths)} 个模型...")
    
    results = {
        'models': {},
        'comparison': {}
    }
    
    # 从数据集配置获取类别信息
    import yaml
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    class_names = {i: name for i, name in enumerate(dataset_config.get('names', []))}
    
    # 评估每个模型
    for i, model_path in enumerate(model_paths):
        model_name = f"model_{i+1}_{Path(model_path).stem}"
        
        try:
            logger.info(f"评估模型: {model_name}")
            
            # 加载模型
            model = YOLO(model_path)
            evaluator = ModelEvaluator(model, class_names)
            
            # 获取模型信息
            manager = ModelManager(model_path)
            model_info = manager.get_model_info()
            
            # 评估性能
            eval_results = evaluator.evaluate_on_dataset(dataset_yaml)
            
            # 合并结果
            results['models'][model_name] = {
                'path': model_path,
                'info': model_info,
                'metrics': eval_results
            }
            
        except Exception as e:
            logger.error(f"模型 {model_name} 评估失败: {e}")
            results['models'][model_name] = {'error': str(e)}
            
    # 生成比较结果
    valid_models = {k: v for k, v in results['models'].items() 
                   if 'error' not in v and 'metrics' in v}
    
    if len(valid_models) > 1:
        # 找出最佳模型
        best_map50 = max(valid_models.values(), 
                        key=lambda x: x['metrics']['map50'])
        best_map50_95 = max(valid_models.values(), 
                           key=lambda x: x['metrics']['map50_95'])
        
        results['comparison'] = {
            'best_map50': {
                'model': [k for k, v in valid_models.items() 
                         if v['metrics']['map50'] == best_map50['metrics']['map50']][0],
                'value': best_map50['metrics']['map50']
            },
            'best_map50_95': {
                'model': [k for k, v in valid_models.items() 
                         if v['metrics']['map50_95'] == best_map50_95['metrics']['map50_95']][0],
                'value': best_map50_95['metrics']['map50_95']
            }
        }
        
    logger.info("✅ 模型比较完成")
    return results


if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
        # 创建模型管理器
        manager = ModelManager(model_path)
        
        # 获取模型信息
        info = manager.get_model_info()
        print("📊 模型信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
        # 保存模型信息
        manager.save_model_info("model_analysis.json")
        
        print("✅ 模型分析完成!")
    else:
        print("用法: python model_utils.py <model_path>")