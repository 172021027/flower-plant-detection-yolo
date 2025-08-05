#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据下载脚本

提供花草图像数据的下载和收集功能。
"""

import os
import sys
import argparse
import requests
import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FlowerDataDownloader:
    """花草数据下载器"""
    
    def __init__(self, output_dir: str = "downloaded_data", 
                 max_images_per_class: int = 100):
        """
        初始化下载器
        
        Args:
            output_dir: 输出目录
            max_images_per_class: 每个类别的最大图像数量
        """
        self.output_dir = Path(output_dir)
        self.max_images_per_class = max_images_per_class
        
        # 花草类别定义
        self.flower_classes = {
            "rose": "玫瑰",
            "sunflower": "向日葵", 
            "tulip": "郁金香",
            "daisy": "雏菊",
            "lily": "百合",
            "carnation": "康乃馨",
            "orchid": "兰花",
            "peony": "牡丹"
        }
        
        # 创建目录结构
        self._create_directories()
        
        # 会话设置
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def _create_directories(self):
        """创建目录结构"""
        self.output_dir.mkdir(exist_ok=True)
        
        # 为每个类别创建目录
        for class_en, class_cn in self.flower_classes.items():
            class_dir = self.output_dir / class_en
            class_dir.mkdir(exist_ok=True)
            
        # 创建元数据目录
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
    def download_from_urls_file(self, urls_file: str, class_name: str) -> int:
        """
        从URL文件下载图像
        
        Args:
            urls_file: URL文件路径
            class_name: 类别名称
            
        Returns:
            下载成功的图像数量
        """
        if not os.path.exists(urls_file):
            logger.error(f"URL文件不存在: {urls_file}")
            return 0
            
        logger.info(f"从文件下载 {class_name} 类别图像: {urls_file}")
        
        try:
            with open(urls_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
                
            return self._download_images_from_urls(urls, class_name)
            
        except Exception as e:
            logger.error(f"读取URL文件失败: {e}")
            return 0
            
    def download_from_csv(self, csv_file: str, url_column: str = 'url', 
                         class_column: str = 'class') -> int:
        """
        从CSV文件下载图像
        
        Args:
            csv_file: CSV文件路径
            url_column: URL列名
            class_column: 类别列名
            
        Returns:
            下载成功的图像数量
        """
        if not os.path.exists(csv_file):
            logger.error(f"CSV文件不存在: {csv_file}")
            return 0
            
        logger.info(f"从CSV文件下载图像: {csv_file}")
        
        total_downloaded = 0
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # 按类别分组URL
                class_urls = {}
                for row in reader:
                    if url_column in row and class_column in row:
                        class_name = row[class_column].lower()
                        if class_name in self.flower_classes:
                            if class_name not in class_urls:
                                class_urls[class_name] = []
                            class_urls[class_name].append(row[url_column])
                            
                # 下载每个类别的图像
                for class_name, urls in class_urls.items():
                    downloaded = self._download_images_from_urls(urls, class_name)
                    total_downloaded += downloaded
                    
            return total_downloaded
            
        except Exception as e:
            logger.error(f"处理CSV文件失败: {e}")
            return 0
            
    def _download_images_from_urls(self, urls: List[str], class_name: str) -> int:
        """
        从URL列表下载图像
        
        Args:
            urls: URL列表
            class_name: 类别名称
            
        Returns:
            下载成功的图像数量
        """
        class_dir = self.output_dir / class_name
        downloaded_count = 0
        failed_urls = []
        
        # 限制下载数量
        urls = urls[:self.max_images_per_class]
        
        logger.info(f"开始下载 {class_name} 类别，共 {len(urls)} 个URL")
        
        for i, url in enumerate(urls, 1):
            try:
                # 生成文件名
                file_extension = self._get_file_extension(url)
                filename = f"{class_name}_{i:04d}{file_extension}"
                filepath = class_dir / filename
                
                # 跳过已存在的文件
                if filepath.exists():
                    logger.info(f"跳过已存在的文件: {filename}")
                    downloaded_count += 1
                    continue
                    
                # 下载图像
                success = self._download_single_image(url, filepath)
                if success:
                    downloaded_count += 1
                    logger.info(f"[{i}/{len(urls)}] 下载成功: {filename}")
                else:
                    failed_urls.append(url)
                    logger.warning(f"[{i}/{len(urls)}] 下载失败: {url}")
                    
                # 添加延迟避免过于频繁的请求
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                logger.info("下载被用户中断")
                break
            except Exception as e:
                logger.error(f"下载URL {url} 时出错: {e}")
                failed_urls.append(url)
                continue
                
        # 保存失败的URL
        if failed_urls:
            failed_file = self.output_dir / "metadata" / f"{class_name}_failed_urls.txt"
            with open(failed_file, 'w', encoding='utf-8') as f:
                for url in failed_urls:
                    f.write(url + '\n')
            logger.info(f"失败的URL已保存到: {failed_file}")
            
        logger.info(f"类别 {class_name} 下载完成: {downloaded_count}/{len(urls)} 成功")
        return downloaded_count
        
    def _download_single_image(self, url: str, filepath: Path, 
                              timeout: int = 10, max_size: int = 10*1024*1024) -> bool:
        """
        下载单张图像
        
        Args:
            url: 图像URL
            filepath: 保存路径
            timeout: 超时时间(秒)
            max_size: 最大文件大小(字节)
            
        Returns:
            是否下载成功
        """
        try:
            response = self.session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/jpeg', 'image/png', 'image/jpg']):
                logger.warning(f"非图像内容类型: {content_type}")
                return False
                
            # 检查文件大小
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > max_size:
                logger.warning(f"文件过大: {content_length} bytes")
                return False
                
            # 下载文件
            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 检查下载大小
                        if downloaded > max_size:
                            logger.warning(f"下载文件过大: {downloaded} bytes")
                            filepath.unlink()  # 删除部分下载的文件
                            return False
                            
            # 验证下载的文件
            if filepath.stat().st_size == 0:
                filepath.unlink()
                return False
                
            # 简单的图像验证
            if not self._validate_image(filepath):
                filepath.unlink()
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"下载失败: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
            
    def _validate_image(self, filepath: Path) -> bool:
        """
        验证图像文件
        
        Args:
            filepath: 图像文件路径
            
        Returns:
            是否为有效图像
        """
        try:
            # 尝试使用PIL验证图像
            from PIL import Image
            with Image.open(filepath) as img:
                img.verify()
            return True
        except Exception:
            try:
                # 尝试使用OpenCV验证图像
                import cv2
                img = cv2.imread(str(filepath))
                return img is not None
            except Exception:
                return False
                
    def _get_file_extension(self, url: str) -> str:
        """从URL获取文件扩展名"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        if path.endswith('.jpg') or path.endswith('.jpeg'):
            return '.jpg'
        elif path.endswith('.png'):
            return '.png'
        elif path.endswith('.bmp'):
            return '.bmp'
        else:
            return '.jpg'  # 默认扩展名
            
    def create_sample_urls_file(self, output_file: str = "sample_urls.txt"):
        """
        创建示例URL文件
        
        Args:
            output_file: 输出文件路径
        """
        sample_urls = [
            "# 这是一个示例URL文件",
            "# 每行一个图像URL",
            "# 以#开头的行为注释",
            "",
            "# 玫瑰图像示例URL (请替换为实际的图像URL)",
            "https://example.com/rose1.jpg",
            "https://example.com/rose2.jpg",
            "",
            "# 向日葵图像示例URL",
            "https://example.com/sunflower1.jpg", 
            "https://example.com/sunflower2.jpg",
            "",
            "# 注意事项:",
            "# 1. 确保URL指向有效的图像文件",
            "# 2. 请遵守网站的使用条款和版权规定",
            "# 3. 避免过于频繁的请求，以免被封禁",
            "# 4. 建议使用公开的数据集API"
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in sample_urls:
                f.write(line + '\n')
                
        logger.info(f"示例URL文件已创建: {output_file}")
        print(f"✅ 示例URL文件已创建: {output_file}")
        print("📝 请编辑此文件，添加实际的图像URL")
        
    def create_sample_csv(self, output_file: str = "sample_dataset.csv"):
        """
        创建示例CSV文件
        
        Args:
            output_file: 输出文件路径
        """
        sample_data = [
            ["url", "class", "description"],
            ["https://example.com/rose1.jpg", "rose", "Red rose flower"],
            ["https://example.com/rose2.jpg", "rose", "Pink rose flower"],
            ["https://example.com/sunflower1.jpg", "sunflower", "Yellow sunflower"],
            ["https://example.com/sunflower2.jpg", "sunflower", "Large sunflower"],
            ["https://example.com/tulip1.jpg", "tulip", "Red tulip"],
            ["https://example.com/daisy1.jpg", "daisy", "White daisy"],
            ["https://example.com/lily1.jpg", "lily", "White lily"],
            ["https://example.com/carnation1.jpg", "carnation", "Pink carnation"],
            ["https://example.com/orchid1.jpg", "orchid", "Purple orchid"],
            ["https://example.com/peony1.jpg", "peony", "Pink peony"]
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in sample_data:
                writer.writerow(row)
                
        logger.info(f"示例CSV文件已创建: {output_file}")
        print(f"✅ 示例CSV文件已创建: {output_file}")
        print("📝 请编辑此文件，添加实际的图像URL和标签")
        
    def generate_download_statistics(self) -> Dict[str, Any]:
        """
        生成下载统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_images': 0,
            'classes': {},
            'file_sizes': [],
            'extensions': {}
        }
        
        for class_en, class_cn in self.flower_classes.items():
            class_dir = self.output_dir / class_en
            
            if class_dir.exists():
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(class_dir.glob(ext))
                    
                class_count = len(image_files)
                stats['classes'][class_cn] = class_count
                stats['total_images'] += class_count
                
                # 统计文件大小和扩展名
                for image_file in image_files:
                    try:
                        size = image_file.stat().st_size
                        stats['file_sizes'].append(size)
                        
                        ext = image_file.suffix.lower()
                        stats['extensions'][ext] = stats['extensions'].get(ext, 0) + 1
                    except Exception:
                        continue
                        
        # 计算文件大小统计
        if stats['file_sizes']:
            import numpy as np
            stats['size_stats'] = {
                'mean': np.mean(stats['file_sizes']),
                'median': np.median(stats['file_sizes']),
                'min': np.min(stats['file_sizes']),
                'max': np.max(stats['file_sizes']),
                'total': np.sum(stats['file_sizes'])
            }
        else:
            stats['size_stats'] = {}
            
        return stats
        
    def print_download_report(self):
        """打印下载报告"""
        stats = self.generate_download_statistics()
        
        print("\n📊 下载统计报告")
        print("="*50)
        print(f"📁 输出目录: {self.output_dir}")
        print(f"🖼️  总图像数: {stats['total_images']}")
        
        if stats['classes']:
            print(f"\n🏷️ 各类别统计:")
            for class_name, count in stats['classes'].items():
                print(f"  {class_name}: {count} 张")
                
        if stats['extensions']:
            print(f"\n📄 文件格式分布:")
            for ext, count in stats['extensions'].items():
                print(f"  {ext}: {count} 张")
                
        if stats['size_stats']:
            size_stats = stats['size_stats']
            print(f"\n💾 文件大小统计:")
            print(f"  平均大小: {size_stats['mean']/1024/1024:.2f} MB")
            print(f"  总大小: {size_stats['total']/1024/1024:.2f} MB")
            print(f"  最大文件: {size_stats['max']/1024/1024:.2f} MB")
            print(f"  最小文件: {size_stats['min']/1024:.2f} KB")
            
        # 保存统计信息
        stats_file = self.output_dir / "metadata" / "download_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            # 转换numpy类型为原生Python类型
            if 'size_stats' in stats:
                for key, value in stats['size_stats'].items():
                    if hasattr(value, 'item'):
                        stats['size_stats'][key] = value.item()
                        
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n💾 统计信息已保存: {stats_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="花草图像数据下载工具")
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='downloaded_data',
        help='输出目录'
    )
    
    parser.add_argument(
        '--max-per-class',
        type=int,
        default=100,
        help='每个类别的最大图像数量'
    )
    
    parser.add_argument(
        '--urls-file',
        type=str,
        help='包含图像URL的文件路径'
    )
    
    parser.add_argument(
        '--class-name',
        type=str,
        help='URL文件对应的类别名称'
    )
    
    parser.add_argument(
        '--csv-file',
        type=str,
        help='包含图像URL和标签的CSV文件路径'
    )
    
    parser.add_argument(
        '--url-column',
        type=str,
        default='url',
        help='CSV文件中的URL列名'
    )
    
    parser.add_argument(
        '--class-column', 
        type=str,
        default='class',
        help='CSV文件中的类别列名'
    )
    
    parser.add_argument(
        '--create-sample-urls',
        action='store_true',
        help='创建示例URL文件'
    )
    
    parser.add_argument(
        '--create-sample-csv',
        action='store_true',
        help='创建示例CSV文件'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='仅显示下载统计信息'
    )
    
    args = parser.parse_args()
    
    print("🌸 花草图像数据下载工具 🌸")
    print("="*50)
    
    try:
        # 创建下载器
        downloader = FlowerDataDownloader(
            args.output,
            args.max_per_class
        )
        
        if args.create_sample_urls:
            # 创建示例URL文件
            downloader.create_sample_urls_file()
            
        elif args.create_sample_csv:
            # 创建示例CSV文件
            downloader.create_sample_csv()
            
        elif args.stats_only:
            # 仅显示统计信息
            downloader.print_download_report()
            
        elif args.urls_file and args.class_name:
            # 从URL文件下载
            downloaded = downloader.download_from_urls_file(
                args.urls_file,
                args.class_name
            )
            print(f"\n✅ 下载完成，成功下载 {downloaded} 张图像")
            downloader.print_download_report()
            
        elif args.csv_file:
            # 从CSV文件下载
            downloaded = downloader.download_from_csv(
                args.csv_file,
                args.url_column,
                args.class_column
            )
            print(f"\n✅ 下载完成，成功下载 {downloaded} 张图像")
            downloader.print_download_report()
            
        else:
            # 显示帮助信息
            print("💡 使用说明:")
            print("1. 创建示例文件:")
            print("   python download_data.py --create-sample-urls")
            print("   python download_data.py --create-sample-csv")
            print("")
            print("2. 从URL文件下载:")
            print("   python download_data.py --urls-file urls.txt --class-name rose")
            print("")
            print("3. 从CSV文件下载:")
            print("   python download_data.py --csv-file dataset.csv")
            print("")
            print("4. 查看下载统计:")
            print("   python download_data.py --stats-only")
            print("")
            print("⚠️  注意事项:")
            print("- 请确保遵守网站的使用条款和版权规定")
            print("- 建议使用公开的数据集或API")
            print("- 避免过于频繁的请求以免被封禁")
            
        print("\n🎉 数据下载工具执行完成! 🎉")
        
    except KeyboardInterrupt:
        print("\n❌ 操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()