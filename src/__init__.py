"""
花草检测YOLO项目

基于YOLOv8的花草检测系统，支持8种花草类别识别。

Authors: 花草检测项目组
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "花草检测项目组"
__description__ = "基于YOLOv8的花草检测系统"

# 支持的花草类别
FLOWER_CLASSES = {
    0: "玫瑰",      # Rose
    1: "向日葵",    # Sunflower
    2: "郁金香",    # Tulip
    3: "雏菊",      # Daisy
    4: "百合",      # Lily
    5: "康乃馨",    # Carnation
    6: "兰花",      # Orchid
    7: "牡丹"       # Peony
}

FLOWER_CLASSES_EN = {
    0: "Rose",
    1: "Sunflower", 
    2: "Tulip",
    3: "Daisy",
    4: "Lily",
    5: "Carnation",
    6: "Orchid", 
    7: "Peony"
}