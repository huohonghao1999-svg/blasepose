#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置文件 - 控制最佳样本选择的各种参数
"""

# 基本配置
NUM_BEST_SAMPLES = 3  # 选择的最佳样本数量
SELECTION_METHOD = 'quality'  # 选择方法: 'quality', 'pck', 'endpoint', 'random'
SAVE_RESULTS = True  # 是否保存结果图像
SHOW_IMAGES = True   # 是否显示图像

# 评估参数
PCK_THRESHOLD = 0.5  # PCK阈值
ENDPOINT_JOINTS = [0, 5, 6, 11]  # 末端关键点索引: 右脚踝, 左脚踝, 右手腕, 左手腕

# 可视化参数
JOINT_RADIUS = 4  # 关节点半径
JOINT_THICKNESS = 3  # 关节点线条粗细
LINE_THICKNESS = 2  # 骨架连线粗细
TEXT_SIZE = 0.6  # 文本大小
TEXT_THICKNESS = 2  # 文本粗细

# 颜色配置 (BGR格式)
PREDICTED_COLOR = (0, 255, 0)  # 预测关节点颜色 (绿色)
GROUND_TRUTH_COLOR = (0, 0, 255)  # 真实关节点颜色 (红色)
SKELETON_COLOR = (0, 255, 0)  # 骨架连线颜色 (绿色)
TEXT_COLOR = (255, 255, 255)  # 文本颜色 (白色)

# 显示参数
DISPLAY_TIME = 2000  # 图像显示时间 (毫秒)
PROGRESS_INTERVAL = 200  # 进度显示间隔

# 输出配置
RESULT_DIR = "./result"  # 结果保存目录
IMAGE_FORMAT = "jpg"  # 图像格式
IMAGE_QUALITY = 95  # 图像质量 (1-100)

# 高级配置
BATCH_SIZE = 50  # 推理批处理大小
ENABLE_PROGRESS_BAR = True  # 是否显示进度条
DETAILED_METRICS = True  # 是否计算详细指标
SAVE_METRICS_JSON = True  # 是否保存指标到JSON文件

# 选择方法说明
SELECTION_METHODS = {
    'quality': '综合质量评分 (PCK + 误差)',
    'pck': '仅基于PCK分数',
    'endpoint': '仅基于末端关键点PCK',
    'random': '随机选择'
}

def get_config_summary():
    """获取配置摘要"""
    return {
        'num_samples': NUM_BEST_SAMPLES,
        'selection_method': SELECTION_METHOD,
        'pck_threshold': PCK_THRESHOLD,
        'endpoint_joints': ENDPOINT_JOINTS,
        'result_dir': RESULT_DIR
    }

def print_config():
    """打印当前配置"""
    print("=== 测试配置 ===")
    print(f"选择样本数量: {NUM_BEST_SAMPLES}")
    print(f"选择方法: {SELECTION_METHOD} - {SELECTION_METHODS[SELECTION_METHOD]}")
    print(f"PCK阈值: {PCK_THRESHOLD}")
    print(f"末端关键点: {ENDPOINT_JOINTS}")
    print(f"结果目录: {RESULT_DIR}")
    print(f"保存结果: {SAVE_RESULTS}")
    print(f"显示图像: {SHOW_IMAGES}")
    print("=" * 20)
