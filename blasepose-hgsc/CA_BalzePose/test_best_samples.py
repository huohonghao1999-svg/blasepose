#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级测试脚本 - 选择效果最好的样本进行可视化
支持多种评估指标和配置选项
"""

import os

# 禁用有问题的MKL优化
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少警告信息

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform
import cv2
import json
from model import BlazePose
from config import total_epoch, train_mode, eval_mode, epoch_to_test
from data import test_dataset, label, data
from test_config import *

# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    if system == "Windows":
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    elif system == "Darwin":  # macOS
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
    else:  # Linux
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei']
    
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 12

setup_chinese_font()

def Eclidian2(a, b):
    """计算欧几里得距离的平方"""
    assert len(a) == len(b)
    summer = 0
    for i in range(len(a)):
        summer += (a[i] - b[i]) ** 2
    return summer

def calculate_sample_metrics(predicted, ground_truth, sample_idx):
    """计算单个样本的详细评估指标"""
    metrics = {}
    
    # 1. PCK分数 (Percentage of Correct Keypoints)
    pck_h = Eclidian2(ground_truth[12], ground_truth[13])  # 头部到颈部距离作为参考
    correct_joints = 0
    joint_errors = []
    
    for j in range(14):
        pck_j = Eclidian2(predicted[j][0:2], ground_truth[j][0:2])
        joint_errors.append(np.sqrt(pck_j))  # 实际像素距离
        if pck_j <= pck_h * 0.5:  # PCK阈值0.5
            correct_joints += 1
    
    metrics['pck_score'] = correct_joints / 14
    metrics['mean_error'] = np.mean(joint_errors)
    metrics['max_error'] = np.max(joint_errors)
    metrics['joint_errors'] = joint_errors
    
    # 2. 末端关键点特殊评估 (手腕和脚踝)
    endpoint_joints = [0, 5, 6, 11]  # 右脚踝, 左脚踝, 右手腕, 左手腕
    endpoint_errors = [joint_errors[i] for i in endpoint_joints]
    metrics['endpoint_mean_error'] = np.mean(endpoint_errors)
    metrics['endpoint_pck'] = sum(1 for err in endpoint_errors if err <= np.sqrt(pck_h * 0.5)) / len(endpoint_joints)
    
    # 3. 整体质量评分 (结合PCK和误差)
    quality_score = metrics['pck_score'] * 0.7 + (1 - metrics['mean_error'] / 100) * 0.3
    metrics['quality_score'] = quality_score
    
    return metrics

def select_best_samples(predictions, ground_truths, num_samples=3, selection_method='quality'):
    """选择效果最好的样本
    
    Args:
        predictions: 预测结果数组
        ground_truths: 真实标签数组
        num_samples: 选择的样本数量
        selection_method: 选择方法 ('quality', 'pck', 'endpoint', 'random')
    """
    print(f"正在评估 {len(predictions)} 个样本...")
    
    sample_metrics = []
    
    for i in range(len(predictions)):
        metrics = calculate_sample_metrics(predictions[i], ground_truths[i], i)
        sample_metrics.append((i, metrics))
        
        if (i + 1) % 200 == 0:
            print(f"已处理 {i + 1}/{len(predictions)} 个样本")
    
    # 根据选择方法排序
    if selection_method == 'quality':
        sample_metrics.sort(key=lambda x: x[1]['quality_score'], reverse=True)
    elif selection_method == 'pck':
        sample_metrics.sort(key=lambda x: x[1]['pck_score'], reverse=True)
    elif selection_method == 'endpoint':
        sample_metrics.sort(key=lambda x: x[1]['endpoint_pck'], reverse=True)
    elif selection_method == 'random':
        np.random.shuffle(sample_metrics)
    
    return sample_metrics[:num_samples]

def visualize_sample(sample_idx, predicted, ground_truth, metrics, save_path):
    """可视化单个样本"""
    img = data[sample_idx].astype(np.uint8)
    
    # 绘制预测的关节点
    for j in range(14):
        cv2.circle(img, center=tuple(predicted[j][0:2]), radius=JOINT_RADIUS, 
                  color=PREDICTED_COLOR, thickness=JOINT_THICKNESS)
    
    # 绘制真实关节点
    for j in range(14):
        cv2.circle(img, center=tuple(ground_truth[j][0:2]), radius=JOINT_RADIUS-1, 
                  color=GROUND_TRUTH_COLOR, thickness=JOINT_THICKNESS-1)
    
    # 绘制骨架连线
    skeleton_connections = ((13, 12), (12, 8), (12, 9), (8, 7), (7, 6), (9, 10), (10, 11), 
                          (2, 3), (2, 1), (1, 0), (3, 4), (4, 5))
    
    for connection in skeleton_connections:
        cv2.line(img, tuple(predicted[connection[0]][0:2]), tuple(predicted[connection[1]][0:2]), 
                color=SKELETON_COLOR, thickness=LINE_THICKNESS)
    
    # 绘制头部到髋部中点的连线
    hip_mid = (predicted[2][0:2] // 2 + predicted[3][0:2] // 2)
    cv2.line(img, tuple(predicted[12][0:2]), tuple(hip_mid), color=SKELETON_COLOR, thickness=LINE_THICKNESS)
    
    # 添加文本信息
    text_info = [
        f"Sample: {sample_idx}",
        f"PCK: {metrics['pck_score']:.3f}",
        f"Mean Error: {metrics['mean_error']:.1f}px",
        f"Endpoint PCK: {metrics['endpoint_pck']:.3f}",
        f"Quality: {metrics['quality_score']:.3f}"
    ]
    
    y_offset = 30
    for text in text_info:
        cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   TEXT_SIZE, TEXT_COLOR, TEXT_THICKNESS)
        y_offset += 25
    
    # 保存图像
    if IMAGE_FORMAT.lower() == 'jpg':
        cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
    else:
        cv2.imwrite(save_path, img)
    
    return img

def main():
    """主函数"""
    print("=== BlazePose 最佳样本测试 ===")
    
    # 打印配置信息
    print_config()
    
    # 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    model = BlazePose()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    
    # 构建模型
    model.build(input_shape=(None, 256, 256, 3))
    
    # 加载权重
    checkpoint_path = f"checkpoints/cp-{epoch_to_test:04d}.weights.h5"
    model.load_weights(checkpoint_path)
    print(f"已加载权重: {checkpoint_path}")
    
    if train_mode:
        print("开始推理...")
        # 批量推理
        predictions = np.zeros((2000, 14, 3)).astype(np.uint8)
        
        for i in range(0, 2000, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, 2000)
            predictions[i:end_idx] = model(data[i:end_idx]).numpy()
            if ENABLE_PROGRESS_BAR and (i + BATCH_SIZE) % PROGRESS_INTERVAL == 0:
                print(f"已处理 {end_idx}/2000 个样本")
        
        print("推理完成，开始选择最佳样本...")
        
        # 选择最佳样本
        best_samples = select_best_samples(predictions, label, NUM_BEST_SAMPLES, SELECTION_METHOD)
        
        print(f"\n=== 选择的最佳样本 (方法: {SELECTION_METHOD}) ===")
        for i, (sample_idx, metrics) in enumerate(best_samples):
            print(f"样本 {i+1}: 索引={sample_idx}, PCK={metrics['pck_score']:.3f}, "
                  f"平均误差={metrics['mean_error']:.1f}px, 质量分数={metrics['quality_score']:.3f}")
        
        # 可视化最佳样本
        print("\n开始可视化最佳样本...")
        all_metrics = []
        
        for i, (sample_idx, metrics) in enumerate(best_samples):
            save_path = f"{RESULT_DIR}/best_sample_{i+1}_idx_{sample_idx}_pck_{metrics['pck_score']:.3f}.{IMAGE_FORMAT}"
            
            img = visualize_sample(sample_idx, predictions[sample_idx], label[sample_idx], 
                                 metrics, save_path)
            
            print(f"已保存: {save_path}")
            all_metrics.append({
                'sample_index': sample_idx,
                'rank': i + 1,
                'metrics': metrics,
                'save_path': save_path
            })
            
            if SHOW_IMAGES:
                cv2.imshow(f"最佳样本 {i+1} (PCK: {metrics['pck_score']:.3f})", img)
                cv2.waitKey(DISPLAY_TIME)
        
        # 保存指标到JSON文件
        if SAVE_METRICS_JSON:
            metrics_file = f"{RESULT_DIR}/best_samples_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            print(f"指标已保存到: {metrics_file}")
        
        if SHOW_IMAGES:
            cv2.destroyAllWindows()
        
        print("\n=== 测试完成 ===")
        print(f"已选择并可视化 {NUM_BEST_SAMPLES} 个最佳样本")
        print(f"结果保存在: {RESULT_DIR}")
        
    else:
        print("热力图模式暂不支持最佳样本选择")
        print("请设置 train_mode=1 使用关节坐标模式")

if __name__ == "__main__":
    main()
