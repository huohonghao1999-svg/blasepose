#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的测试脚本 - 解决坐标归一化问题
"""

import tensorflow as tf
import numpy as np
import json
import os
import cv2
from datetime import datetime
from model import BlazePose
from config import *
from analysis import getPCK

def pixel_to_normalized(coords, img_size=256):
    """将像素坐标转换为归一化坐标"""
    return coords / img_size

def normalized_to_pixel(coords, img_size=256):
    """将归一化坐标转换为像素坐标"""
    return coords * img_size

def ensure_coordinate_range(coords, min_val=-1.0, max_val=1.0):
    """确保坐标在指定范围内"""
    coords = tf.clip_by_value(coords, min_val, max_val)
    return coords

def test_with_coordinate_fix():
    """使用坐标修复进行测试"""
    
    print("开始修复后的测试...")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    model = BlazePose()
    
    # 构建模型
    dummy_input = tf.random.normal((1, 256, 256, 3))
    _ = model(dummy_input)
    print("模型构建完成")
    
    # 加载权重
    epoch_to_test = 400
    checkpoint_path = f"checkpoints_regression/cp-{epoch_to_test:04d}.weights.h5"
    # checkpoint_path = f"checkpoints_regression/final_correct_regression_model.weights.h5"

    
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print(f"成功加载权重: {checkpoint_path}")
    else:
        print(f"权重文件不存在: {checkpoint_path}")
        return
    
    # 2. 加载测试数据
    print("\n2. 加载测试数据...")
    try:
        from data import finetune_validation as test_data
        print("成功加载测试数据")
    except Exception as e:
        print(f"测试数据加载失败: {e}")
        return
    
    # 3. 进行预测并修复坐标
    print("\n3. 进行预测并修复坐标...")
    
    all_predictions = []
    all_labels = []
    
    batch_count = 0
    for batch_x, batch_y in test_data:
        # 模型预测（输出归一化坐标）
        predictions = model(batch_x)
        
        # 将归一化坐标转换为像素坐标
        predictions_pixel = normalized_to_pixel(predictions, img_size=256)
        
        # 确保坐标在合理范围内
        predictions_pixel = ensure_coordinate_range(predictions_pixel, 0, 255)
        
        all_predictions.append(predictions_pixel.numpy())
        all_labels.append(batch_y.numpy())
        
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"已处理 {batch_count} 个批次...")
    
    # 合并结果
    y = np.concatenate(all_predictions, axis=0)
    label_data = np.concatenate(all_labels, axis=0)
    
    print(f"总共处理了 {len(y)} 个样本")
    print(f"预测坐标范围: {np.min(y):.4f} - {np.max(y):.4f}")
    print(f"标签坐标范围: {np.min(label_data):.4f} - {np.max(label_data):.4f}")
    
    # 4. 计算PCK分数
    print("\n4. 计算PCK分数...")
    
    # 只取x,y坐标
    y_coords = y[:, :, 0:2].astype(float)
    label_coords = label_data[:, :, 0:2].astype(float)
    
    def Eclidian2(a, b):
        summer = 0
        for i in range(len(a)):
            summer += (a[i] - b[i]) ** 2
        return summer
    
    score_j = np.zeros(14)
    pck_metric = 0.5
    
    for i in range(len(y_coords)):
        # 计算头部到颈部的距离作为参考
        pck_h = Eclidian2(label_coords[i][12], label_coords[i][13])
        for j in range(14):
            pck_j = Eclidian2(y_coords[i][j], label_coords[i][j])
            if pck_j <= pck_h * pck_metric:
                score_j[j] += 1
    
    # 转换为百分比
    score_j = score_j / len(y_coords) * 100
    score_avg = sum(score_j) / 14
    
    print(f"PCK分数: {score_avg:.4f}")
    
    # 5. 计算MAE
    mae = np.mean(np.abs(y_coords - label_coords))
    print(f"MAE: {mae:.4f} 像素")
    
    # 6. 保存结果
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_epoch": epoch_to_test,
        "test_mae_pixels": float(mae),
        "pck_score": float(score_avg),
        "coordinate_fix_applied": True,
        "total_samples": len(y_coords)
    }
    
    results_file = f"test_results_fixed_epoch_{epoch_to_test}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试结果已保存到: {results_file}")
    print(f"修复后PCK分数: {score_avg:.2f}%")
    print(f"修复后MAE: {mae:.2f} 像素")
    
    # 7. 生成可视化结果
    print("\n5. 生成可视化结果...")
    
    # 选择最好的3个样本
    sample_scores = []
    for i in range(len(y_coords)):
        pck_h = Eclidian2(label_coords[i][12], label_coords[i][13])
        correct_joints = 0
        for j in range(14):
            pck_j = Eclidian2(y_coords[i][j], label_coords[i][j])
            if pck_j <= pck_h * pck_metric:
                correct_joints += 1
        pck_score = correct_joints / 14
        sample_scores.append((i, pck_score))
    
    # 按PCK分数排序
    sample_scores.sort(key=lambda x: x[1], reverse=True)
    best_samples = sample_scores[:3]
    
    print(f"选择效果最好的3个样本:")
    for i, (sample_idx, score) in enumerate(best_samples):
        print(f"样本 {sample_idx}: PCK分数 = {score:.3f}")
    
    # 生成结果图片
    os.makedirs("result_fixed", exist_ok=True)
    
    for i, (sample_idx, score) in enumerate(best_samples):
        # 创建空白图像用于可视化
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255  # 白色背景
        sample_pred = y_coords[sample_idx]
        sample_label = label_coords[sample_idx]
        
        # 图像已经是正确的格式（BGR）
        
        # 绘制预测关键点（红色）
        for j in range(14):
            x, y = int(sample_pred[j][0]), int(sample_pred[j][1])
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)  # 红色
        
        # 绘制标签关键点（绿色）
        for j in range(14):
            x, y = int(sample_label[j][0]), int(sample_label[j][1])
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # 绿色
        
        # 保存图片
        result_file = f"result_fixed/best_sample_{i+1}_idx_{sample_idx}_pck_{score:.3f}.jpg"
        cv2.imwrite(result_file, img)
        print(f"结果图片: {result_file}")
    
    print("\n修复完成！")
    print("=" * 60)
    print("修复总结:")
    print("1. 将模型输出的归一化坐标转换为像素坐标")
    print("2. 确保坐标在合理范围内")
    print("3. 重新计算PCK分数和MAE")
    print("4. 生成修复后的结果图片")

if __name__ == "__main__":
    test_with_coordinate_fix()
