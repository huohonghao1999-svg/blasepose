#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级回归训练脚本 - 解决准确率低的问题
包含学习率调度、改进的损失函数、数据增强等
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tqdm import tqdm
from config import *

# 设置中文字体
def setup_chinese_font():
    try:
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.size'] = 12
    except:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.size'] = 12

setup_chinese_font()

# 导入配置
from config import *

# 高级早停配置
EARLY_STOPPING_CONFIG = {
    'min_epochs': 30,       # 最少训练轮数
    'check_epochs': 10,     # 检查最近多少个epoch
    'patience': 10,         # 连续多少个epoch没有改善就停止
    'min_delta': 0.001,    # 改善阈值
}

# 创建检查点目录
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print("高级回归训练 - 解决准确率低的问题")
print("=" * 60)

# 数据加载
from data import finetune_train as train_dataset
from data import finetune_validation as test_dataset
print("使用回归训练数据集")

# 检查数据格式
print("\n检查数据格式...")
for images, labels in train_dataset.take(1):
    print(f"图像形状: {images.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"标签值范围: {tf.reduce_min(labels):.4f} - {tf.reduce_max(labels):.4f}")
    print(f"标签示例: {labels[0, :3, :]}")  # 显示前3个关节点的坐标
    break

# 创建模型
model = BlazePose()

# 用虚拟输入构建模型
print("\n构建模型...")
dummy_input = tf.random.normal([1, 256, 256, 3])
try:
    dummy_output = model(dummy_input)
    print(f"模型构建成功，总参数: {model.count_params()}")
    print(f"虚拟输出形状: {dummy_output.shape}")
except Exception as e:
    print(f"模型构建失败: {e}")
    exit(1)

# 关键修复：只训练回归分支的参数
print("\n设置可训练参数...")
regression_layers = [
    'conv12a', 'conv12b', 'conv13a', 'conv13b', 
    'conv14a', 'conv14b', 'conv15', 'conv16', 'regression_ca'
]

# 冻结热力图分支
for layer_name in ['conv1', 'conv2_1', 'conv2_2', 'conv3', 'conv4', 'conv5', 'conv6',
                   'conv7a', 'conv7b', 'conv8a', 'conv8b', 'conv9a', 'conv9b', 
                   'conv10a', 'conv10b', 'conv11', 'endpoint_ca']:
    if hasattr(model, layer_name):
        layer = getattr(model, layer_name)
        layer.trainable = False
        print(f"冻结层: {layer_name}")

# 只训练回归分支
for layer_name in regression_layers:
    if hasattr(model, layer_name):
        layer = getattr(model, layer_name)
        layer.trainable = True
        print(f"训练层: {layer_name}")

# 检查可训练变量
trainable_vars = [var for var in model.trainable_variables if var.trainable]
print(f"\n可训练变量数量: {len(trainable_vars)}")

# 高级数据预处理函数
def advanced_preprocess_labels(labels):
    """高级标签预处理 - 包含数据增强"""
    labels_processed = tf.cast(labels, tf.float32)
    
    # 数据增强策略
    if tf.random.uniform([]) < 0.3:  # 30%概率进行数据增强
        # 1. 随机噪声
        noise = tf.random.normal(tf.shape(labels_processed), 0.0, 2.0)
        labels_processed = labels_processed + noise
        
        # 2. 随机缩放 (0.95-1.05)
        scale = tf.random.uniform([], 0.95, 1.05)
        labels_processed = labels_processed * scale
        
        # 3. 随机平移 (-5到5像素)
        translation = tf.random.uniform([2], -5.0, 5.0)
        labels_processed = labels_processed + tf.expand_dims(translation, 0)
    
    # 确保坐标在合理范围内
    labels_processed = tf.clip_by_value(labels_processed, 0.0, 255.0)
    
    return labels_processed

# 高级损失函数 - 结合多种损失
def advanced_loss(y_true, y_pred):
    """高级损失函数 - 结合MSE、MAE和Huber损失"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 计算每个关节点的欧几里得距离
    diff = y_true - y_pred
    distances = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))  # (batch_size, 14)
    
    # 1. Huber损失 (对异常值鲁棒)
    delta = 10.0
    abs_diff = tf.abs(distances)
    huber_loss = tf.where(
        abs_diff < delta,
        0.5 * tf.square(distances),
        delta * abs_diff - 0.5 * tf.square(delta)
    )
    
    # 2. 加权损失 - 给不同关节点不同权重
    # 重要关节点 (头部、肩膀、臀部) 权重更高
    joint_weights = tf.constant([
        2.0,  # 头部
        1.5,  # 左肩
        1.5,  # 右肩
        1.0,  # 左肘
        1.0,  # 右肘
        1.0,  # 左腕
        1.0,  # 右腕
        1.5,  # 左臀
        1.5,  # 右臀
        1.0,  # 左膝
        1.0,  # 右膝
        1.0,  # 左踝
        1.0,  # 右踝
        1.0   # 其他
    ])
    
    weighted_loss = huber_loss * joint_weights
    return tf.reduce_mean(weighted_loss)

# 改进的准确率计算 - 使用多级阈值
def advanced_accuracy(y_true, y_pred):
    """高级准确率计算 - 使用多级PCK阈值"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 计算每个关节点的欧几里得距离
    diff = y_true - y_pred
    distances = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))  # (batch_size, 14)
    
    # 多级PCK阈值
    thresholds = [10.0, 20.0, 30.0]  # 像素
    accuracies = []
    
    for threshold in thresholds:
        pck_accuracy = tf.cast(distances < threshold, tf.float32)
        accuracies.append(tf.reduce_mean(pck_accuracy))
    
    # 返回平均准确率 (使用中等阈值)
    return accuracies[1]  # 使用20像素阈值

# 学习率调度器
def create_lr_schedule():
    """创建学习率调度器"""
    def lr_schedule(epoch):
        if epoch < 10:
            return 0.0001
        elif epoch < 20:
            return 0.00005
        elif epoch < 30:
            return 0.00001
        else:
            return 0.000005
    
    return lr_schedule

# 编译模型 - 使用学习率调度器
initial_lr = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr, clipnorm=1.0),
    loss=advanced_loss,
    metrics=[advanced_accuracy]
)

print(f"\n开始训练...")
print(f"总轮数: {total_epoch}")
print(f"批处理大小: {batch_size}")
print(f"初始学习率: {initial_lr}")

# 训练记录
record = {
    "train_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "learning_rate": []
}

# 早停机制
best_val_acc = 0.0
patience_counter = 0
best_epoch = 0

# 训练循环
for epoch in range(total_epoch):
    print(f"\nEpoch {epoch + 1}/{total_epoch}")
    print("-" * 50)
    
    # 学习率调度
    current_lr = create_lr_schedule()(epoch)
    model.optimizer.learning_rate = current_lr
    print(f"当前学习率: {current_lr}")
    
    # 训练阶段
    train_loss = 0.0
    train_acc = 0.0
    train_batches = 0
    
    train_pbar = tqdm(train_dataset, desc="训练")
    
    for batch_idx, (images, labels) in enumerate(train_pbar):
        try:
            # 高级数据预处理
            labels_processed = advanced_preprocess_labels(labels)
            
            # 确保数据类型一致
            images = tf.cast(images, tf.float32)
            labels_processed = tf.cast(labels_processed, tf.float32)
            
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = advanced_loss(labels_processed, predictions)
            
            # 检查损失是否有效
            if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                print(f"警告: 损失值无效 ({loss.numpy()})，跳过此批次")
                continue
            
            # 计算梯度
            gradients = tape.gradient(loss, trainable_vars)
            
            # 检查梯度
            if gradients is None:
                print("警告: 所有梯度为None，跳过此批次")
                continue
            
            # 检查每个梯度
            valid_gradients = []
            invalid_count = 0
            
            for i, grad in enumerate(gradients):
                if grad is None:
                    invalid_count += 1
                    zero_grad = tf.zeros_like(trainable_vars[i])
                    valid_gradients.append(zero_grad)
                elif tf.reduce_any(tf.math.is_nan(grad)) or tf.reduce_any(tf.math.is_inf(grad)):
                    invalid_count += 1
                    zero_grad = tf.zeros_like(trainable_vars[i])
                    valid_gradients.append(zero_grad)
                else:
                    # 应用梯度裁剪
                    grad_clipped = tf.clip_by_norm(grad, clip_norm=1.0)
                    valid_gradients.append(grad_clipped)
            
            if invalid_count > 0:
                print(f"发现 {invalid_count} 个无效梯度，使用零梯度替代")
            
            # 应用梯度
            model.optimizer.apply_gradients(zip(valid_gradients, trainable_vars))
            
        except Exception as e:
            print(f"训练批次 {batch_idx} 出错: {e}")
            continue
        
        # 计算准确率
        accuracy = advanced_accuracy(labels_processed, predictions)
        
        # 调试信息：打印前几个批次的准确率
        if batch_idx < 3:
            print(f"批次 {batch_idx}: 损失 = {loss.numpy():.6f}, 准确率 = {accuracy.numpy():.6f}")
        
        train_loss += loss.numpy()
        train_acc += accuracy.numpy()
        train_batches += 1
        
        # 更新进度条
        train_pbar.set_postfix({
            'Loss': f'{loss.numpy():.6f}',
            'Acc': f'{accuracy.numpy():.6f}',
            'LR': f'{current_lr:.6f}'
        })
    
    # 计算平均训练指标
    avg_train_loss = train_loss / max(train_batches, 1)
    avg_train_acc = train_acc / max(train_batches, 1)
    
    # 验证阶段
    val_loss = 0.0
    val_acc = 0.0
    val_batches = 0
    
    val_pbar = tqdm(test_dataset, desc="验证")
    
    for images, labels in val_pbar:
        try:
            labels_processed = advanced_preprocess_labels(labels)
            images = tf.cast(images, tf.float32)
            labels_processed = tf.cast(labels_processed, tf.float32)
            
            predictions = model(images, training=False)
            loss = advanced_loss(labels_processed, predictions)
            accuracy = advanced_accuracy(labels_processed, predictions)
            
            val_loss += loss.numpy()
            val_acc += accuracy.numpy()
            val_batches += 1
            
            val_pbar.set_postfix({
                'Val Loss': f'{loss.numpy():.6f}',
                'Val Acc': f'{accuracy.numpy():.6f}'
            })
        except Exception as e:
            print(f"验证批次出错: {e}")
            continue
    
    # 计算平均验证指标
    avg_val_loss = val_loss / max(val_batches, 1)
    avg_val_acc = val_acc / max(val_batches, 1)
    
    # 记录指标
    record["train_loss"].append(float(avg_train_loss))
    record["train_accuracy"].append(float(avg_train_acc))
    record["val_accuracy"].append(float(avg_val_acc))
    record["learning_rate"].append(float(current_lr))
    
    # 打印epoch结果
    print(f"训练损失: {avg_train_loss:.6f}, 训练准确率: {avg_train_acc:.6f}")
    print(f"验证损失: {avg_val_loss:.6f}, 验证准确率: {avg_val_acc:.6f}")
    
    # 早停检查
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        best_epoch = epoch
        patience_counter = 0
        
        # 保存最佳模型
        model.save_weights(f"{checkpoint_dir}/best_model_advanced.weights.h5")
        print(f"保存最佳模型 (验证准确率: {best_val_acc:.6f})")
    else:
        patience_counter += 1
        print(f"验证准确率未改善 ({patience_counter}/{EARLY_STOPPING_CONFIG['patience']})")
    
    # 保存记录
    with open("train_record_advanced.json", "w") as f:
        json.dump(record, f, indent=2)
    
    # 实时保存训练曲线（每5个epoch保存一次）
    if (epoch + 1) % 5 == 0:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(record["train_loss"], label='训练损失', color='blue')
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(record["train_accuracy"], label='训练准确率', color='green')
        plt.plot(record["val_accuracy"], label='验证准确率', color='red')
        plt.title('准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(record["learning_rate"], label='学习率', color='orange')
        plt.title('学习率调度')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("plots/training_curves_advanced.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # 早停检查
    if patience_counter >= EARLY_STOPPING_CONFIG['patience'] and epoch >= EARLY_STOPPING_CONFIG['min_epochs']:
        print(f"\n早停触发！最佳验证准确率: {best_val_acc:.6f} (Epoch {best_epoch + 1})")
        break

print(f"\n训练完成！最佳验证准确率: {best_val_acc:.6f}")
print(f"最佳模型保存在: {checkpoint_dir}/best_model_advanced.weights.h5")
