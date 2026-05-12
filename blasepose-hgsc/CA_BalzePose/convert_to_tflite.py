#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将训练好的 BlazePose 模型转换为 TensorFlow Lite 格式
用于部署到 BlazePose (MediaPipe Pose) 框架

使用方法:
    python convert_to_tflite.py

转换完成后，将 output_blazepose.tflite 替换 BlazePose 框架中的模型文件。
在 MediaPipe Pose 中，通常是: pose_landmark_lite.tflite / pose_landmark_full.tflite 等。
"""

import os
import tensorflow as tf
from model import BlazePose
from config import train_mode

# ========== 配置区 ==========
# 训练好的权重文件路径
WEIGHTS_PATH = "checkpoints_regression/best_model.weights.h5"

# 转换后输出的 tflite 文件路径
OUTPUT_TFLITE = "output_blazepose.tflite"

# 输入图片尺寸
INPUT_SIZE = (256, 256)
# ============================


def convert_to_tflite(weights_path, output_path):
    """将 BlazePose .weights.h5 转换为 .tflite"""

    print("=" * 60)
    print("BlazePose 模型转换工具")
    print("=" * 60)

    # 1. 构建模型
    print("\n[1/4] 构建模型...")
    model = BlazePose()

    # 用虚拟输入构建模型（必需，用于初始化所有层）
    dummy_input = tf.random.normal([1, *INPUT_SIZE, 3])
    _ = model(dummy_input)
    print(f"    模型构建完成，总参数: {model.count_params():,}")

    # 2. 加载权重
    print(f"\n[2/4] 加载权重: {weights_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    model.load_weights(weights_path)
    print("    权重加载成功")

    # 3. 准备代表模型（只保留推理需要的输出）
    print("\n[3/4] 准备 TensorFlow Lite 模型...")

    # 创建带输入输出签名的完整模型
    # BlazePose.train_mode=1 时 call() 返回 joints (14, 3)
    # 但 TFLite 推理通常期望一个输出，这里取回归分支的输出
    class BlazePoseWrapper(tf.keras.Model):
        """包装 BlazePose，只输出回归坐标"""
        def __init__(self, original_model):
            super().__init__()
            self.inner = original_model

        def call(self, x):
            # train_mode=1 -> 输出 regression joints (1, 14, 3)
            return self.inner(x)

    wrapper = BlazePoseWrapper(model)

    # 获取输入和输出张量名称
    input_tensor = tf.keras.Input(shape=(*INPUT_SIZE, 3), name="input")
    output_tensor = wrapper(input_tensor)
    print(f"    输入张量: {input_tensor.name}  shape={input_tensor.shape}")
    print(f"    输出张量: {output_tensor.name}  shape={output_tensor.shape}")

    # 4. 转换为 TFLite
    print(f"\n[4/4] 转换为 TFLite -> {output_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(wrapper)

    # 优化选项（可选，减小模型体积）
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 设置允许自定义 ops（如果模型中有自定义层）
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    tflite_model = converter.convert()

    # 保存
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    转换完成！文件大小: {size_mb:.2f} MB")
    print(f"    保存路径: {os.path.abspath(output_path)}")

    # 5. 验证转换后的模型
    print("\n[验证] 测试转换后的模型...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"    输入: {input_details[0]['shape']}")
    print(f"    输出: {output_details[0]['shape']}")

    # 用随机数据测试
    test_input = tf.random.normal([1, *INPUT_SIZE, 3]).numpy().astype('float32')
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    result = interpreter.get_tensor(output_details[0]['index'])
    print(f"    测试输出形状: {result.shape}")
    print(f"    测试输出范围: [{result.min():.4f}, {result.max():.4f}]")

    print("\n" + "=" * 60)
    print("转换成功！下一步:")
    print(f"  1. 将 '{output_path}' 复制到 BlazePose 框架目录")
    print(f"  2. 替换对应的 .tflite 模型文件")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    convert_to_tflite(WEIGHTS_PATH, OUTPUT_TFLITE)
