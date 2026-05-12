import json
import numpy as np
import random

# 设置随机种子以确保可重现性
np.random.seed(42)
random.seed(42)

def generate_platform_loss(epochs=50):
    """生成有明显平台期的损失数据"""
    loss_data = []
    
    # 定义关键点 - 确保有明显的平台期
    key_points = [
        (0, 0.25),      # 起始点
        (15, 0.12),     # 第15轮 - 快速下降阶段结束
        (25, 0.08),     # 第25轮 - 开始进入平台期
        (35, 0.07),     # 第35轮 - 平台期中间
        (45, 0.065),    # 第45轮 - 平台期结束
        (49, 0.06)      # 第49轮 - 最终值
    ]
    
    # 使用分段线性插值生成平滑曲线
    for i in range(epochs):
        # 找到当前epoch在关键点之间的位置
        for j in range(len(key_points) - 1):
            if key_points[j][0] <= i <= key_points[j+1][0]:
                # 线性插值
                x1, y1 = key_points[j]
                x2, y2 = key_points[j+1]
                
                # 计算插值比例
                if x2 == x1:
                    ratio = 0
                else:
                    ratio = (i - x1) / (x2 - x1)
                
                # 线性插值
                base_loss = y1 + ratio * (y2 - y1)
                break
        else:
            # 如果超出范围，使用最后一个值
            base_loss = key_points[-1][1]
        
        # 添加很小的噪声
        noise = np.random.normal(0, 0.0005)
        loss = max(base_loss + noise, 0.055)
        loss_data.append(loss)
    
    return loss_data

def generate_s_curve_accuracy(epochs=50):
    """生成S型曲线的准确率数据"""
    train_acc = []
    val_acc = []
    
    for i in range(epochs):
        # 使用S型函数（sigmoid）生成先快后慢的上升曲线
        progress = i / epochs
        
        # S型函数：快速上升阶段（前30轮），然后平稳
        s_curve = 1 / (1 + np.exp(-8 * (progress - 0.3)))
        
        # 基础准确率：从0.18上升到0.78
        base_train = 0.18 + s_curve * 0.60
        base_val = 0.17 + s_curve * 0.57
        
        # 添加噪声
        train_noise = np.random.normal(0, 0.003)
        val_noise = np.random.normal(0, 0.004)
        
        # 确保训练准确率略高于验证准确率
        train_acc_val = base_train + train_noise
        val_acc_val = base_val + val_noise
        
        if train_acc_val < val_acc_val:
            train_acc_val = val_acc_val + 0.008
        
        # 限制范围
        train_acc_val = max(0.15, min(0.82, train_acc_val))
        val_acc_val = max(0.15, min(0.78, val_acc_val))
        
        train_acc.append(train_acc_val)
        val_acc.append(val_acc_val)
    
    return train_acc, val_acc

# 生成数据
print("正在生成有明显平台期的损失曲线数据...")
train_loss = generate_platform_loss()
train_accuracy, val_accuracy = generate_s_curve_accuracy()

# 创建数据字典
training_data = {
    "train_loss": train_loss,
    "train_accuracy": train_accuracy,
    "val_accuracy": val_accuracy
}

# 保存到JSON文件
with open('train_record_platform.json', 'w') as f:
    json.dump(training_data, f, indent=2)

print("数据已保存到 'train_record_platform.json'")
print(f"损失数据点数: {len(train_loss)}")
print(f"训练准确率数据点数: {len(train_accuracy)}")
print(f"验证准确率数据点数: {len(val_accuracy)}")

# 显示数据统计
print("\n=== 数据统计 ===")
print(f"损失范围: {min(train_loss):.6f} - {max(train_loss):.6f}")
print(f"训练准确率范围: {min(train_accuracy):.6f} - {max(train_accuracy):.6f}")
print(f"验证准确率范围: {min(val_accuracy):.6f} - {max(val_accuracy):.6f}")

# 分析平台期
print(f"\n=== 平台期分析 ===")
print(f"第25轮损失: {train_loss[25]:.6f}")
print(f"第35轮损失: {train_loss[35]:.6f}")
print(f"第45轮损失: {train_loss[45]:.6f}")
print(f"25-35轮损失变化: {train_loss[35] - train_loss[25]:.6f}")
print(f"35-45轮损失变化: {train_loss[45] - train_loss[35]:.6f}")

# 检查平台期是否明显
platform_range = max(train_loss[25:46]) - min(train_loss[25:46])
print(f"平台期(25-45轮)损失范围: {platform_range:.6f}")
print(f"平台期是否明显: {'是' if platform_range < 0.01 else '否'}")

# 检查不同阶段
print(f"\n=== 不同阶段分析 ===")
print(f"快速下降期(0-15轮): {train_loss[0]:.6f} -> {train_loss[15]:.6f}")
print(f"平台期(25-45轮): {train_loss[25]:.6f} -> {train_loss[45]:.6f}")
print(f"最终期(45-49轮): {train_loss[45]:.6f} -> {train_loss[-1]:.6f}")

# 检查平台期的稳定性
print(f"\n=== 平台期稳定性检查 ===")
platform_losses = train_loss[25:46]
platform_std = np.std(platform_losses)
print(f"平台期损失标准差: {platform_std:.6f}")
print(f"平台期是否稳定: {'是' if platform_std < 0.005 else '否'}")

# 检查关键点之间的插值
print(f"\n=== 关键点插值检查 ===")
key_points = [(0, 0.25), (15, 0.12), (25, 0.08), (35, 0.07), (45, 0.065), (49, 0.06)]
for i in range(len(key_points) - 1):
    x1, y1 = key_points[i]
    x2, y2 = key_points[i+1]
    print(f"第{x1}-{x2}轮: {y1:.3f} -> {y2:.3f}, 变化: {y2-y1:.3f}")

# 特别检查平台期的变化
print(f"\n=== 平台期详细变化 ===")
for i in range(25, 46):
    if i < len(train_loss) - 1:
        diff = train_loss[i+1] - train_loss[i]
        print(f"第{i+1}轮变化: {diff:.6f}")

# 检查是否有突然的直线下降
print(f"\n=== 突然下降检查 ===")
loss_diffs = [train_loss[i+1] - train_loss[i] for i in range(len(train_loss)-1)]
max_drop = min(loss_diffs)  # 最大下降（负值）
max_rise = max(loss_diffs)  # 最大上升（正值）

print(f"最大单轮下降: {max_drop:.6f}")
print(f"最大单轮上升: {max_rise:.6f}")
print(f"是否有突然直线下降: {'是' if abs(max_drop) > 0.005 else '否'}")

# 计算平滑度指标
print(f"\n=== 平滑度指标 ===")
abs_diffs = [abs(train_loss[i+1] - train_loss[i]) for i in range(len(train_loss)-1)]
mean_abs_diff = np.mean(abs_diffs)
max_abs_diff = max(abs_diffs)
smoothness_ratio = max_abs_diff / mean_abs_diff

print(f"平均绝对变化: {mean_abs_diff:.6f}")
print(f"最大绝对变化: {max_abs_diff:.6f}")
print(f"平滑度比率: {smoothness_ratio:.2f}")
print(f"曲线平滑度: {'非常平滑' if smoothness_ratio < 2 else '平滑' if smoothness_ratio < 3 else '一般' if smoothness_ratio < 5 else '不平滑'}")

# 检查平台期的特征
print(f"\n=== 平台期特征检查 ===")
# 检查平台期是否真的平坦
platform_flatness = max(platform_losses) - min(platform_losses)
print(f"平台期平坦度: {platform_flatness:.6f}")
print(f"平台期是否平坦: {'是' if platform_flatness < 0.008 else '否'}")

# 检查平台期前后的对比
before_platform = train_loss[15:25]  # 平台期前
after_platform = train_loss[45:50]   # 平台期后

before_std = np.std(before_platform)
after_std = np.std(after_platform)
platform_std = np.std(platform_losses)

print(f"平台期前标准差: {before_std:.6f}")
print(f"平台期标准差: {platform_std:.6f}")
print(f"平台期后标准差: {after_std:.6f}")
print(f"平台期是否比前后更稳定: {'是' if platform_std < min(before_std, after_std) else '否'}")

