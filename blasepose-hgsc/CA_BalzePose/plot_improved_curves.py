import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
with open('train_record_improved.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 提取数据
train_loss = train_data['train_loss']
train_accuracy = train_data['train_accuracy']
val_accuracy = train_data['val_accuracy']

# 创建epochs数组
epochs = list(range(len(train_loss)))

# 创建图形，左右两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 左图：模型损失
ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='训练损失')
ax1.set_title('模型损失', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 右图：模型准确率
ax2.plot(epochs, train_accuracy, 'g-', linewidth=2, label='训练准确率')
ax2.plot(epochs, val_accuracy, 'r-', linewidth=2, label='验证准确率')
ax2.set_title('模型准确率', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

# 设置y轴范围
ax2.set_ylim(0.0, 1.0)

# 调整布局
plt.tight_layout()

# 确保plots目录存在
os.makedirs('plots', exist_ok=True)

# 保存图片
output_path = 'plots/training_curves_improved.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"训练曲线图已生成并保存为 '{output_path}'")

# 显示统计信息
print(f"\n数据点数统计:")
print(f"  损失数据点数: {len(train_loss)}")
print(f"  训练准确率数据点数: {len(train_accuracy)}")
print(f"  验证准确率数据点数: {len(val_accuracy)}")

# 显示数据统计
print("\n=== 数据统计 ===")
print(f"损失范围: {min(train_loss):.2f} - {max(train_loss):.2f}")
print(f"训练准确率范围: {min(train_accuracy):.6f} - {max(train_accuracy):.6f}")
print(f"验证准确率范围: {min(val_accuracy):.6f} - {max(val_accuracy):.6f}")

# 显示最终值
print("\n=== 最终训练结果 ===")
print(f"最终训练损失: {train_loss[-1]:.6f}")
print(f"最终训练准确率: {train_accuracy[-1]:.6f}")
print(f"最终验证准确率: {val_accuracy[-1]:.6f}")

# 显示最大值
print("\n=== 最佳结果 ===")
print(f"最佳训练准确率: {max(train_accuracy):.6f} (Epoch {train_accuracy.index(max(train_accuracy)) + 1})")
print(f"最佳验证准确率: {max(val_accuracy):.6f} (Epoch {val_accuracy.index(max(val_accuracy)) + 1})")

plt.show()

