import json
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
with open('train_record_generated.json', 'r') as f:
    train_data = json.load(f)

# 提取数据
train_loss = train_data['train_loss']
train_accuracy = train_data['train_accuracy']
val_accuracy = train_data['val_accuracy']

# 创建图形，左右两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 左图：模型损失
ax1.plot(train_loss, 'b-', linewidth=2, label='训练损失')
ax1.set_title('模型损失', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 右图：模型准确率
ax2.plot(train_accuracy, 'g-', linewidth=2, label='训练准确率')
ax2.plot(val_accuracy, 'r-', linewidth=2, label='验证准确率')
ax2.set_title('模型准确率', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

# 设置y轴范围
ax2.set_ylim(0.0, 0.9)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("训练曲线图已生成并保存为 'training_curves.png'")
print(f"损失数据点数: {len(train_loss)}")
print(f"训练准确率数据点数: {len(train_accuracy)}")
print(f"验证准确率数据点数: {len(val_accuracy)}")

# 显示数据统计
print("\n=== 数据统计 ===")
print(f"损失范围: {min(train_loss):.2f} - {max(train_loss):.2f}")
print(f"训练准确率范围: {min(train_accuracy):.6f} - {max(train_accuracy):.6f}")
print(f"验证准确率范围: {min(val_accuracy):.6f} - {max(val_accuracy):.6f}")

# 分析不同阶段的下降速度
print(f"\n=== 不同阶段下降速度分析 ===")
# 前80轮
early_loss = train_loss[:80]
early_loss_decrease = early_loss[0] - early_loss[-1]
print(f"前80轮损失下降: {early_loss_decrease:.2f} (平均每轮下降: {early_loss_decrease/80:.2f})")

# 80-150轮
mid_loss = train_loss[80:150]
mid_loss_decrease = mid_loss[0] - mid_loss[-1]
print(f"80-150轮损失下降: {mid_loss_decrease:.2f} (平均每轮下降: {mid_loss_decrease/70:.2f})")

# 150-250轮
late_loss = train_loss[150:250]
late_loss_decrease = late_loss[0] - late_loss[-1]
print(f"150-250轮损失下降: {late_loss_decrease:.2f} (平均每轮下降: {late_loss_decrease/100:.2f})")

# 最后70轮平台期
platform_loss = train_loss[-70:]
platform_loss_decrease = platform_loss[0] - platform_loss[-1]
print(f"最后70轮平台期损失下降: {platform_loss_decrease:.2f} (平均每轮下降: {platform_loss_decrease/70:.2f})")

# 分析准确率变化
print(f"\n=== 准确率变化分析 ===")
print(f"训练准确率变化: {train_accuracy[0]:.6f} -> {train_accuracy[-1]:.6f}")
print(f"验证准确率变化: {val_accuracy[0]:.6f} -> {val_accuracy[-1]:.6f}")

# 计算相邻元素差值
loss_diffs = [abs(train_loss[i+1] - train_loss[i]) for i in range(len(train_loss)-1)]
train_diffs = [abs(train_accuracy[i+1] - train_accuracy[i]) for i in range(len(train_accuracy)-1)]
val_diffs = [abs(val_accuracy[i+1] - val_accuracy[i]) for i in range(len(val_accuracy)-1)]

print(f"\n=== 相邻元素差值统计 ===")
print(f"损失最大相邻差值: {max(loss_diffs):.2f}")
print(f"训练准确率最大相邻差值: {max(train_diffs):.6f}")
print(f"验证准确率最大相邻差值: {max(val_diffs):.6f}")
print(f"损失平均相邻差值: {np.mean(loss_diffs):.2f}")
print(f"训练准确率平均相邻差值: {np.mean(train_diffs):.6f}")
print(f"验证准确率平均相邻差值: {np.mean(val_diffs):.6f}")

# 分析最后70轮的平台期
print(f"\n=== 最后70轮平台期分析 ===")
last_70_loss = train_loss[-70:]
last_70_train = train_accuracy[-70:]
last_70_val = val_accuracy[-70:]

print(f"最后70轮损失变化: {last_70_loss[0]:.2f} -> {last_70_loss[-1]:.2f}")
print(f"最后70轮训练准确率变化: {last_70_train[0]:.6f} -> {last_70_train[-1]:.6f}")
print(f"最后70轮验证准确率变化: {last_70_val[0]:.6f} -> {last_70_val[-1]:.6f}")
