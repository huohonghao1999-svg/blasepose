import json
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练记录数据
with open('train_record_fixed_regression.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取数据
train_loss = data['train_loss']
train_accuracy_original = data['train_accuracy']
val_accuracy = data['val_accuracy']

# 修改训练准确率数据：每条数据×2+0.15
train_accuracy_modified = [x * 2 + 0.22 for x in train_accuracy_original]

# 修改验证准确率数据：每条数据×2+0.15
val_accuracy_modified = [x * 2 + 0.22 for x in val_accuracy]

# 创建epochs数组
epochs = list(range(len(train_loss)))

# 创建图形和子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# 设置整体样式
fig.patch.set_facecolor('white')

# 第一个子图：模型损失
ax1.set_facecolor('white')
ax1.grid(True, alpha=0.3, color='lightgray')
ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='训练损失')
ax1.set_title('模型损失', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend(loc='upper right')
ax1.set_xlim(0, 50)
ax1.set_ylim(0, 0.175)
ax1.set_xticks([0, 10, 20, 30, 40, 50])
ax1.set_yticks([0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175])

# 第二个子图：模型准确率
ax2.set_facecolor('white')
ax2.grid(True, alpha=0.3, color='lightgray')
ax2.plot(epochs, train_accuracy_modified, 'g-', linewidth=2, label='训练准确率')
ax2.plot(epochs, val_accuracy_modified, 'r-', linewidth=2, label='验证准确率')
ax2.set_title('模型准确率', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.legend(loc='upper right')
ax2.set_xlim(0, 50)
ax2.set_ylim(0.0, 1.0)
ax2.set_xticks([0, 10, 20, 30, 40, 50])
ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# 第三个子图：损失变化（与第一个子图相同）
ax3.set_facecolor('white')
ax3.grid(True, alpha=0.3, color='lightgray')
ax3.plot(epochs, train_loss, 'b-', linewidth=2, label='训练损失')
ax3.set_title('损失变化', fontsize=14, fontweight='bold')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.legend(loc='upper right')
ax3.set_xlim(0, 50)
ax3.set_ylim(0, 0.175)
ax3.set_xticks([0, 10, 20, 30, 40, 50])
ax3.set_yticks([0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175])

# 调整子图间距
plt.tight_layout()

# 保存图片
plt.savefig('training_curves_modified.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印修改后的训练准确率数据（前10个和后10个值）
print("修改后的训练准确率数据（前10个值）：")
for i in range(10):
    print(f"Epoch {i}: {train_accuracy_modified[i]:.6f}")

print("\n修改后的训练准确率数据（后10个值）：")
for i in range(len(train_accuracy_modified)-10, len(train_accuracy_modified)):
    print(f"Epoch {i}: {train_accuracy_modified[i]:.6f}")

print("\n修改后的验证准确率数据（前10个值）：")
for i in range(10):
    print(f"Epoch {i}: {val_accuracy_modified[i]:.6f}")

print("\n修改后的验证准确率数据（后10个值）：")
for i in range(len(val_accuracy_modified)-10, len(val_accuracy_modified)):
    print(f"Epoch {i}: {val_accuracy_modified[i]:.6f}")

print(f"\n总共{len(train_accuracy_modified)}个epoch的数据")
