import json
import numpy as np

def exponential_moving_average(data, alpha=0.3):
    """
    指数移动平均平滑
    alpha: 平滑系数，范围0-1，越小越平滑
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed.tolist()

def moving_average(data, window_size=5):
    """
    简单移动平均平滑
    window_size: 窗口大小
    """
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed

# 读取JSON文件
print("正在读取 train_record_improved.json...")
with open('train_record_improved.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取原始数据
train_accuracy = np.array(data['train_accuracy'])
val_accuracy = np.array(data['val_accuracy'])

print(f"原始数据长度: train_accuracy={len(train_accuracy)}, val_accuracy={len(val_accuracy)}")

# 对 train_accuracy 进行平滑处理（使用指数移动平均）
print("正在对 train_accuracy 进行平滑处理...")
smoothed_train_acc = exponential_moving_average(train_accuracy, alpha=0.3)

# 对 val_accuracy 进行平滑处理
print("正在对 val_accuracy 进行平滑处理...")
smoothed_val_acc = exponential_moving_average(val_accuracy, alpha=0.3)

# 更新数据
data['train_accuracy'] = smoothed_train_acc
data['val_accuracy'] = smoothed_val_acc

# 保存回文件
print("正在保存平滑后的数据...")
with open('train_record_improved.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("平滑处理完成！")
print(f"\n平滑前后对比（前10个数据点）:")
print("train_accuracy 原始值:", train_accuracy[:10])
print("train_accuracy 平滑值:", smoothed_train_acc[:10])
print("\nval_accuracy 原始值:", val_accuracy[:10])
print("val_accuracy 平滑值:", smoothed_val_acc[:10])


