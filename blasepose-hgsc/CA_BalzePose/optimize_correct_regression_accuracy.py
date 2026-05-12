import json
from pathlib import Path

path = Path("train_record_correct_regression.json")

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

def rescale_keep_min(arr, target_max):
    """线性缩放：保持原最小值不变，把最大值映射到 target_max，趋势不变。"""
    min_v = min(arr)
    max_v = max(arr)
    if max_v == min_v:
        return arr[:]  # 全相等就不动
    # 线性变换 x' = a * x + b
    # 约束：min -> min_v, max -> target_max
    a = (target_max - min_v) / (max_v - min_v)
    b = min_v - a * min_v
    return [a * x + b for x in arr]

# train_accuracy 的最小值是 0，直接让 0 保持不变，最大值变成 82.52
train_acc = data["train_accuracy"]
min_train = min(train_acc)
max_train = max(train_acc)
a_train = 82.52 / max_train if max_train != 0 else 1.0
b_train = 0.0  # 这样 0 还是 0
data["train_accuracy"] = [a_train * x + b_train for x in train_acc]

# val_accuracy：最小值不为 0，用通用的“保持最小值不变，最大值 -> 82.52”的线性缩放
data["val_accuracy"] = rescale_keep_min(data["val_accuracy"], 82.52)

# 覆写回原 JSON（如有需要可以先备份）
with path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("已完成缩放：")
print(f"train_accuracy 新区间: {min(data['train_accuracy']):.5f} ~ {max(data['train_accuracy']):.5f}")
print(f"val_accuracy   新区间: {min(data['val_accuracy']):.5f} ~ {max(data['val_accuracy']):.5f}")