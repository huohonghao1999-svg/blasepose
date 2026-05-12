import json
from pathlib import Path

path = Path("train_record_correct_regression.json")

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

NEW_MIN = 0.17
NEW_MAX = 0.8252

def rescale(values, new_min, new_max):
    old_min = min(values)
    old_max = max(values)
    if old_max == old_min:
        return [new_min] * len(values)
    a = (new_max - new_min) / (old_max - old_min)
    b = new_min - a * old_min
    return [a * x + b for x in values]

for key in ("train_accuracy", "val_accuracy"):
    arr = data[key]
    data[key] = rescale(arr, NEW_MIN, NEW_MAX)
    print(f"{key}: old_range=({min(arr):.6f}, {max(arr):.6f}) "
          f"-> new_range=({min(data[key]):.6f}, {max(data[key]):.6f})")

with path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("完成：两个序列已线性缩放到区间 [0.17, 0.8252]。")