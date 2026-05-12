#!~/miniconda3/envs/tf2/bin/python
import os
import json
import numpy as np
import tensorflow as tf
from config import num_joints, batch_size, gaussian_sigma, gpu_dynamic_memory

# guassian generation
def getGaussianMap(joint = (16, 16), heat_size = 128, sigma = 2):
    # by default, the function returns a gaussian map with range [0, 1] of typr float32
    heatmap = np.zeros((heat_size, heat_size),dtype=np.float32)
    tmp_size = sigma * 3
    ul = [int(joint[0] - tmp_size), int(joint[1] - tmp_size)]
    br = [int(joint[0] + tmp_size + 1), int(joint[1] + tmp_size + 1)]
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))
    g.shape
    # usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], heat_size) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heat_size) - ul[1]
    # image range
    img_x = max(0, ul[0]), min(br[0], heat_size)
    img_y = max(0, ul[1]), min(br[1], heat_size)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    """
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)
    cv2.imshow("debug", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return heatmap

if gpu_dynamic_memory:
    # Limit GPU memory usage if necessary
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

# COCO 数据集路径配置
COCO_ROOT = "./datasets3"
COCO_SPLIT = os.environ.get("COCO_SPLIT", "val2017")  # 可切换为 "train2017"
COCO_MAX_SAMPLES = int(os.environ.get("COCO_MAX_SAMPLES", "600"))
COCO_MIN_KEYPOINTS = int(os.environ.get("COCO_MIN_KEYPOINTS", "5"))

annotations_file = os.path.join(
    COCO_ROOT,
    "annotations_trainval2017",
    "annotations",
    f"person_keypoints_{COCO_SPLIT}.json",
)
images_dir = os.path.join(COCO_ROOT, COCO_SPLIT, COCO_SPLIT)

if not os.path.exists(annotations_file):
    raise Exception(f"COCO 标注文件不存在: {annotations_file}")

if not os.path.isdir(images_dir):
    raise Exception(f"COCO 图像目录不存在: {images_dir}")

print(f"使用 COCO 数据集: split={COCO_SPLIT}")
print(f"图像目录: {images_dir}")
print(f"标注文件: {annotations_file}")

print("读取 COCO 标注...")
with open(annotations_file, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

image_map = {img["id"]: img for img in coco_data.get("images", [])}
annotations = coco_data.get("annotations", [])
print(f"标注总数: {len(annotations)}, 图像总数: {len(image_map)}")


def get_coco_keypoint(kps, idx):
    base = idx * 3
    return kps[base], kps[base + 1], kps[base + 2]


def average_keypoints(kps, indices):
    xs, ys, vs = [], [], []
    for idx in indices:
        x, y, v = get_coco_keypoint(kps, idx)
        if v > 0 and x > 0 and y > 0:
            xs.append(x)
            ys.append(y)
            vs.append(v)
    if not xs:
        return -1.0, -1.0, 0
    return float(np.mean(xs)), float(np.mean(ys)), max(vs)


COCO_TO_LSP = [
    16,  # 0: right ankle
    14,  # 1: right knee
    12,  # 2: right hip
    11,  # 3: left hip
    13,  # 4: left knee
    15,  # 5: left ankle
    10,  # 6: right wrist
    8,   # 7: right elbow
    6,   # 8: right shoulder
    5,   # 9: left shoulder
    7,   # 10: left elbow
    9,   # 11: left wrist
    "neck",  # 12: neck (由双肩平均得到)
    "head",  # 13: head top (由头部关键点平均得到)
]

records = []
for ann in annotations:
    if ann.get("category_id") != 1 or ann.get("iscrowd", 0) != 0:
        continue
    if ann.get("num_keypoints", 0) < COCO_MIN_KEYPOINTS:
        continue
    img_info = image_map.get(ann.get("image_id"))
    if not img_info:
        continue
    img_path = os.path.join(images_dir, img_info["file_name"])
    if not os.path.exists(img_path):
        continue
    records.append((ann, img_info, img_path))

if not records:
    raise Exception("未找到可用的人体姿态样本，请检查 COCO 数据集。")

total_available = len(records)
max_samples = min(total_available, COCO_MAX_SAMPLES)
records = records[:max_samples]
print(f"可用样本: {total_available}，实际加载: {max_samples}")

data = np.zeros([max_samples, 256, 256, 3], dtype=np.float32)
label = np.full([max_samples, num_joints, 3], -1.0, dtype=np.float32)
label[:, :, 2] = 0
heatmap_set = np.zeros((max_samples, 128, 128, num_joints), dtype=np.float32)

print("开始读取 COCO 图像与关键点 (可能耗时几分钟)...")
for i, (ann, img_info, img_path) in enumerate(records):
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=3)
    img_resized = tf.image.resize(img, [256, 256])
    data[i] = img_resized.numpy()

    width = img_info["width"]
    height = img_info["height"]
    keypoints = ann.get("keypoints", [])

    for lsp_idx, source in enumerate(COCO_TO_LSP):
        if source == "neck":
            kp = average_keypoints(keypoints, [5, 6])
        elif source == "head":
            kp = average_keypoints(keypoints, [0, 1, 2, 3, 4])
        else:
            kp = get_coco_keypoint(keypoints, source)
        x, y, v = kp
        if v > 0 and x > 0 and y > 0 and width > 0 and height > 0:
            label[i, lsp_idx, 0] = x * (256.0 / width)
            label[i, lsp_idx, 1] = y * (256.0 / height)
            label[i, lsp_idx, 2] = 1
        else:
            label[i, lsp_idx, 0] = -1
            label[i, lsp_idx, 1] = -1
            label[i, lsp_idx, 2] = 0

    for j in range(num_joints):
        if label[i, j, 2] > 0:
            _joint = (label[i, j, 0:2] // 2).astype(np.uint16)
            heatmap_set[i, :, :, j] = getGaussianMap(
                joint=_joint, heat_size=128, sigma=gaussian_sigma
            )

    if max_samples >= 20 and not i % max(1, max_samples // 20):
        progress = (i + 1) / max_samples * 100
        print(f"\rProgress: {progress:.1f}% ({i+1}/{max_samples})", end="", flush=True)

print("\n生成训练/测试数据集...")
train_size = int(max_samples * 0.8)
if max_samples > 1:
    train_size = min(max_samples - 1, max(1, train_size))
else:
    train_size = max_samples

train_dataset = tf.data.Dataset.from_tensor_slices(
    (data[:train_size], heatmap_set[:train_size])
)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (data[train_size:], heatmap_set[train_size:])
)

if train_size > 0:
    shuffle_buffer = min(1000, train_size)
    train_dataset = train_dataset.shuffle(shuffle_buffer).batch(batch_size)
else:
    train_dataset = train_dataset.batch(batch_size)

test_dataset = test_dataset.batch(batch_size)

finetune_train = tf.data.Dataset.from_tensor_slices((data[:train_size], label[:train_size]))
finetune_validation = tf.data.Dataset.from_tensor_slices((data[train_size:], label[train_size:]))

if train_size > 0:
    finetune_train = finetune_train.shuffle(shuffle_buffer).batch(batch_size)
else:
    finetune_train = finetune_train.batch(batch_size)

finetune_validation = finetune_validation.batch(batch_size)

print("Done.")
