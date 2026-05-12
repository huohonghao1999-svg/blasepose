#!~/miniconda3/envs/tf2/bin/python
import os
import numpy as np
import tensorflow as tf
import json
from scipy.io import loadmat
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
    if img_x[1] > img_x[0] and img_y[1] > img_y[0] and g_x[1] > g_x[0] and g_y[1] > g_y[0]:
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

# 检查数据集是否存在
images_dir = "./dataset2/raw/MPII_Human_Pose/images"
annotations_file = "./dataset2/dsdl/dsdl_KeyDet_full/dsdl_KeyDet_full/set-train/train_samples.json"

if not os.path.exists(images_dir):
    raise Exception(f"MPII images directory not found: {images_dir}")

if not os.path.exists(annotations_file):
    raise Exception(f"MPII annotations file not found: {annotations_file}")

print("Found MPII Human Pose dataset.")
print(f"Images directory: {images_dir}")
print(f"Annotations file: {annotations_file}")

# MPII 16个关键点到LSP 14个关键点的映射
# MPII顺序: 0-r ankle, 1-r knee, 2-r hip, 3-l hip, 4-l knee, 5-l ankle, 
#           6-pelvis, 7-thorax, 8-upper neck, 9-head top, 
#           10-r wrist, 11-r elbow, 12-r shoulder, 13-l shoulder, 14-l elbow, 15-l wrist
# LSP顺序: 0-right ankle, 1-right knee, 2-right hip, 3-left hip, 4-left knee, 5-left ankle,
#          6-right wrist, 7-right elbow, 8-right shoulder, 9-left shoulder, 10-left elbow, 11-left wrist,
#          12-neck, 13-head top
MPII_TO_LSP_MAPPING = [
    0,   # MPII 0 (r ankle) -> LSP 0
    1,   # MPII 1 (r knee) -> LSP 1
    2,   # MPII 2 (r hip) -> LSP 2
    3,   # MPII 3 (l hip) -> LSP 3
    4,   # MPII 4 (l knee) -> LSP 4
    5,   # MPII 5 (l ankle) -> LSP 5
    10,  # MPII 10 (r wrist) -> LSP 6
    11,  # MPII 11 (r elbow) -> LSP 7
    12,  # MPII 12 (r shoulder) -> LSP 8
    13,  # MPII 13 (l shoulder) -> LSP 9
    14,  # MPII 14 (l elbow) -> LSP 10
    15,  # MPII 15 (l wrist) -> LSP 11
    7,   # MPII 7 (thorax) -> LSP 12 (neck)
    9,   # MPII 9 (head top) -> LSP 13
]

# 读取标注文件
print("Loading annotations...")
with open(annotations_file, 'r', encoding='utf-8') as f:
    annotations_data = json.load(f)

samples = annotations_data.get('samples', [])
print(f"Total samples in JSON: {len(samples)}")

# 过滤出有有效标注的样本（只选择第一个人的标注，且关键点可见）
valid_samples = []
for sample in samples:
    if sample.get('annotations') and len(sample['annotations']) > 0:
        # 选择第一个人
        person = sample['annotations'][0]
        keypoints = person.get('keypoints', [])
        if len(keypoints) == 16:  # MPII有16个关键点
            valid_samples.append(sample)

print(f"Valid samples with annotations: {len(valid_samples)}")

# 使用所有可用样本（如果内存不足，可以设置上限，例如：max_samples = min(len(valid_samples), 2000)）
max_samples = len(valid_samples)  # 使用所有样本
valid_samples = valid_samples[:max_samples]
print(f"Using {max_samples} samples for training/testing")

# 读取图片和标注
# 先过滤掉图片不存在的样本
print("Checking image files...")
valid_samples_with_images = []
for sample in valid_samples:
    # media字段可能是 "images/xxx.jpg" 格式，需要提取文件名
    media_path = sample['media']
    # 如果包含路径分隔符，只取文件名部分
    if '/' in media_path or '\\' in media_path:
        filename = os.path.basename(media_path)
    else:
        filename = media_path
    img_path = os.path.join(images_dir, filename)
    if os.path.exists(img_path):
        valid_samples_with_images.append(sample)
    else:
        print(f"Warning: Image not found: {img_path}")

actual_samples = len(valid_samples_with_images)
print(f"Found {actual_samples} samples with valid images")

if actual_samples == 0:
    raise Exception("No valid samples found with images!")

# 更新实际使用的样本数量
max_samples = min(actual_samples, max_samples, 2000)
valid_samples_with_images = valid_samples_with_images[:max_samples]

data = np.zeros([max_samples, 256, 256, 3], dtype=np.float32)
label = np.zeros([max_samples, num_joints, 3], dtype=np.float32)  # (x, y, visibility)
heatmap_set = np.zeros((max_samples, 128, 128, num_joints), dtype=np.float32)

print("Reading dataset...")
print("This may take several minutes...")

for i, sample in enumerate(valid_samples_with_images):
    # 读取图片
    # media字段可能是 "images/xxx.jpg" 格式，需要提取文件名
    media_path = sample['media']
    # 如果包含路径分隔符，只取文件名部分
    if '/' in media_path or '\\' in media_path:
        filename = os.path.basename(media_path)
    else:
        filename = media_path
    img_path = os.path.join(images_dir, filename)
    
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img_shape = tf.shape(img)
    img_height = img_shape[0].numpy()
    img_width = img_shape[1].numpy()
    
    # 调整图片大小到256x256
    img_resized = tf.image.resize(img, [256, 256])
    data[i] = img_resized.numpy()
    
    # 读取关键点标注
    person = sample['annotations'][0]
    mpii_keypoints = person.get('keypoints', [])
    
    # 将MPII的16个关键点映射到LSP的14个关键点
    for lsp_idx, mpii_idx in enumerate(MPII_TO_LSP_MAPPING):
        if mpii_idx < len(mpii_keypoints):
            kp = mpii_keypoints[mpii_idx]
            x, y, visibility = kp[0], kp[1], kp[2]
            
            # 如果关键点不可见，设置为-1
            if visibility == 0 or x < 0 or y < 0:
                label[i, lsp_idx, 0] = -1
                label[i, lsp_idx, 1] = -1
                label[i, lsp_idx, 2] = 0
            else:
                # 将坐标缩放到256x256
                x_scaled = x * (256.0 / img_width)
                y_scaled = y * (256.0 / img_height)
                label[i, lsp_idx, 0] = x_scaled
                label[i, lsp_idx, 1] = y_scaled
                label[i, lsp_idx, 2] = 1
    
    # 生成热图
    for j in range(num_joints):
        if label[i, j, 2] > 0:  # 如果关键点可见
            _joint = (label[i, j, 0:2] // 2).astype(np.uint16)
            heatmap_set[i, :, :, j] = getGaussianMap(joint=_joint, heat_size=128, sigma=gaussian_sigma)
    
    # 显示进度
    if (i + 1) % (max_samples // 20) == 0 or (i + 1) == max_samples:
        progress = (i + 1) / max_samples * 100
        print(f"\rProgress: {progress:.1f}% ({i+1}/{max_samples})", end='', flush=True)

# dataset
print("\nGenerating training and testing data batches...")
# 划分训练集和测试集（80%训练，20%测试）
train_ratio = 0.6
train_size = int(max_samples * train_ratio)
if train_size == 0 and max_samples > 0:
    train_size = 1
test_size = max_samples - train_size

print(f"Training samples: {train_size}, Testing samples: {test_size}")

train_dataset = tf.data.Dataset.from_tensor_slices((data[0:train_size], heatmap_set[0:train_size]))
test_dataset = tf.data.Dataset.from_tensor_slices((data[train_size:max_samples], heatmap_set[train_size:max_samples]))

SHUFFLE_BUFFER_SIZE = max(1, min(1000, train_size)) if train_size > 0 else 1
if train_size > 0:
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE)
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Finetune - 使用部分数据进行微调
finetune_train_size = min(100, train_size)
finetune_val_size = test_size  # 使用所有测试样本进行评估

finetune_train = tf.data.Dataset.from_tensor_slices((data[0:finetune_train_size], label[0:finetune_train_size]))
finetune_validation = tf.data.Dataset.from_tensor_slices((data[train_size:train_size+finetune_val_size], label[train_size:train_size+finetune_val_size]))

finetune_train = finetune_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
finetune_validation = finetune_validation.batch(batch_size)

print("Done.")
