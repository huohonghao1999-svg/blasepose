#!~/miniconda3/envs/tf2/bin/python
import os
import numpy as np
import tensorflow as tf
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
if not os.path.exists("./dataset"):
    os.makedirs("./dataset", exist_ok=True)

if os.path.exists("./dataset/lsp/joints.mat"):
    print("Found lsp dataset.")
else:
    raise Exception("LSP dataset not found. Please ensure the dataset is placed in './dataset/lsp/' directory with 'joints.mat' and 'images/' folder.")

# read annotations
annotations = loadmat("./dataset/lsp/joints.mat")
label = annotations["joints"].swapaxes(0, 2)    # shape (3, 14, 2000) -> (2000, 14, 3)

# read images
data = np.zeros([2000, 256, 256, 3])
heatmap_set = np.zeros((2000, 128, 128, num_joints), dtype=np.float32)
print("Reading dataset...")
print("This may take several minutes...")
for i in range(2000):
    FileName = "./dataset/lsp/images/im%04d.jpg" % (i + 1)
    img = tf.io.read_file(FileName)
    img = tf.image.decode_image(img)
    img_shape = img.shape
    # Attention here img_shape[0] is height and [1] is width
    label[i, :, 0] *= (256 / img_shape[1])
    label[i, :, 1] *= (256 / img_shape[0])
    data[i] = tf.image.resize(img, [256, 256])
    # generate heatmap set
    for j in range(num_joints):
        _joint = (label[i, j, 0:2] // 2).astype(np.uint16)
        # print(_joint)
        heatmap_set[i, :, :, j] = getGaussianMap(joint = _joint, heat_size = 128, sigma = gaussian_sigma)
    # print status with percentage
    if not i%(2000//20):  # 每100张图片显示一次进度
        progress = (i + 1) / 2000 * 100
        print(f"\rProgress: {progress:.1f}% ({i+1}/2000)", end='', flush=True)

# dataset
print("\nGenerating training and testing data batches...")
# 减少内存使用，只加载前200个样本进行训练
train_dataset = tf.data.Dataset.from_tensor_slices((data[0:200], heatmap_set[0:200]))
test_dataset = tf.data.Dataset.from_tensor_slices((data[200:300], heatmap_set[200:300]))

SHUFFLE_BUFFER_SIZE = 1000
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Finetune - 减少内存使用，只加载前100个样本进行测试
finetune_train = tf.data.Dataset.from_tensor_slices((data[0:100], label[0:100]))
finetune_validation = tf.data.Dataset.from_tensor_slices((data[100:200], label[100:200]))

finetune_train = finetune_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
finetune_validation = finetune_validation.batch(batch_size)

print("Done.")
