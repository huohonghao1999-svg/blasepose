num_joints = 14     # lsp dataset

# CPU优化配置
batch_size = 128     # CPU优化批处理大小
total_epoch = 1000   # 训练轮数
gaussian_sigma = 4  # 高斯核标准差
gpu_dynamic_memory = False  # GPU动态内存分配

# Train mode: 0-pre-train, 1-finetune
train_mode = 1

# Evaluation mode: 0-get result images, 1-get PCK score only
eval_mode = 1

show_batch_loss = 0
continue_train = 0  # 0 for random initialize, >0 for num epoch

if train_mode:
    best_pre_train = 29 # num of epoch where the training loss drops but testing accuracy achieve the optimal

# for test only
epoch_to_test = 1
# for test the heatmap only
vis_img_id = 1797

# CA注意力机制开关
use_ca_attention = True  # True: 使用CA注意力机制, False: 使用原始版本

# 权重保存路径配置
# 根据训练模式选择不同的保存路径
if train_mode:
    # 回归训练模式 (finetune)
    checkpoint_dir = "checkpoints_regression"
    json_name = "train_record.json"
else:
    # 热图训练模式 (pre-train)
    checkpoint_dir = "checkpoints_heatmap"
    json_name = "train_record_pretrain.json"
