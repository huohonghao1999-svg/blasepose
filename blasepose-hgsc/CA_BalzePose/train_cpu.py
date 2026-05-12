#!~/miniconda3/envs/tf2/bin/python
import os
import tensorflow as tf
import time
from tqdm import tqdm
from model import BlazePose
from config import total_epoch, train_mode, continue_train, show_batch_loss, batch_size
from analysis import save_record, load_record
import matplotlib.pyplot as plt
import matplotlib
import platform
from early_stopping import EasyStopping, LearningRateScheduler

# 强制使用CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    if system == "Windows":
        # Windows系统字体
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    elif system == "Darwin":  # macOS
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
    else:  # Linux
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei']
    
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    matplotlib.rcParams['font.size'] = 12

# 调用字体设置
setup_chinese_font()

# 导入配置
from config import checkpoint_dir

# Create checkpoints directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs("plots", exist_ok=True)

if train_mode:
    from data import finetune_train as train_dataset
    from data import finetune_validation as test_dataset
    loss_func = tf.keras.losses.MeanSquaredError()
else:
    from data import train_dataset, test_dataset
    loss_func = tf.keras.losses.BinaryCrossentropy()

# 创建模型
model = BlazePose()

# 编译模型 - CPU优化
# 注意：由于train_mode=0，只有heatmap分支会被使用，regression分支不会产生梯度
# 这是正常的，因为模型设计就是根据train_mode选择使用哪个分支
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 降低学习率
    loss=loss_func,
    metrics=['accuracy']
)

# 早停和学习率调度 - 可调整参数
early_stopping = EasyStopping(patience=50, min_delta=0.001)  # 增加patience到20
lr_scheduler = LearningRateScheduler(patience=5, factor=0.5, min_lr=1e-6)

# 检查是否有之前的训练记录
if continue_train > 0:
    print(f"继续训练，从第 {continue_train} 轮开始...")
    train_loss_list, train_acc_list, val_acc_list = load_record()
    record = {
        "train_loss": list(train_loss_list),
        "train_accuracy": list(train_acc_list),
        "val_accuracy": list(val_acc_list)
    }
    start_epoch = continue_train
else:
    print("开始新的训练...")
    record = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }
    start_epoch = 0

print(f"使用CPU训练，批处理大小: {batch_size}")
print(f"总训练轮数: {total_epoch}")
print(f"开始轮数: {start_epoch}")

# 训练循环
for epoch in range(start_epoch, total_epoch):
    print(f"\n=== Epoch {epoch + 1}/{total_epoch} ===")
    
    # 训练阶段
    train_loss = 0.0
    train_acc = 0.0
    train_batches = 0
    
    # 使用tqdm显示进度
    train_pbar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}")
    
    for batch_idx, (images, heatmaps) in enumerate(train_pbar):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_func(heatmaps, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # 计算准确率
        accuracy = tf.keras.metrics.binary_accuracy(heatmaps, predictions)
        accuracy = tf.reduce_mean(accuracy)
        
        train_loss += loss.numpy()
        train_acc += accuracy.numpy()
        train_batches += 1
        
        # 更新进度条
        train_pbar.set_postfix({
            'Loss': f'{loss.numpy():.4f}',
            'Acc': f'{accuracy.numpy():.4f}'
        })
        
        if show_batch_loss and batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}: Loss = {loss.numpy():.6f}, Acc = {accuracy.numpy():.6f}")
    
    # 计算平均训练指标
    avg_train_loss = train_loss / train_batches
    avg_train_acc = train_acc / train_batches
    
    # 验证阶段
    val_loss = 0.0
    val_acc = 0.0
    val_batches = 0
    
    val_pbar = tqdm(test_dataset, desc="Validation")
    
    for images, heatmaps in val_pbar:
        predictions = model(images, training=False)
        loss = loss_func(heatmaps, predictions)
        accuracy = tf.keras.metrics.binary_accuracy(heatmaps, predictions)
        accuracy = tf.reduce_mean(accuracy)
        
        val_loss += loss.numpy()
        val_acc += accuracy.numpy()
        val_batches += 1
        
        val_pbar.set_postfix({
            'Val Loss': f'{loss.numpy():.4f}',
            'Val Acc': f'{accuracy.numpy():.4f}'
        })
    
    # 计算平均验证指标
    avg_val_loss = val_loss / val_batches
    avg_val_acc = val_acc / val_batches
    
    # 记录指标
    record["train_loss"].append(avg_train_loss)
    record["train_accuracy"].append(avg_train_acc)
    record["val_accuracy"].append(avg_val_acc)
    
    # 打印epoch结果
    print(f"训练损失: {avg_train_loss:.6f}, 训练准确率: {avg_train_acc:.6f}")
    print(f"验证损失: {avg_val_loss:.6f}, 验证准确率: {avg_val_acc:.6f}")
    
    # 保存记录
    save_record(record["train_loss"], record["train_accuracy"], record["val_accuracy"])
    
    # 实时保存训练曲线（每5个epoch保存一次）
    if (epoch + 1) % 5 == 0:
        # 绘制当前训练曲线（使用英文标签确保兼容性）
        epochs = range(1, len(record['train_loss']) + 1)
        
        plt.figure(figsize=(15, 5))
        
        # 训练损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(epochs, record['train_loss'], 'b-', label='Training Loss', linewidth=2)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 训练准确率曲线
        plt.subplot(1, 3, 2)
        plt.plot(epochs, record['train_accuracy'], 'g-', label='Training Accuracy', linewidth=2)
        plt.title('Training Accuracy Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 验证准确率曲线
        plt.subplot(1, 3, 3)
        plt.plot(epochs, record['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        plt.title('Validation Accuracy Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = "plots/training_curves_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
        print(f"训练曲线已更新: {plot_path}")
    
    # 保存检查点
    if (epoch + 1) % 5 == 0:
        checkpoint_path = f"{checkpoint_dir}/cp-{epoch + 1:04d}.weights.h5"
        model.save_weights(checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
    
    # 早停检查
    early_stopping.on_epoch_end(epoch, model, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc)
    if early_stopping.early_stop:
        print(f"早停触发！在第 {epoch + 1} 轮停止训练")
        break
    
    # 学习率调度
    lr_scheduler.on_epoch_end(epoch, model.optimizer, avg_val_loss)
    current_lr = model.optimizer.learning_rate.numpy()
    print(f"当前学习率: {current_lr:.8f}")

print("\n训练完成！")
print(f"最终训练损失: {record['train_loss'][-1]:.6f}")
print(f"最终验证准确率: {record['val_accuracy'][-1]:.6f}")

# 绘制训练曲线
def plot_training_curves(record):
    """绘制并保存训练曲线"""
    epochs = range(1, len(record['train_loss']) + 1)
    
    # 创建两个版本的图表：中文版和英文版
    create_plot_version(epochs, record, "chinese", "plots/training_curves_heatmap.png")
    create_plot_version(epochs, record, "english", "plots/training_curves_heatmap_english.png")

def create_plot_version(epochs, record, version, save_path):
    """创建指定版本的训练曲线"""
    plt.figure(figsize=(15, 5))
    
    if version == "chinese":
        # 中文版本
        labels = {
            'train_loss': '训练损失',
            'train_acc': '训练准确率', 
            'val_acc': '验证准确率',
            'train_loss_title': '训练损失曲线',
            'train_acc_title': '训练准确率曲线',
            'val_acc_title': '验证准确率曲线'
        }
    else:
        # 英文版本
        labels = {
            'train_loss': 'Training Loss',
            'train_acc': 'Training Accuracy',
            'val_acc': 'Validation Accuracy', 
            'train_loss_title': 'Training Loss Curve',
            'train_acc_title': 'Training Accuracy Curve',
            'val_acc_title': 'Validation Accuracy Curve'
        }
    
    # 训练损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, record['train_loss'], 'b-', label=labels['train_loss'], linewidth=2)
    plt.title(labels['train_loss_title'], fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 训练准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, record['train_accuracy'], 'g-', label=labels['train_acc'], linewidth=2)
    plt.title(labels['train_acc_title'], fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 验证准确率曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, record['val_accuracy'], 'r-', label=labels['val_acc'], linewidth=2)
    plt.title(labels['val_acc_title'], fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    
    plt.close()

# 绘制训练曲线
plot_training_curves(record)

# 保存最终模型
final_checkpoint = f"{checkpoint_dir}/cp-{len(record['train_loss']):04d}.weights.h5"
model.save_weights(final_checkpoint)
print(f"最终模型已保存: {final_checkpoint}")
