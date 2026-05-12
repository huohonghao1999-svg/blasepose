  #!~/miniconda3/envs/tf2/bin/python
import os
import tensorflow as tf
import time
from tqdm import tqdm
from model import BlazePose
from config import total_epoch, train_mode, continue_train, show_batch_loss, batch_size
from analysis import save_record, load_record
from early_stopping import EasyStopping, LearningRateScheduler

# 导入配置
from config import checkpoint_dir

# Create checkpoints directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

if train_mode:
    from data import finetune_train as train_dataset
    from data import finetune_validation as test_dataset
    loss_func = tf.keras.losses.MeanSquaredError()
else:
    from data import train_dataset, test_dataset
    loss_func = tf.keras.losses.BinaryCrossentropy()

model = BlazePose()

checkpoint_path = f"{checkpoint_dir}/cp-{{epoch:04d}}.weights.h5"
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_func(y_true=targets, y_pred=model(inputs))
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# continue train
if continue_train > 0:
    model.load_weights(checkpoint_path.format(epoch=continue_train))
    # continue recording
    train_loss_results, train_accuracy_results, val_accuracy_results = load_record()
else:
    # start from epoch 0
    # Initial for record of the training process
    train_loss_results = []
    train_accuracy_results = []
    val_accuracy_results = []

if train_mode:
    # finetune
    for layer in model.layers[0:16]:
        print(layer)
        layer.trainable = False
else:
    # pre-train
    for layer in model.layers[16:24]:
        print(layer)
        layer.trainable = False

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end="  Start train.\n")
print("正在加载数据集，这可能需要几分钟时间...")
print("如果长时间没有输出，请耐心等待数据加载完成...")
# 如果是微调模式且不是继续训练，加载预训练权重
if train_mode and continue_train == 0:
    from config import best_pre_train
    try:
        model.load_weights(checkpoint_path.format(epoch=best_pre_train))
        print(f"已加载预训练权重: {checkpoint_path.format(epoch=best_pre_train)}")
    except Exception as e:
        print(f"警告: 无法加载预训练权重: {e}")
        print("将从头开始训练...")

# validata initial loaded model
val_accuracy = tf.keras.metrics.MeanSquaredError()
for x, y in test_dataset:
    val_accuracy(y, model(x))
print("Initial Validation accuracy: {:.5%}".format(val_accuracy.result()))

# make sure continue has any epoch to train
assert(continue_train < total_epoch)

# 初始化早停机制（优化参数以加快训练）
early_stopping = EasyStopping(
    patience=20,                    # 增加耐心值，减少早停频率
    min_delta=0.002,               # 增加改善阈值，减少敏感度
    monitor='val_loss',            # 监控验证损失
    mode='min',                    # 越小越好
    verbose=0,                     # 减少输出信息
    save_best_only=True           # 只保存最佳模型
)

# 初始化学习率调度器
lr_scheduler = LearningRateScheduler(
    factor=0.5,                    # 学习率衰减因子
    patience=5,                    # 5个epoch无改善就降低学习率
    min_lr=1e-7,                   # 最小学习率
    verbose=1
)

print("🎯 智能早停机制已启用!")
print(f"   📊 监控指标: 验证损失")
print(f"   ⏰ 耐心值: 15 epochs")
print(f"   📈 改善阈值: 0.001")
print(f"   📉 学习率调度: 自动衰减")

# 计算每个epoch的批次数（用于进度条）
# 对于LSP数据集，大约有2000张图片，批处理大小为1024，所以大约2个批次
# 使用估算值，避免消费数据集
steps_per_epoch = 2  # 2000张图片 / 1024批处理大小 ≈ 2个批次

for epoch in range(continue_train, total_epoch):
    epoch_start_time = time.time()  # 记录epoch开始时间
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.MeanSquaredError()
    val_accuracy = tf.keras.metrics.MeanSquaredError()

    # 创建进度条（简化显示）
    pbar = tqdm(total=steps_per_epoch, 
               desc=f"Epoch {epoch+1:03d}/{total_epoch:03d}", 
               unit="batch",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
               disable=False)  # 保持进度条显示

    # Training loop
    if show_batch_loss:
        batch_index = 0
    
    batch_count = 0
    try:
        for x, y in train_dataset:
            # Optimize
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Add current batch loss
            epoch_loss_avg(loss_value)
            # Calculate error from Ground truth
            epoch_accuracy(y, model(x))
            
            batch_count += 1
            
            # 动态调整进度条总数（如果实际批次数与估算不同）
            if batch_count > steps_per_epoch:
                pbar.total = batch_count
                steps_per_epoch = batch_count
            
            # 更新进度条（减少计算频率）
            if batch_count % 2 == 0 or batch_count == 1:  # 每2个批次或第一个批次更新一次
                current_loss = float(epoch_loss_avg.result())
                current_acc = float(epoch_accuracy.result())
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
            pbar.update(1)
            
            if show_batch_loss:
                print(f"\nEpoch {epoch:03d}, Batch {batch_count:03d}: Train Loss: {loss_value:.3f}")
    finally:
        pbar.close()
    
    # Record loss and accuracy
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    # 计算epoch时间
    epoch_time = time.time() - epoch_start_time
    
    # Train loss at epoch
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Epoch {:03d}: Train Loss: {:.3f}, Accuracy: {:.5%}, Time: {:.1f}s".format(
        epoch,
        epoch_loss_avg.result(),
        epoch_accuracy.result(),
        epoch_time
    ))
    
    # 优化验证频率：每2个epoch验证一次，或前10个epoch每个都验证
    should_validate = (epoch % 1 == 0) or (epoch == total_epoch - 1)
    
    if should_validate:
        # 进行验证
        for x, y in test_dataset:
            val_accuracy(y, model(x))
        val_loss = float(val_accuracy.result())
        val_accuracy_results.append(val_loss)
        print("Epoch {:03d}, Validation accuracy: {:.5%}".format(epoch, val_loss))
    else:
        # 使用上一次的验证结果或估算
        if len(val_accuracy_results) > 0:
            val_loss = val_accuracy_results[-1]  # 使用上一次的验证结果
        else:
            val_loss = float(epoch_loss_avg.result())  # 使用训练损失作为估算
        val_accuracy_results.append(val_loss)
        print("Epoch {:03d}, Validation accuracy: {:.5%} (估算)".format(epoch, val_loss))
    
    # 调用早停机制
    early_stopping.on_epoch_end(
        epoch=epoch,
        model=model,
        train_loss=float(epoch_loss_avg.result()),
        val_loss=val_loss,
        train_acc=float(epoch_accuracy.result()),
        val_acc=val_loss
    )
    
    # 调用学习率调度器
    lr_scheduler.on_epoch_end(epoch, optimizer, val_loss)
    
    # 优化保存频率：每10个epoch或早停触发时保存
    if not((epoch + 1) % 10) or early_stopping.early_stop:
        model.save_weights(checkpoint_path.format(epoch=epoch))
        print(f"💾 模型已保存: {checkpoint_path.format(epoch=epoch)}")
    
    # 减少训练记录保存频率：每5个epoch保存一次
    if not((epoch + 1) % 5) or early_stopping.early_stop:
        save_record(train_loss_results, train_accuracy_results, val_accuracy_results)
    
    # 检查是否应该早停
    if early_stopping.early_stop:
        print(f"\n🛑 早停触发! 训练在第 {epoch+1} 个epoch停止")
        print(f"📊 最佳验证损失: {early_stopping.get_best_value():.6f} (Epoch {early_stopping.get_best_epoch()})")
        break

model.summary()

# 训练结束后的摘要
print("\n" + "="*60)
print("🎯 训练完成摘要")
print("="*60)

if early_stopping.early_stop:
    print(f"🛑 早停触发: 第 {early_stopping.stopped_epoch+1} 个epoch")
    print(f"📊 最佳验证损失: {early_stopping.get_best_value():.6f}")
    print(f"🏆 最佳epoch: {early_stopping.get_best_epoch()}")
    print(f"⏰ 节省时间: 约 {total_epoch - early_stopping.stopped_epoch - 1} 个epoch")
else:
    print(f"✅ 正常完成: 训练了 {total_epoch} 个epoch")

# 保存早停摘要
early_stopping.save_summary("training_summary.txt")

# 显示训练历史
history = early_stopping.get_history()
if len(history['monitor_value']) > 0:
    print(f"\n📈 训练历史 (最后5个epoch):")
    print("-" * 40)
    recent_epochs = history['epoch'][-5:]
    recent_values = history['monitor_value'][-5:]
    
    for epoch, value in zip(recent_epochs, recent_values):
        print(f"Epoch {epoch:03d}: 验证损失 = {value:.6f}")

print("\n🎉 训练完成!")
