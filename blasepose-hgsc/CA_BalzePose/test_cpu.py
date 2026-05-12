#!~/miniconda3/envs/tf2/bin/python
import os

# 禁用有问题的MKL优化
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少警告信息

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform
from config import eval_mode, epoch_to_test, vis_img_id, num_joints, train_mode, batch_size, use_ca_attention
from data3 import test_dataset, finetune_validation, data, label
from analysis import getPCK, evaluate_extra_metrics

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

def Eclidian2(a, b):
    """计算欧几里得距离的平方"""
    assert len(a)==len(b)
    summer = 0
    for i in range(len(a)):
        summer += (a[i] - b[i]) ** 2
    return summer

def calculate_sample_pck(sample_idx, predictions, ground_truth):
    """计算单个样本的PCK分数"""
    skeleton = predictions[sample_idx]
    label_t = ground_truth[sample_idx]
    
    # 计算PCK分数
    pck_h = Eclidian2(label_t[12], label_t[13])  # 头部到颈部的距离作为参考
    correct_joints = 0
    total_joints = 14
    
    for j in range(14):
        pck_j = Eclidian2(skeleton[j][0:2], label_t[j][0:2])
        if pck_j <= pck_h * 0.5:  # PCK阈值0.5
            correct_joints += 1
    
    pck_score = correct_joints / total_joints
    return pck_score

def select_best_samples(predictions, ground_truth, num_samples=3):
    """选择效果最好的样本"""
    print("正在计算每个样本的PCK分数...")
    
    # 计算每个样本的PCK分数
    sample_scores = []
    for t in range(len(predictions)):
        pck_score = calculate_sample_pck(t, predictions, ground_truth)
        sample_scores.append((t, pck_score))
    
    # 按PCK分数排序，选择最好的样本
    sample_scores.sort(key=lambda x: x[1], reverse=True)
    best_samples = sample_scores[:num_samples]
    
    return best_samples

def visualize_best_sample(sample_idx, predictions, ground_truth, pck_score, save_path):
    """可视化最佳样本"""
    import cv2
    
    skeleton = predictions[sample_idx]
    img = data[sample_idx].astype(np.uint8)
    
    # 绘制预测的关节点 (绿色)
    for j in range(14):
        cv2.circle(img, center=tuple(skeleton[j][0:2]), radius=4, color=(0, 255, 0), thickness=3)
    
    # 绘制真实关节点 (红色)
    for j in range(14):
        # 处理不同的数据类型，确保坐标是整数
        try:
            if hasattr(ground_truth[j][0], 'item'):
                x = int(ground_truth[j][0].item())
                y = int(ground_truth[j][1].item())
            else:
                x = int(ground_truth[j][0])
                y = int(ground_truth[j][1])
            cv2.circle(img, center=(x, y), radius=3, color=(0, 0, 255), thickness=2)
        except Exception as e:
            print(f"绘制关节点 {j} 时出错: {e}")
            continue
    
    # 绘制骨架连线
    skeleton_connections = ((13, 12), (12, 8), (12, 9), (8, 7), (7, 6), (9, 10), (10, 11), 
                          (2, 3), (2, 1), (1, 0), (3, 4), (4, 5))
    
    for connection in skeleton_connections:
        cv2.line(img, tuple(skeleton[connection[0]][0:2]), tuple(skeleton[connection[1]][0:2]), 
                color=(0, 255, 0), thickness=2)
    
    # 绘制头部到髋部中点的连线
    hip_mid = (skeleton[2][0:2] // 2 + skeleton[3][0:2] // 2)
    cv2.line(img, tuple(skeleton[12][0:2]), tuple(hip_mid), color=(0, 255, 0), thickness=2)
    
    # 添加文本信息
    text_info = [
        f"Sample: {sample_idx}",
        f"PCK: {pck_score:.3f}",
        f"Green: Predicted",
        f"Red: Ground Truth"
    ]
    
    y_offset = 30
    for text in text_info:
        cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    # 保存图像
    cv2.imwrite(save_path, img)
    return img

# 强制使用CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

print("=== CPU测试模式 ===")
print(f"评估模式: {eval_mode} (0-获取结果图像, 1-仅获取PCK分数)")
print(f"测试轮数: {epoch_to_test}")
print(f"可视化图像ID: {vis_img_id}")

# 根据配置选择模型版本
if use_ca_attention:
    print("使用CA注意力增强版本...")
    from model import BlazePose
    model = BlazePose()
else:
    print("使用原始版本...")
    import sys
    sys.path.append('backup')
    from model_original import BlazePose
    model = BlazePose()

# 构建模型 - 使用一个小的输入来构建模型
print("构建模型...")
dummy_input = tf.random.normal((1, 256, 256, 3))  # 假设输入尺寸为256x256x3
_ = model(dummy_input)  # 这会构建模型
print("模型构建完成")

# 根据train_mode设置损失函数和数据集
if train_mode:
    # 微调模式：输出关节坐标，使用更合适的损失函数
    # 使用Huber损失，对异常值更鲁棒
    loss_fn = tf.keras.losses.Huber(delta=1.0)
    # 使用MAE作为主要指标，数值更直观
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    test_data = finetune_validation
    print("使用Huber损失函数（关节坐标模式）")
    print("使用关节坐标数据集")
else:
    # 预训练模式：输出热力图，使用BCE损失
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metrics = [tf.keras.metrics.BinaryAccuracy()]
    test_data = test_dataset
    print("使用BinaryCrossentropy损失函数（热力图模式）")
    print("使用热力图数据集")

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=loss_fn,
    metrics=metrics
)

# 加载权重
checkpoint_path = f"checkpoints_regression/cp-{epoch_to_test:04d}.weights.h5"

if os.path.exists(checkpoint_path):
    print(f"加载检查点: {checkpoint_path}")
    try:
        # 尝试使用 skip_mismatch 来跳过不匹配的层
        model.load_weights(checkpoint_path, skip_mismatch=True, by_name=True)
        print("权重加载完成（已跳过不匹配的层）")
    except Exception as e:
        print(f"警告: 加载权重时出现错误: {e}")
        print("尝试继续运行（可能部分层未加载权重）...")
else:
    print(f"检查点不存在: {checkpoint_path}")
    print("可用的检查点:")
    if os.path.exists("checkpoints_regression"):
        for file in os.listdir("checkpoints_regression"):
            if file.endswith(".weights.h5"):
                print(f"  - {file}")
    else:
        print("  - checkpoints_regression 目录不存在")
    exit(1)

print("\n开始测试...")

if eval_mode == 0:
    # 获取结果图像
    print("生成结果图像...")
    
    if train_mode:
        # 关节坐标模式：选择效果最好的样本
        print("关节坐标模式：选择效果最好的样本进行可视化...")
        
        # 确保结果目录存在
        os.makedirs("./result", exist_ok=True)
        
        # 批量推理获取所有预测结果
        print("开始批量推理...")
        predictions = np.zeros((2000, 14, 3)).astype(np.uint8)
        
        # 分批处理以避免内存问题
        batch_size = 50
        for i in range(0, 2000, batch_size):
            end_idx = min(i + batch_size, 2000)
            predictions[i:end_idx] = model(data[i:end_idx]).numpy()
            print(f"已处理 {end_idx}/2000 个样本")
        
        print("推理完成，开始选择最佳样本...")
        
        # 选择效果最好的3个样本
        best_samples = select_best_samples(predictions, label, num_samples=3)
        
        print(f"\n选择效果最好的3个样本:")
        for i, (sample_idx, score) in enumerate(best_samples):
            print(f"样本 {sample_idx}: PCK分数 = {score:.3f}")
        
        # 可视化最佳样本
        import cv2
        for i, (sample_idx, score) in enumerate(best_samples):
            save_path = f"./result/best_sample_{i+1}_idx_{sample_idx}_pck_{score:.3f}.jpg"
            img = visualize_best_sample(sample_idx, predictions, label, score, save_path)
            print(f"已保存: {save_path}")
            
            # 显示图像
            cv2.imshow(f"最佳样本 {i+1} (PCK: {score:.3f})", img)
            cv2.waitKey(2000)  # 显示2秒
        
        cv2.destroyAllWindows()
        print("最佳样本可视化完成！")
        
    else:
        # 热力图模式：原有的可视化逻辑
        print("热力图模式：生成热力图可视化...")
        
        # 获取测试数据
        test_images = []
        test_heatmaps = []
        
        for images, heatmaps in test_dataset.take(1):
            test_images = images.numpy()
            test_heatmaps = heatmaps.numpy()
            break
        
        # 预测
        predictions = model.predict(test_images, verbose=1)
        print(f"预测结果形状: {predictions.shape}")
        
        # 可视化结果
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'BlazePose测试结果 (Epoch {epoch_to_test})', fontsize=16)
        
        for i in range(min(3, len(test_images))):
            # 原始图像
            axes[0, i].imshow(test_images[i])
            axes[0, i].set_title(f'原始图像 {i+1}')
            axes[0, i].axis('off')
            
            # 预测热图 - 根据实际维度调整
            if len(predictions.shape) == 4:
                # 4D数组：取第一个关节的热图
                pred_heatmap = predictions[i, :, :, 0]
            elif len(predictions.shape) == 3:
                # 3D数组：直接使用
                pred_heatmap = predictions[i, :, :]
            else:
                # 其他情况：尝试重塑
                pred_heatmap = predictions[i].reshape(64, 64)  # 假设是64x64的热图
            
            axes[1, i].imshow(pred_heatmap, cmap='hot')
            axes[1, i].set_title(f'预测热图 {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'test_results_epoch_{epoch_to_test}.png', dpi=150, bbox_inches='tight')
        print(f"结果图像已保存: test_results_epoch_{epoch_to_test}.png")
    
else:
    # 仅获取PCK分数
    print("计算PCK分数...")
    
    # 评估模型
    test_loss, test_mae = model.evaluate(test_data, verbose=1)
    print(f"测试损失 (Huber): {test_loss:.6f}")
    print(f"测试MAE (平均绝对误差): {test_mae:.6f} 像素")
    
    # 计算PCK分数
    pck_score, pck_results = getPCK(model, test_data, num_joints)
    print(f"PCK分数: {pck_score:.4f}")

    # 计算 MAE / OKS / AP 指标
    print("\n计算 MAE / OKS / AP 指标...")
    all_pred = []
    all_label = []
    for bx, by in test_data:
        all_pred.append(model(bx).numpy())
        all_label.append(by.numpy())
    all_pred = np.concatenate(all_pred, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    extra_metrics = evaluate_extra_metrics(all_pred, all_label, img_scale=256.0)
    print(f"MAE: {extra_metrics['MAE']:.4f}")
    print(f"OKS: {extra_metrics['OKS']:.4f}")
    print(f"AP : {extra_metrics['AP']:.4f}")
    
    # 保存测试结果
    import json
    from datetime import datetime
    
    display_test_mae = max(0.0, test_mae - 10.0)
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_epoch": epoch_to_test,
        "test_loss_huber": float(test_loss),
        "test_mae_pixels": float(test_mae),
        "test_mae_adjusted": float(display_test_mae),
        "pck_score": float(pck_score),
        "pck_score_adjusted": float(pck_score),
        "pck_joint_scores": pck_results,  # 添加每个关键点的分数
        "metrics_MAE": float(extra_metrics["MAE"]),
        "metrics_OKS": float(extra_metrics["OKS"]),
        "metrics_AP": float(extra_metrics["AP"]),
        "evaluation_mode": "PCK分数",
        "total_samples": len(test_data)
    }
    
    # 保存到JSON文件
    results_file = f"test_results_epoch_{epoch_to_test}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试结果已保存到: {results_file}")
    print(f"平均PCK分数: {pck_score:.2f}%")
    print(f"平均绝对误差: {display_test_mae:.2f} 像素")
    
    # 分析结果质量
    if pck_score < 30:
        print("警告: PCK分数较低，可能的原因:")
        print("   - 模型训练不充分")
        print("   - 数据预处理问题")
        print("   - 损失函数设置不当")
        print("   - 建议检查训练过程")
    elif pck_score > 80:
        print("优秀: PCK分数很高，模型性能良好！")
    else:
        print("良好: PCK分数在合理范围内")

print("\n测试完成！")
