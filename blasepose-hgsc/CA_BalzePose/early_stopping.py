"""
智能早停机制 (EasyStopping)
监控训练过程，自动判断何时停止训练以避免过拟合
"""

import numpy as np
import os
import time

class EasyStopping:
    """
    智能早停机制类
    
    功能：
    1. 监控验证损失，防止过拟合
    2. 监控训练损失，检测收敛
    3. 自动保存最佳模型
    4. 提供多种停止条件
    """
    
    def __init__(self, 
                 patience=10,           # 耐心值：连续多少个epoch没有改善就停止
                 min_delta=0.001,       # 最小改善阈值
                 restore_best_weights=True,  # 是否恢复最佳权重
                 monitor='val_loss',    # 监控指标：'val_loss', 'val_accuracy', 'loss'
                 mode='min',           # 模式：'min'表示越小越好，'max'表示越大越好
                 verbose=1,            # 详细程度：0=静默，1=显示信息
                 save_best_only=True,  # 是否只保存最佳模型
                 checkpoint_dir="checkpoints"):  # 检查点目录
        
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.checkpoint_dir = checkpoint_dir
        
        # 内部状态
        self.wait = 0                    # 等待计数器
        self.stopped_epoch = 0           # 停止的epoch
        self.best_epoch = 0              # 最佳epoch
        self.best_value = None           # 最佳值
        self.best_weights = None         # 最佳权重
        self.early_stop = False          # 是否早停
        
        # 历史记录
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'monitor_value': []
        }
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if self.verbose:
            print(f"🎯 早停机制已启用:")
            print(f"   📊 监控指标: {monitor}")
            print(f"   ⏰ 耐心值: {patience} epochs")
            print(f"   📈 改善阈值: {min_delta}")
            print(f"   💾 保存最佳: {save_best_only}")
    
    def on_epoch_end(self, epoch, model, train_loss, val_loss=None, 
                    train_acc=None, val_acc=None, **kwargs):
        """
        每个epoch结束时调用
        
        Args:
            epoch: 当前epoch数
            model: 模型对象
            train_loss: 训练损失
            val_loss: 验证损失
            train_acc: 训练准确率
            val_acc: 验证准确率
        """
        
        # 记录历史
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss) if val_loss is not None else None)
        self.history['train_acc'].append(float(train_acc) if train_acc is not None else None)
        self.history['val_acc'].append(float(val_acc) if val_acc is not None else None)
        
        # 确定监控值
        if self.monitor == 'val_loss' and val_loss is not None:
            current_value = float(val_loss)
        elif self.monitor == 'val_accuracy' and val_acc is not None:
            current_value = float(val_acc)
        elif self.monitor == 'loss':
            current_value = float(train_loss)
        else:
            # 如果没有验证数据，使用训练损失
            current_value = float(train_loss)
            if self.verbose:
                print(f"⚠️ 警告: 没有验证数据，使用训练损失作为监控指标")
        
        self.history['monitor_value'].append(current_value)
        
        # 初始化最佳值
        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = epoch
            self._save_best_weights(model)
            if self.verbose:
                print(f"📊 初始最佳值: {self.best_value:.6f}")
            return
        
        # 检查是否有改善
        improved = self._is_improved(current_value)
        
        if improved:
            # 有改善，更新最佳值
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            self._save_best_weights(model)
            
            if self.verbose:
                print(f"✅ Epoch {epoch:03d}: 改善! {self.monitor}: {current_value:.6f} "
                      f"(最佳: {self.best_value:.6f})")
        else:
            # 没有改善，增加等待计数
            self.wait += 1
            
            if self.verbose:
                print(f"⏳ Epoch {epoch:03d}: 无改善 ({self.wait}/{self.patience}) "
                      f"{self.monitor}: {current_value:.6f} (最佳: {self.best_value:.6f})")
        
        # 检查是否应该早停
        if self.wait >= self.patience:
            self.early_stop = True
            self.stopped_epoch = epoch
            
            if self.verbose:
                print(f"\n🛑 早停触发!")
                print(f"   📊 最佳 {self.monitor}: {self.best_value:.6f} (Epoch {self.best_epoch})")
                print(f"   ⏰ 耐心值耗尽: {self.patience} epochs 无改善")
                print(f"   🎯 停止于 Epoch {epoch}")
            
            # 恢复最佳权重
            if self.restore_best_weights and self.best_weights is not None:
                model.set_weights(self.best_weights)
                if self.verbose:
                    print(f"💾 已恢复最佳权重 (Epoch {self.best_epoch})")
    
    def _is_improved(self, current_value):
        """检查当前值是否有改善"""
        if self.mode == 'min':
            return current_value < (self.best_value - self.min_delta)
        else:  # mode == 'max'
            return current_value > (self.best_value + self.min_delta)
    
    def _save_best_weights(self, model):
        """保存最佳权重"""
        if self.save_best_only:
            # 使用模型的get_weights()方法获取所有权重
            self.best_weights = model.get_weights()
    
    def get_best_epoch(self):
        """获取最佳epoch"""
        return self.best_epoch
    
    def get_best_value(self):
        """获取最佳值"""
        return self.best_value
    
    def get_history(self):
        """获取训练历史"""
        return self.history
    
    def save_summary(self, filepath="early_stopping_summary.txt"):
        """保存早停摘要"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("🎯 早停机制训练摘要\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"📊 监控指标: {self.monitor}\n")
            f.write(f"⏰ 耐心值: {self.patience} epochs\n")
            f.write(f"📈 改善阈值: {self.min_delta}\n")
            f.write(f"🎯 最佳值: {self.best_value:.6f} (Epoch {self.best_epoch})\n")
            f.write(f"🛑 停止原因: {'早停触发' if self.early_stop else '正常完成'}\n")
            f.write(f"📅 停止时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if self.early_stop:
                f.write("📈 训练历史 (最后10个epoch):\n")
                f.write("-" * 30 + "\n")
                recent_epochs = self.history['epoch'][-10:]
                recent_values = self.history['monitor_value'][-10:]
                
                for epoch, value in zip(recent_epochs, recent_values):
                    f.write(f"Epoch {epoch:03d}: {self.monitor} = {value:.6f}\n")
        
        if self.verbose:
            print(f"📄 训练摘要已保存到: {filepath}")


class LearningRateScheduler:
    """
    学习率调度器
    当验证损失不再改善时，自动降低学习率
    """
    
    def __init__(self, 
                 factor=0.5,           # 学习率衰减因子
                 patience=5,           # 耐心值
                 min_lr=1e-7,          # 最小学习率
                 verbose=1):
        
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.wait = 0
        self.best_loss = None
        self.original_lr = None
    
    def on_epoch_end(self, epoch, optimizer, val_loss):
        """每个epoch结束时调用"""
        
        if self.original_lr is None:
            self.original_lr = float(optimizer.learning_rate)
        
        if self.best_loss is None:
            self.best_loss = val_loss
            return
        
        if val_loss < (self.best_loss - 1e-6):
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                old_lr = float(optimizer.learning_rate)
                new_lr = max(old_lr * self.factor, self.min_lr)
                optimizer.learning_rate.assign(new_lr)
                self.wait = 0
                
                if self.verbose:
                    print(f"📉 学习率调整: {old_lr:.2e} → {new_lr:.2e}")


# 使用示例
if __name__ == "__main__":
    # 创建早停机制
    early_stopping = EasyStopping(
        patience=15,           # 15个epoch无改善就停止
        min_delta=0.001,       # 改善阈值
        monitor='val_loss',    # 监控验证损失
        mode='min',           # 越小越好
        verbose=1
    )
    
    print("早停机制测试完成!")
