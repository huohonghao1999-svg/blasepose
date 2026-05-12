"""
早停机制配置文件
可以根据需要调整早停参数
"""

# 早停机制配置
EARLY_STOPPING_CONFIG = {
    # 基础配置
    'patience': 50,                    # 耐心值：连续多少个epoch没有改善就停止
    'min_delta': 0.001,               # 最小改善阈值
    'monitor': 'val_loss',            # 监控指标：'val_loss', 'val_accuracy', 'loss'
    'mode': 'min',                    # 模式：'min'表示越小越好，'max'表示越大越好
    'verbose': 1,                     # 详细程度：0=静默，1=显示信息
    'save_best_only': True,           # 是否只保存最佳模型
    'restore_best_weights': True,     # 是否恢复最佳权重
}

# 学习率调度器配置
LR_SCHEDULER_CONFIG = {
    'factor': 0.5,                    # 学习率衰减因子
    'patience': 5,                    # 耐心值：多少个epoch无改善就降低学习率
    'min_lr': 1e-7,                   # 最小学习率
    'verbose': 1,                     # 详细程度
}

# 不同场景的预设配置
PRESETS = {
    # 快速训练（适合调试）
    'fast': {
        'patience': 5,
        'min_delta': 0.01,
        'monitor': 'val_loss',
        'verbose': 1
    },
    
    # 标准训练（推荐）
    'standard': {
        'patience': 50,
        'min_delta': 0.001,
        'monitor': 'val_loss',
        'verbose': 1
    },
    
    # 精细训练（追求最佳性能）
    'precise': {
        'patience': 100,
        'min_delta': 0.0001,
        'monitor': 'val_loss',
        'verbose': 1
    },
    
    # 监控准确率（适合分类任务）
    'accuracy': {
        'patience': 50,
        'min_delta': 0.001,
        'monitor': 'val_accuracy',
        'mode': 'max',
        'verbose': 1
    }
}

def get_config(preset='standard'):
    """
    获取早停配置
    
    Args:
        preset: 预设名称 ('fast', 'standard', 'precise', 'accuracy')
    
    Returns:
        dict: 早停配置字典
    """
    if preset in PRESETS:
        config = EARLY_STOPPING_CONFIG.copy()
        config.update(PRESETS[preset])
        return config
    else:
        print(f"⚠️ 未知预设: {preset}, 使用标准配置")
        return EARLY_STOPPING_CONFIG

def print_available_presets():
    """打印可用的预设配置"""
    print("📋 可用的早停预设配置:")
    print("-" * 50)
    
    for name, config in PRESETS.items():
        print(f"🎯 {name.upper()}:")
        print(f"   耐心值: {config['patience']} epochs")
        print(f"   改善阈值: {config['min_delta']}")
        print(f"   监控指标: {config['monitor']}")
        if 'mode' in config:
            print(f"   模式: {config['mode']}")
        print()

if __name__ == "__main__":
    print_available_presets()
    
    # 测试配置获取
    print("🔧 测试配置获取:")
    for preset in ['fast', 'standard', 'precise', 'accuracy']:
        config = get_config(preset)
        print(f"{preset}: patience={config['patience']}, min_delta={config['min_delta']}")
