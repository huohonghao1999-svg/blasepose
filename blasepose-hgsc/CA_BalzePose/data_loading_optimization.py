#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载优化模块
解决数据加载瓶颈，提升CPU利用率
"""

import tensorflow as tf
import numpy as np
import os
from functools import partial

class DataLoadingOptimizer:
    """数据加载优化器"""
    
    def __init__(self):
        self.optimization_config = {
            'prefetch_buffer_size': tf.data.AUTOTUNE,
            'num_parallel_calls': tf.data.AUTOTUNE,
            'cache_size': 1000,
            'shuffle_buffer_size': 1000,
            'interleave_cycle_length': tf.data.AUTOTUNE,
            'interleave_block_length': 1,
        }
        
    def optimize_dataset(self, dataset, cache=True, prefetch=True, shuffle=True, 
                        parallel_calls=None, interleave=False):
        """优化数据集性能"""
        print("📦 优化数据集性能...")
        
        # 1. 缓存数据集
        if cache:
            dataset = dataset.cache()
            print("   ✅ 数据集缓存已启用")
        
        # 2. 打乱数据
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.optimization_config['shuffle_buffer_size']
            )
            print("   ✅ 数据打乱已启用")
        
        # 3. 并行处理
        if parallel_calls is None:
            parallel_calls = self.optimization_config['num_parallel_calls']
            
        dataset = dataset.map(
            lambda x, y: (x, y),
            num_parallel_calls=parallel_calls
        )
        print(f"   ✅ 并行数据处理已启用 (并行数: {parallel_calls})")
        
        # 4. 交错处理（可选）
        if interleave:
            dataset = dataset.interleave(
                lambda x: x,
                cycle_length=self.optimization_config['interleave_cycle_length'],
                block_length=self.optimization_config['interleave_block_length'],
                num_parallel_calls=parallel_calls
            )
            print("   ✅ 数据交错处理已启用")
        
        # 5. 预取数据
        if prefetch:
            dataset = dataset.prefetch(
                buffer_size=self.optimization_config['prefetch_buffer_size']
            )
            print("   ✅ 数据预取已启用")
        
        return dataset
    
    def create_optimized_data_pipeline(self, data_generator, batch_size, 
                                     num_workers=4, prefetch_factor=2):
        """创建优化的数据管道"""
        print("🏗️ 创建优化数据管道...")
        
        # 创建数据集
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 14, 2), dtype=tf.float32)
            )
        )
        
        # 应用优化
        dataset = self.optimize_dataset(
            dataset,
            cache=True,
            prefetch=True,
            shuffle=True,
            parallel_calls=num_workers
        )
        
        # 批处理
        dataset = dataset.batch(batch_size)
        
        print(f"   ✅ 数据管道创建完成")
        print(f"   📊 批处理大小: {batch_size}")
        print(f"   👥 并行工作数: {num_workers}")
        print(f"   🔄 预取因子: {prefetch_factor}")
        
        return dataset
    
    def optimize_memory_usage(self, dataset, memory_limit_gb=8):
        """优化内存使用"""
        print("💾 优化内存使用...")
        
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024**3)
            
            if available_memory < memory_limit_gb:
                print(f"   ⚠️ 可用内存: {available_memory:.1f}GB < {memory_limit_gb}GB")
                
                # 减少缓存大小
                self.optimization_config['cache_size'] = max(100, 
                    int(self.optimization_config['cache_size'] * 0.5))
                
                # 减少预取缓冲区
                self.optimization_config['prefetch_buffer_size'] = 1
                
                # 减少并行数
                self.optimization_config['num_parallel_calls'] = 2
                
                print("   🔄 已调整优化参数以适应内存限制")
            else:
                print(f"   ✅ 可用内存: {available_memory:.1f}GB (充足)")
                
        except ImportError:
            print("   ⚠️ psutil未安装，无法检查内存使用")
        
        return dataset
    
    def benchmark_data_loading(self, dataset, num_batches=10):
        """基准测试数据加载性能"""
        print("🧪 基准测试数据加载性能...")
        
        import time
        
        # 预热
        print("   🔥 预热数据加载...")
        for i, batch in enumerate(dataset.take(2)):
            pass
        
        # 测试
        print("   ⏱️ 开始性能测试...")
        start_time = time.time()
        
        batch_times = []
        for i, batch in enumerate(dataset.take(num_batches)):
            batch_start = time.time()
            # 模拟处理
            _ = batch
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            if i % 5 == 0:
                print(f"   📊 批次 {i+1}/{num_batches}: {batch_times[-1]:.3f}s")
        
        total_time = time.time() - start_time
        avg_batch_time = np.mean(batch_times)
        
        print(f"   📈 性能测试结果:")
        print(f"     总时间: {total_time:.3f}s")
        print(f"     平均批次时间: {avg_batch_time:.3f}s")
        print(f"     批次/秒: {1/avg_batch_time:.2f}")
        
        return {
            'total_time': total_time,
            'avg_batch_time': avg_batch_time,
            'batches_per_second': 1/avg_batch_time
        }
    
    def get_optimal_config(self, dataset_size, available_memory_gb=8):
        """获取最优配置"""
        print("🎯 计算最优配置...")
        
        # 根据数据集大小和内存调整配置
        if dataset_size < 1000:
            # 小数据集
            config = {
                'cache_size': min(500, dataset_size),
                'shuffle_buffer_size': min(500, dataset_size),
                'num_parallel_calls': 2,
                'prefetch_buffer_size': 1
            }
        elif dataset_size < 10000:
            # 中等数据集
            config = {
                'cache_size': min(1000, dataset_size // 2),
                'shuffle_buffer_size': min(1000, dataset_size // 2),
                'num_parallel_calls': 4,
                'prefetch_buffer_size': tf.data.AUTOTUNE
            }
        else:
            # 大数据集
            config = {
                'cache_size': 2000,
                'shuffle_buffer_size': 2000,
                'num_parallel_calls': 8,
                'prefetch_buffer_size': tf.data.AUTOTUNE
            }
        
        # 根据内存调整
        if available_memory_gb < 4:
            config['cache_size'] = min(config['cache_size'], 200)
            config['num_parallel_calls'] = min(config['num_parallel_calls'], 2)
        elif available_memory_gb < 8:
            config['cache_size'] = min(config['cache_size'], 500)
            config['num_parallel_calls'] = min(config['num_parallel_calls'], 4)
        
        print(f"   📊 最优配置:")
        for key, value in config.items():
            print(f"     {key}: {value}")
        
        return config

# 全局优化器实例
data_optimizer = DataLoadingOptimizer()

def optimize_training_data(dataset):
    """优化训练数据"""
    return data_optimizer.optimize_dataset(dataset)

def optimize_validation_data(dataset):
    """优化验证数据"""
    return data_optimizer.optimize_dataset(
        dataset, 
        cache=False,  # 验证数据不需要缓存
        shuffle=False  # 验证数据不需要打乱
    )

def benchmark_data_performance(dataset):
    """基准测试数据性能"""
    return data_optimizer.benchmark_data_loading(dataset)

if __name__ == "__main__":
    print("📦 数据加载优化模块测试")
    print("=" * 50)
    
    # 创建测试数据集
    def test_data_generator():
        for i in range(100):
            x = np.random.random((32, 224, 224, 3)).astype(np.float32)
            y = np.random.random((32, 14, 2)).astype(np.float32)
            yield x, y
    
    # 创建数据集
    dataset = tf.data.Dataset.from_generator(
        test_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(32, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(32, 14, 2), dtype=tf.float32)
        )
    )
    
    # 优化数据集
    optimized_dataset = data_optimizer.optimize_dataset(dataset)
    
    # 基准测试
    benchmark_data_performance(optimized_dataset)
    
    print("\n✅ 数据加载优化模块测试完成!")
