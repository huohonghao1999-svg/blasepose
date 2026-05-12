#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯CPU性能优化模块
针对没有GPU和TensorFlow的环境进行优化
"""

import os
import sys
import time
import numpy as np
import psutil
from multiprocessing import Pool, cpu_count
import threading
import queue

class CPUOptimizer:
    """CPU优化器"""
    
    def __init__(self):
        self.cpu_count = cpu_count()
        self.optimization_config = {
            'max_workers': min(self.cpu_count, 8),
            'chunk_size': 1000,
            'memory_limit_gb': 6,
            'batch_size': 64,
            'prefetch_size': 4,
        }
        
    def get_system_info(self):
        """获取系统信息"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'cpu_count': self.cpu_count,
                'cpu_percent': cpu_percent,
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
            }
        except Exception as e:
            print(f"无法获取系统信息: {e}")
            return None
    
    def optimize_memory_usage(self):
        """优化内存使用"""
        print("优化内存使用...")
        
        # 获取系统信息
        sys_info = self.get_system_info()
        if not sys_info:
            return False
        
        # 根据可用内存调整配置
        available_memory = sys_info['memory_available_gb']
        
        if available_memory < 4:
            # 低内存系统
            self.optimization_config['batch_size'] = 32
            self.optimization_config['max_workers'] = 2
            self.optimization_config['chunk_size'] = 500
            print("   低内存模式: 批处理大小=32, 工作进程=2")
        elif available_memory < 8:
            # 中等内存系统
            self.optimization_config['batch_size'] = 64
            self.optimization_config['max_workers'] = 4
            self.optimization_config['chunk_size'] = 1000
            print("   中等内存模式: 批处理大小=64, 工作进程=4")
        else:
            # 高内存系统
            self.optimization_config['batch_size'] = 128
            self.optimization_config['max_workers'] = min(8, self.cpu_count)
            self.optimization_config['chunk_size'] = 2000
            print("   高内存模式: 批处理大小=128, 工作进程=8")
        
        print(f"   可用内存: {available_memory:.1f}GB")
        print(f"   批处理大小: {self.optimization_config['batch_size']}")
        print(f"   工作进程数: {self.optimization_config['max_workers']}")
        
        return True
    
    def optimize_cpu_usage(self):
        """优化CPU使用"""
        print("优化CPU使用...")
        
        # 设置线程优先级
        try:
            import os
            # 设置进程优先级为高
            os.nice(-5)
            print("   进程优先级已提升")
        except:
            print("   无法设置进程优先级")
        
        # 优化多进程配置
        max_workers = self.optimization_config['max_workers']
        print(f"   最大工作进程数: {max_workers}")
        print(f"   CPU核心数: {self.cpu_count}")
        
        return True
    
    def create_data_loader(self, data_source, batch_size=None):
        """创建优化的数据加载器"""
        if batch_size is None:
            batch_size = self.optimization_config['batch_size']
        
        print(f"📦 创建数据加载器 (批处理大小: {batch_size})")
        
        class OptimizedDataLoader:
            def __init__(self, data_source, batch_size, max_workers):
                self.data_source = data_source
                self.batch_size = batch_size
                self.max_workers = max_workers
                self.queue = queue.Queue(maxsize=4)  # 预取4个批次
                self.thread = None
                self.stop_flag = False
            
            def _prefetch_worker(self):
                """预取工作线程"""
                try:
                    batch_data = []
                    for item in self.data_source:
                        if self.stop_flag:
                            break
                        batch_data.append(item)
                        if len(batch_data) >= self.batch_size:
                            self.queue.put(batch_data)
                            batch_data = []
                except Exception as e:
                    print(f"预取线程错误: {e}")
                finally:
                    if batch_data:
                        self.queue.put(batch_data)
            
            def start(self):
                """启动预取"""
                self.thread = threading.Thread(target=self._prefetch_worker)
                self.thread.daemon = True
                self.thread.start()
            
            def stop(self):
                """停止预取"""
                self.stop_flag = True
                if self.thread:
                    self.thread.join()
            
            def get_batch(self):
                """获取批次数据"""
                try:
                    return self.queue.get(timeout=5)
                except queue.Empty:
                    return None
        
        return OptimizedDataLoader(data_source, batch_size, self.optimization_config['max_workers'])
    
    def parallel_data_processing(self, data, process_func, chunk_size=None):
        """并行数据处理"""
        if chunk_size is None:
            chunk_size = self.optimization_config['chunk_size']
        
        print(f"并行数据处理 (块大小: {chunk_size})")
        
        # 将数据分块
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        # 并行处理
        with Pool(processes=self.optimization_config['max_workers']) as pool:
            results = pool.map(process_func, chunks)
        
        # 合并结果
        processed_data = []
        for result in results:
            processed_data.extend(result)
        
        print(f"   处理完成: {len(processed_data)} 个样本")
        return processed_data
    
    def monitor_performance(self, duration=10):
        """监控性能"""
        print(f"监控性能 ({duration}秒)...")
        
        start_time = time.time()
        cpu_samples = []
        memory_samples = []
        
        while time.time() - start_time < duration:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            cpu_samples.append(cpu_percent)
            memory_samples.append(memory_percent)
            
            time.sleep(0.5)
        
        avg_cpu = np.mean(cpu_samples)
        avg_memory = np.mean(memory_samples)
        max_cpu = np.max(cpu_samples)
        max_memory = np.max(memory_samples)
        
        print(f"   CPU使用率: 平均 {avg_cpu:.1f}%, 峰值 {max_cpu:.1f}%")
        print(f"   内存使用率: 平均 {avg_memory:.1f}%, 峰值 {max_memory:.1f}%")
        
        return {
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'avg_memory': avg_memory,
            'max_memory': max_memory
        }
    
    def benchmark_data_loading(self, data_source, num_batches=10):
        """基准测试数据加载性能"""
        print("🧪 基准测试数据加载性能...")
        
        # 测试优化前
        print("   测试优化前...")
        start_time = time.time()
        batch_count = 0
        for batch in data_source:
            batch_count += 1
            if batch_count >= num_batches:
                break
        time_before = time.time() - start_time
        
        # 测试优化后
        print("   测试优化后...")
        data_loader = self.create_data_loader(data_source)
        data_loader.start()
        
        start_time = time.time()
        batch_count = 0
        while batch_count < num_batches:
            batch = data_loader.get_batch()
            if batch is None:
                break
            batch_count += 1
        time_after = time.time() - start_time
        
        data_loader.stop()
        
        # 计算改善
        improvement = (time_before - time_after) / time_before * 100
        print(f"   数据加载速度提升: {improvement:.1f}%")
        
        return {
            'time_before': time_before,
            'time_after': time_after,
            'improvement': improvement
        }
    
    def get_optimal_config(self):
        """获取最优配置"""
        sys_info = self.get_system_info()
        if not sys_info:
            return self.optimization_config
        
        # 根据系统配置调整
        available_memory = sys_info['memory_available_gb']
        cpu_count = sys_info['cpu_count']
        
        if available_memory < 4:
            return {
                'batch_size': 32,
                'max_workers': 2,
                'chunk_size': 500,
                'prefetch_size': 2
            }
        elif available_memory < 8:
            return {
                'batch_size': 64,
                'max_workers': min(4, cpu_count),
                'chunk_size': 1000,
                'prefetch_size': 4
            }
        else:
            return {
                'batch_size': 128,
                'max_workers': min(8, cpu_count),
                'chunk_size': 2000,
                'prefetch_size': 8
            }

# 全局优化器实例
cpu_optimizer = CPUOptimizer()

def apply_cpu_optimizations():
    """应用CPU优化"""
    print("应用CPU优化...")
    
    # 优化内存使用
    memory_ok = cpu_optimizer.optimize_memory_usage()
    
    # 优化CPU使用
    cpu_ok = cpu_optimizer.optimize_cpu_usage()
    
    if memory_ok and cpu_ok:
        print("CPU优化应用成功!")
        return True
    else:
        print("❌ CPU优化应用失败!")
        return False

def get_optimal_batch_size():
    """获取最优批处理大小"""
    config = cpu_optimizer.get_optimal_config()
    return config['batch_size']

def monitor_system_performance():
    """监控系统性能"""
    return cpu_optimizer.get_system_info()

def benchmark_performance(data_source):
    """基准测试性能"""
    return cpu_optimizer.benchmark_data_loading(data_source)

if __name__ == "__main__":
    print("CPU性能优化模块测试")
    print("=" * 50)
    
    # 应用优化
    apply_cpu_optimizations()
    
    # 获取系统信息
    sys_info = monitor_system_performance()
    if sys_info:
        print(f"\n系统信息:")
        print(f"   CPU核心数: {sys_info['cpu_count']}")
        print(f"   CPU使用率: {sys_info['cpu_percent']:.1f}%")
        print(f"   内存总量: {sys_info['memory_total_gb']:.1f}GB")
        print(f"   可用内存: {sys_info['memory_available_gb']:.1f}GB")
        print(f"   内存使用率: {sys_info['memory_percent']:.1f}%")
    
    # 获取最优配置
    optimal_config = cpu_optimizer.get_optimal_config()
    print(f"\n最优配置:")
    for key, value in optimal_config.items():
        print(f"   {key}: {value}")
    
    print("\nCPU优化模块测试完成!")
