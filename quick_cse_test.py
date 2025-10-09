#!/usr/bin/env python3
"""
快速测试CSE算法的简化版本
"""

import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from metrics import calculate_cse_from_harmonic_features, calculate_cycle_sample_entropy

def quick_test_cse():
    """快速测试CSE算法"""
    print("=== 快速CSE测试 ===")
    
    # 创建测试数据：规律信号
    num_periods = 10
    num_harmonics = 10
    
    # 规律信号：所有周期的谐波特征都相同
    regular_features = []
    base_magnitudes = np.random.uniform(0.1, 1.0, num_harmonics)
    base_phases = np.random.uniform(-np.pi, np.pi, num_harmonics)
    base_features = np.concatenate([base_magnitudes, base_phases])
    
    for i in range(num_periods):
        regular_features.append(base_features.copy())
    
    # 随机信号：每个周期的谐波特征都不同
    random_features = []
    for i in range(num_periods):
        magnitudes = np.random.uniform(0.1, 1.0, num_harmonics)
        phases = np.random.uniform(-np.pi, np.pi, num_harmonics)
        features = np.concatenate([magnitudes, phases])
        random_features.append(features)
    
    # 测试规律信号
    regular_cse = calculate_cse_from_harmonic_features(regular_features, m=2, r=0.1)
    print(f"规律信号CSE: {regular_cse:.6f}")
    
    # 测试随机信号
    random_cse = calculate_cse_from_harmonic_features(random_features, m=2, r=0.1)
    print(f"随机信号CSE: {random_cse:.6f}")
    
    # 测试混合信号
    mixed_features = regular_features[:5] + random_features[5:]
    mixed_cse = calculate_cse_from_harmonic_features(mixed_features, m=2, r=0.1)
    print(f"混合信号CSE: {mixed_cse:.6f}")
    
    print(f"\n预期结果:")
    print(f"规律信号CSE应该接近0: {regular_cse < 1.0}")
    print(f"随机信号CSE应该 >> 1: {random_cse > 2.0}")
    print(f"混合信号CSE应该介于两者之间: {0.5 < mixed_cse < random_cse}")

if __name__ == "__main__":
    quick_test_cse()






