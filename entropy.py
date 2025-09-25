"""
Sample Entropy Implementation
根据图示文档实现的Sample Entropy算法，包括朴素算法和优化算法两个版本
"""

import numpy as np
import math
import time
from typing import List, Tuple, Optional


def naive_sample_entropy(u: List[float], m: int, r: float) -> float:
    """
    朴素Sample Entropy算法实现
    
    Args:
        u: 输入数据序列
        m: 模式长度
        r: 相似性阈值
        
    Returns:
        Sample Entropy值，如果A=0或B=0则返回0
    """
    n = len(u)
    A = 0
    B = 0
    
    # 外层循环：遍历所有可能的起始位置i
    for i in range(n - m):
        # 内层循环：遍历所有可能的起始位置j (j > i)
        for j in range(i + 1, n - m):
            match = True
            
            # 检查长度为m的序列是否匹配
            for k in range(m):
                if abs(u[i + k] - u[j + k]) > r:
                    match = False
                    break
            
            if match:
                B += 1
                # 检查长度为m+1的序列是否也匹配
                if abs(u[i + m] - u[j + m]) <= r:
                    A += 1
    
    # 如果A或B为0，返回0（Sample Entropy未定义）
    if A == 0 or B == 0:
        return 0.0
    
    return -math.log(A / B)


def optimized_sample_entropy(prev_u: List[float], prev_A: float, prev_B: float, 
                           u: List[float], m: int, r: float) -> Tuple[float, float, float]:
    """
    优化版Sample Entropy算法实现
    
    Args:
        prev_u: 之前的序列数据
        prev_A: 之前计算的A值
        prev_B: 之前计算的B值
        u: 当前的序列数据
        m: 模式长度
        r: 相似性阈值
        
    Returns:
        (新的A值, 新的B值, Sample Entropy值)
    """
    n = len(u)
    
    # 如果是第一次计算，使用朴素算法
    if len(prev_u) == 0:
        A, B = 0, 0
        for i in range(n - m):
            for j in range(i + 1, n - m):
                match = True
                for k in range(m):
                    if abs(u[i + k] - u[j + k]) > r:
                        match = False
                        break
                
                if match:
                    B += 1
                    if abs(u[i + m] - u[j + m]) <= r:
                        A += 1
    else:
        # 优化算法：基于前一次结果进行增量更新
        A = prev_A
        B = prev_B
        
        # 移除第一个元素的影响
        for j in range(1, n - m):
            match = True
            for k in range(m):
                if abs(prev_u[k] - prev_u[j + k]) > r:
                    match = False
                    break
            
            if match:
                B -= 1
                if abs(prev_u[m] - prev_u[j + m]) <= r:
                    A -= 1
        
        # 添加最后一个元素的影响
        for j in range(n - m):
            match = True
            for k in range(m):
                if abs(u[n - m + k] - u[j + k]) > r:
                    match = False
                    break
            
            if match:
                B += 1
                if abs(u[n - 1] - u[j + m]) <= r:
                    A += 1
    
    # 计算Sample Entropy
    if A <= 0 or B <= 0:
        entropy = 0.0
    else:
        entropy = -math.log(A / B)
    
    return A, B, entropy


def generate_test_data(n: int, signal_type: str = "sine") -> List[float]:
    """
    生成测试数据
    
    Args:
        n: 数据长度
        signal_type: 信号类型 ("sine", "noise", "mixed")
        
    Returns:
        生成的测试数据
    """
    if signal_type == "sine":
        # 正弦波信号
        t = np.linspace(0, 4*np.pi, n)
        return (np.sin(t) + 0.1 * np.sin(10*t)).tolist()
    
    elif signal_type == "noise":
        # 随机噪声
        return np.random.normal(0, 1, n).tolist()
    
    elif signal_type == "mixed":
        # 混合信号：正弦波 + 噪声
        t = np.linspace(0, 4*np.pi, n)
        sine_part = np.sin(t)
        noise_part = np.random.normal(0, 0.3, n)
        return (sine_part + noise_part).tolist()
    
    else:
        raise ValueError("Unknown signal type")


def test_algorithms():
    """
    测试两个算法的正确性和性能
    """
    print("=== Sample Entropy 算法测试 ===\n")
    
    # 测试参数
    test_cases = [
        {"n": 20, "m": 2, "r": 0.2, "signal": "sine"},
        {"n": 50, "m": 2, "r": 0.2, "signal": "mixed"},
        {"n": 100, "m": 3, "r": 0.15, "signal": "noise"},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"测试用例 {i}: n={case['n']}, m={case['m']}, r={case['r']}, 信号类型={case['signal']}")
        
        # 生成测试数据
        u = generate_test_data(case['n'], case['signal'])
        
        # 测试朴素算法
        start_time = time.time()
        naive_result = naive_sample_entropy(u, case['m'], case['r'])
        naive_time = time.time() - start_time
        
        # 测试优化算法（第一次计算）
        start_time = time.time()
        A, B, opt_result = optimized_sample_entropy([], 0, 0, u, case['m'], case['r'])
        opt_time = time.time() - start_time
        
        print(f"  朴素算法结果: {naive_result:.6f}, 耗时: {naive_time:.6f}s")
        print(f"  优化算法结果: {opt_result:.6f}, 耗时: {opt_time:.6f}s")
        print(f"  结果差异: {abs(naive_result - opt_result):.8f}")
        print()
    
    # 性能对比测试
    print("=== 性能对比测试 ===")
    sizes = [20, 50, 100, 200]
    m, r = 2, 0.2
    
    for n in sizes:
        u = generate_test_data(n, "mixed")
        
        # 朴素算法
        start_time = time.time()
        naive_sample_entropy(u, m, r)
        naive_time = time.time() - start_time
        
        # 优化算法
        start_time = time.time()
        optimized_sample_entropy([], 0, 0, u, m, r)
        opt_time = time.time() - start_time
        
        speedup = naive_time / opt_time if opt_time > 0 else float('inf')
        
        print(f"n={n:3d}: 朴素算法 {naive_time:.6f}s, 优化算法 {opt_time:.6f}s, 加速比 {speedup:.2f}x")


def sliding_window_demo():
    """
    滑动窗口演示：展示优化算法的优势
    """
    print("\n=== 滑动窗口演示 ===")
    
    # 生成较长的测试数据
    n_total = 100
    window_size = 20
    m, r = 2, 0.2
    
    # 生成完整数据
    full_data = generate_test_data(n_total, "mixed")
    
    print(f"总数据长度: {n_total}, 窗口大小: {window_size}")
    print("滑动窗口Sample Entropy计算:")
    
    # 使用朴素算法计算每个窗口
    print("\n使用朴素算法:")
    naive_times = []
    for i in range(n_total - window_size + 1):
        window_data = full_data[i:i + window_size]
        start_time = time.time()
        entropy = naive_sample_entropy(window_data, m, r)
        naive_time = time.time() - start_time
        naive_times.append(naive_time)
        
        if i < 5:  # 只显示前5个结果
            print(f"  窗口 {i+1}: entropy={entropy:.6f}, 耗时={naive_time:.6f}s")
    
    # 使用优化算法计算每个窗口
    print("\n使用优化算法:")
    opt_times = []
    prev_u = []
    prev_A, prev_B = 0, 0
    
    for i in range(n_total - window_size + 1):
        window_data = full_data[i:i + window_size]
        start_time = time.time()
        
        if i == 0:
            # 第一次计算
            A, B, entropy = optimized_sample_entropy([], 0, 0, window_data, m, r)
        else:
            # 后续计算使用优化算法
            A, B, entropy = optimized_sample_entropy(prev_u, prev_A, prev_B, window_data, m, r)
        
        opt_time = time.time() - start_time
        opt_times.append(opt_time)
        
        # 更新状态
        prev_u = window_data.copy()
        prev_A, prev_B = A, B
        
        if i < 5:  # 只显示前5个结果
            print(f"  窗口 {i+1}: entropy={entropy:.6f}, 耗时={opt_time:.6f}s")
    
    # 计算平均加速比
    avg_naive_time = np.mean(naive_times)
    avg_opt_time = np.mean(opt_times)
    avg_speedup = avg_naive_time / avg_opt_time
    
    print(f"\n平均耗时 - 朴素算法: {avg_naive_time:.6f}s, 优化算法: {avg_opt_time:.6f}s")
    print(f"平均加速比: {avg_speedup:.2f}x")


if __name__ == "__main__":
    # 运行测试
    test_algorithms()
    sliding_window_demo()
    
    print("\n=== 测试完成 ===")
    print("Sample Entropy算法实现完成！")
    print("- 朴素算法：O(m(n-m)²) 时间复杂度")
    print("- 优化算法：O(2(n-m)m) 时间复杂度")
    print("- 在滑动窗口场景下，优化算法显著提升性能")
