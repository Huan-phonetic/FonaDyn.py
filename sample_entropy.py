"""
Sample Entropy 核心实现
用于集成到主程序的简化版本
"""

import math
from typing import List, Tuple, Optional


class SampleEntropy:
    """
    Sample Entropy计算器
    支持朴素算法和优化算法两种实现方式
    """
    
    def __init__(self, m: int = 2, r: float = 0.2):
        """
        初始化Sample Entropy计算器
        
        Args:
            m: 模式长度，默认为2
            r: 相似性阈值，默认为0.2
        """
        self.m = m
        self.r = r
        self.prev_data = []
        self.prev_A = 0
        self.prev_B = 0
    
    def naive_calculate(self, data: List[float]) -> float:
        """
        使用朴素算法计算Sample Entropy
        
        Args:
            data: 输入数据序列
            
        Returns:
            Sample Entropy值
        """
        n = len(data)
        A = 0
        B = 0
        
        for i in range(n - self.m):
            for j in range(i + 1, n - self.m):
                match = True
                for k in range(self.m):
                    if abs(data[i + k] - data[j + k]) > self.r:
                        match = False
                        break
                
                if match:
                    B += 1
                    if abs(data[i + self.m] - data[j + self.m]) <= self.r:
                        A += 1
        
        if A == 0 or B == 0:
            return 0.0
        
        return -math.log(A / B)
    
    def optimized_calculate(self, data: List[float]) -> float:
        """
        使用优化算法计算Sample Entropy
        适用于滑动窗口场景
        
        Args:
            data: 输入数据序列
            
        Returns:
            Sample Entropy值
        """
        n = len(data)
        
        # 如果是第一次计算，使用朴素算法
        if len(self.prev_data) == 0:
            A, B = 0, 0
            for i in range(n - self.m):
                for j in range(i + 1, n - self.m):
                    match = True
                    for k in range(self.m):
                        if abs(data[i + k] - data[j + k]) > self.r:
                            match = False
                            break
                    
                    if match:
                        B += 1
                        if abs(data[i + self.m] - data[j + self.m]) <= self.r:
                            A += 1
            
            self.prev_A = A
            self.prev_B = B
            self.prev_data = data.copy()
            
            if A == 0 or B == 0:
                return 0.0
            return -math.log(A / B)
        
        # 优化算法：增量更新
        A = self.prev_A
        B = self.prev_B
        
        # 移除第一个元素的影响
        for j in range(1, n - self.m):
            match = True
            for k in range(self.m):
                if abs(self.prev_data[k] - self.prev_data[j + k]) > self.r:
                    match = False
                    break
            
            if match:
                B -= 1
                if abs(self.prev_data[self.m] - self.prev_data[j + self.m]) <= self.r:
                    A -= 1
        
        # 添加最后一个元素的影响
        for j in range(n - self.m):
            match = True
            for k in range(self.m):
                if abs(data[n - self.m + k] - data[j + k]) > self.r:
                    match = False
                    break
            
            if match:
                B += 1
                if abs(data[n - 1] - data[j + self.m]) <= self.r:
                    A += 1
        
        # 更新状态
        self.prev_A = A
        self.prev_B = B
        self.prev_data = data.copy()
        
        if A <= 0 or B <= 0:
            return 0.0
        
        return -math.log(A / B)
    
    def reset(self):
        """重置计算器状态"""
        self.prev_data = []
        self.prev_A = 0
        self.prev_B = 0
    
    def set_parameters(self, m: int, r: float):
        """
        设置计算参数
        
        Args:
            m: 模式长度
            r: 相似性阈值
        """
        self.m = m
        self.r = r
        self.reset()


def calculate_sample_entropy(data: List[float], m: int = 2, r: float = 0.2, 
                          optimized: bool = False) -> float:
    """
    便捷函数：计算Sample Entropy
    
    Args:
        data: 输入数据序列
        m: 模式长度，默认为2
        r: 相似性阈值，默认为0.2
        optimized: 是否使用优化算法，默认为False
        
    Returns:
        Sample Entropy值
    """
    calculator = SampleEntropy(m, r)
    
    if optimized:
        return calculator.optimized_calculate(data)
    else:
        return calculator.naive_calculate(data)


# 使用示例
if __name__ == "__main__":
    import numpy as np
    
    # 生成测试数据
    t = np.linspace(0, 4*np.pi, 50)
    test_data = (np.sin(t) + 0.1 * np.sin(10*t)).tolist()
    
    print("Sample Entropy 计算示例:")
    print(f"数据长度: {len(test_data)}")
    print(f"数据范围: [{min(test_data):.3f}, {max(test_data):.3f}]")
    
    # 朴素算法
    naive_result = calculate_sample_entropy(test_data, m=2, r=0.2, optimized=False)
    print(f"朴素算法结果: {naive_result:.6f}")
    
    # 优化算法
    optimized_result = calculate_sample_entropy(test_data, m=2, r=0.2, optimized=True)
    print(f"优化算法结果: {optimized_result:.6f}")
    
    # 滑动窗口示例
    print("\n滑动窗口示例:")
    calculator = SampleEntropy(m=2, r=0.2)
    
    window_size = 20
    for i in range(5):
        window_data = test_data[i:i + window_size]
        entropy = calculator.optimized_calculate(window_data)
        print(f"窗口 {i+1}: entropy = {entropy:.6f}")

