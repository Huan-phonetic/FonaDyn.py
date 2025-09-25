# Sample Entropy 实现

根据图示文档实现的Sample Entropy算法，包括朴素算法和优化算法两个版本。

## 文件说明

- `entropy.py`: 完整的测试程序，包含详细的测试用例和性能对比
- `sample_entropy.py`: 核心实现，适合集成到主程序中

## 算法说明

### Sample Entropy 定义
Sample Entropy是用于衡量时间序列复杂度的指标，定义为：
```
SampleEntropy(m, r) = -ln(Am(r) / Bm(r))
```

其中：
- `m`: 模式长度
- `r`: 相似性阈值
- `Am(r)`: 长度为m+1的匹配序列对数量
- `Bm(r)`: 长度为m的匹配序列对数量

### 两种实现方式

1. **朴素算法** (`naive_calculate`)
   - 时间复杂度: O(m(n-m)²)
   - 每次完整计算所有序列对的匹配情况
   - 适用于单次计算

2. **优化算法** (`optimized_calculate`)
   - 时间复杂度: O(2(n-m)m)
   - 基于前一次结果进行增量更新
   - 适用于滑动窗口场景

## 使用方法

### 基本使用

```python
from sample_entropy import calculate_sample_entropy

# 计算Sample Entropy
data = [1.0, 1.1, 0.9, 1.2, 0.8, ...]  # 你的数据
entropy = calculate_sample_entropy(data, m=2, r=0.2, optimized=False)
```

### 滑动窗口使用

```python
from sample_entropy import SampleEntropy

calculator = SampleEntropy(m=2, r=0.2)

# 在滑动窗口中使用优化算法
for window_data in sliding_windows:
    entropy = calculator.optimized_calculate(window_data)
    print(f"Sample Entropy: {entropy:.6f}")
```

### 参数设置

- `m`: 模式长度，通常设置为2或3
- `r`: 相似性阈值，通常设置为数据标准差的0.1-0.25倍

## 性能特点

- 朴素算法：适合单次计算，实现简单
- 优化算法：在滑动窗口场景下性能提升显著（通常4-10倍加速）

## 测试

运行完整测试：
```bash
python entropy.py
```

运行核心功能测试：
```bash
python sample_entropy.py
```

## 集成到主程序

将 `sample_entropy.py` 中的 `SampleEntropy` 类或 `calculate_sample_entropy` 函数集成到你的主程序中即可使用。

