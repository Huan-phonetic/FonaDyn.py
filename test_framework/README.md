# 完整VRP测试框架

这个框架实现了完整的VRP（Voice Range Profile）分析，基于src中的metrics实现，包含所有重要的声学参数。

## 功能

1. **Segment Detection** - 固定窗口检测（1024样本，约23ms）
2. **F0 Extraction** - 基频提取（转换为MIDI）
3. **SPL Extraction** - 声压级提取（基于old_github的正确实现）
4. **完整Metrics** - 包含所有VRP参数

## 文件结构

```
test_framework/
├── simple_vrp.py              # 简化的VRP处理类（仅MIDI-SPL）
├── complete_vrp.py            # 完整的VRP处理类（所有metrics）
├── visualize_results.py       # 简化结果可视化
├── visualize_complete_vrp.py  # 完整结果可视化
├── convert_to_standard_vrp.py # 转换为标准VRP格式
├── debug_spl.py               # SPL调试脚本
├── complete_vrp_results.csv   # 完整结果（运行后生成）
├── standard_vrp.csv           # 标准VRP格式（运行后生成）
├── complete_vrp_result.png    # 完整可视化图片（运行后生成）
└── README.md                  # 使用说明
```

## 使用方法

### 1. 运行完整VRP提取
```bash
cd test_framework
python complete_vrp.py
```

### 2. 可视化完整结果
```bash
python visualize_complete_vrp.py
```

### 3. 转换为标准VRP格式
```bash
python convert_to_standard_vrp.py
```

## 核心参数

- **窗口大小**: 1024样本（约23ms at 44.1kHz）
- **跳跃大小**: 512样本（50%重叠）
- **预处理**: 30Hz高通滤波（与old_github一致）
- **SPL计算**: 信号放大20倍 + 20e-6参考值
- **离群值过滤**: MIDI 30-80, SPL 40-100

## 输出Metrics

### 帧级特征（固定窗口）
- **MIDI**: 基频（MIDI音符）
- **SPL**: 声压级（dB）
- **Clarity**: 清晰度
- **CPP**: 倒谱峰值突出度
- **SpectrumBalance**: 频谱平衡

### 周期级特征
- **CrestFactor**: 峰值因子
- **QCI**: 接触指数（EGG）
- **dEGGmax**: 最大EGG导数

## 特点

- **完整实现**: 包含所有VRP相关metrics
- **正确SPL**: 基于old_github的正确实现
- **离群值处理**: 自动过滤异常数据
- **标准格式**: 输出标准VRP CSV格式
- **易于调试**: 代码结构清晰，便于修改参数

## 结果示例

```
=== 完整VRP提取 ===
处理音频文件: ../audio/test_Voice_EGG.wav
音频长度: 69.98秒
提取所有metrics...
提取到 6026 个数据点
移除离群值...
过滤后有效数据点: 5936
MIDI范围: 45 - 71
SPL范围: 40 - 95

=== 结果统计 ===
数据点数量: 5936
MIDI平均值: 54.7
SPL平均值: 71.5

其他metrics范围:
Clarity: 0.111 - 0.875
CPP: 12.104 - 35.095
SpectrumBalance: -37.410 - -3.589
CrestFactor: 1.520 - 4.656
```

## 下一步

现在2维坐标框架（MIDI, SPL）已经稳定，并且包含了所有重要的VRP metrics。可以在此基础上：
- 添加更多EGG相关metrics
- 实现更精确的周期检测
- 添加数据聚合和网格化
- 集成到主VRP系统中