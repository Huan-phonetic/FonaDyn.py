import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# 加载测试数据
data, sample_rate = sf.read('../audio/test_Voice_EGG.wav')
voice = data[:, 0]  # 第一个通道
egg = data[:, 1]    # 第二个通道

# 预处理EGG
def preprocess_egg(egg_signal, sample_rate):
    from scipy.signal import butter, sosfilt
    sos = butter(2, 30, 'hp', fs=sample_rate, output='sos')
    return sosfilt(sos, egg_signal)

egg_proc = preprocess_egg(egg, sample_rate)

# 测试dEGGmax计算
def debug_deggmax_detailed(egg_segment, sample_rate):
    print(f"=== dEGGmax详细调试 ===")
    print(f"输入segment长度: {len(egg_segment)}")
    print(f"输入segment范围: {np.min(egg_segment):.6f} 到 {np.max(egg_segment):.6f}")
    
    if len(egg_segment) < 4:
        print("Segment太短")
        return 0.0
    
    try:
        # 1. 计算peak-to-peak (min - max as in FonaDyn)
        peak2peak = np.min(egg_segment) - np.max(egg_segment)
        print(f"1. Peak-to-peak (min-max): {peak2peak:.6f}")
        
        if peak2peak == 0:
            print("Peak-to-peak为0")
            return 0.0
        
        # 2. 计算ticks (sample rate)
        ticks = sample_rate
        print(f"2. Ticks (sample rate): {ticks}")
        
        # 3. 计算EGG信号的导数
        differentiated_egg = np.diff(egg_segment)
        print(f"3. 导数信号长度: {len(differentiated_egg)}")
        print(f"   导数信号范围: {np.min(differentiated_egg):.6f} 到 {np.max(differentiated_egg):.6f}")
        
        # 4. 找到最大导数
        delta = np.max(np.abs(differentiated_egg))
        print(f"4. 最大导数 (delta): {delta:.6f}")
        
        # 5. 计算amplitude scale factor
        sin_term = np.sin(2 * np.pi / ticks)
        print(f"5. sin(2π/ticks): {sin_term:.6f}")
        
        ampScale = 1.0 / (peak2peak * (-0.5) * sin_term)
        print(f"6. ampScale: {ampScale:.6f}")
        
        # 6. 计算dEGGmax
        dEGGmax = delta * ampScale
        print(f"7. dEGGmax (before log): {dEGGmax:.6f}")
        
        # 7. 应用log10变换
        dEGGmax_log = np.log10(max(1.0, dEGGmax))
        print(f"8. dEGGmax (after log10): {dEGGmax_log:.6f}")
        
        return dEGGmax_log
    except Exception as e:
        print(f"错误: {e}")
        return 0.0

# 测试几个窗口
window_size = 1024
for i in range(3):
    start = i * 1024
    end = start + window_size
    segment = egg_proc[start:end]
    print(f"\n=== 窗口 {i} ===")
    result = debug_deggmax_detailed(segment, sample_rate)
    print(f"最终结果: {result}")
