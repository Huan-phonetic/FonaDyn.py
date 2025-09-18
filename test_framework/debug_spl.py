"""
调试SPL计算，分析信号特征
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def analyze_signal():
    """分析信号特征"""
    # 加载音频
    signal, sr = sf.read("../audio/test_Voice_EGG.wav")
    voice = signal[:, 0]  # 语音信号
    
    print(f"=== 信号分析 ===")
    print(f"采样率: {sr} Hz")
    print(f"信号长度: {len(voice)} 样本 ({len(voice)/sr:.2f}秒)")
    print(f"信号范围: {voice.min():.6f} - {voice.max():.6f}")
    print(f"信号RMS: {np.sqrt(np.mean(voice**2)):.6f}")
    print(f"信号峰值: {np.max(np.abs(voice)):.6f}")
    
    # 分析不同窗口大小的RMS
    window_sizes = [512, 1024, 2048]
    hop_size = 512
    
    for window_size in window_sizes:
        rms_values = []
        for start in range(0, len(voice) - window_size, hop_size):
            window = voice[start:start + window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)
        
        rms_values = np.array(rms_values)
        print(f"\n窗口大小 {window_size} ({window_size/sr*1000:.1f}ms):")
        print(f"  RMS范围: {rms_values.min():.6f} - {rms_values.max():.6f}")
        print(f"  RMS平均值: {rms_values.mean():.6f}")
        print(f"  RMS中位数: {np.median(rms_values):.6f}")
    
    # 测试不同的SPL参考值
    print(f"\n=== SPL参考值测试 ===")
    test_rms = np.sqrt(np.mean(voice**2))
    references = [20e-6, 20e-6/10, 20e-6/100, 20e-6/1000]
    
    for ref in references:
        spl = 20 * np.log10(test_rms / ref)
        print(f"参考值 {ref:.2e}: SPL = {spl:.1f} dB")
    
    # 检查是否需要预处理
    print(f"\n=== 预处理分析 ===")
    
    # 高通滤波
    from scipy.signal import butter, filtfilt
    nyquist = sr / 2
    low_cutoff = 80 / nyquist
    b, a = butter(4, low_cutoff, btype='high')
    voice_filtered = filtfilt(b, a, voice)
    
    print(f"滤波后RMS: {np.sqrt(np.mean(voice_filtered**2)):.6f}")
    
    # 归一化
    voice_norm = voice_filtered / np.max(np.abs(voice_filtered))
    print(f"归一化后RMS: {np.sqrt(np.mean(voice_norm**2)):.6f}")
    
    return voice, voice_filtered, voice_norm

if __name__ == "__main__":
    analyze_signal()
