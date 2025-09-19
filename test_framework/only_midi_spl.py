"""
基础VRP提取 - 只提取MIDI和dB
专注于核心功能：
1. F0 extraction (基频提取)
2. SPL extraction with fixed window (固定窗口SPL提取)
输出：MIDI, dB 两列数据
"""

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fftpack import fft, ifft
import pandas as pd

class BasicVRP:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.window_size = 1024  # 约23ms at 44.1kHz
        self.hop_size = 512      # 50% overlap
        self.spl_reference = 20e-6  # 标准声压参考值，可调节
        
        # FonaDyn标准轴范围 - 固定不变
        self.MIDI_MIN = 30      # FonaDyn nMinMIDI
        self.MIDI_MAX = 96      # FonaDyn nMaxMIDI  
        self.SPL_MIN = 40       # FonaDyn nMinSPL
        self.SPL_MAX = 120      # FonaDyn nMaxSPL (普通模式)
        
    def load_audio(self, file_path):
        """加载音频文件"""
        signal, sr = sf.read(file_path)
        if signal.ndim == 2:
            voice = signal[:, 0]  # 第一通道：语音
            egg = signal[:, 1]    # 第二通道：EGG
        else:
            voice = signal
            egg = None
        return voice, egg, sr
    
    def detect_segments(self, egg_signal):
        """简单的周期检测"""
        if egg_signal is None:
            # 如果没有EGG信号，使用语音信号的包络
            envelope = np.abs(egg_signal)
        else:
            envelope = np.abs(egg_signal)
        
        # 简单的峰值检测
        peaks, _ = find_peaks(envelope, distance=self.sample_rate//20)  # 最小距离50ms
        
        # 将峰值转换为周期边界
        segments = []
        for i in range(len(peaks)-1):
            start = peaks[i]
            end = peaks[i+1]
            if end - start > self.sample_rate//50:  # 最小周期20ms
                segments.append((start, end))
        
        return segments
    
    def extract_f0(self, signal_segment):
        """简单的F0提取"""
        if len(signal_segment) < 8:
            return 0.0
        
        # 预处理
        x = signal_segment - np.mean(signal_segment)
        x = x / (np.max(np.abs(x)) + 1e-8)
        
        # 自相关
        ac = np.correlate(x, x, mode='full')
        ac = ac[len(x)-1:]
        nac = ac / (ac[0] + 1e-8)
        
        # 寻找峰值
        min_period = self.sample_rate // 500  # 500Hz max
        max_period = self.sample_rate // 50   # 50Hz min
        
        if max_period >= len(nac):
            max_period = len(nac) - 1
            
        roi = nac[min_period:max_period]
        peaks, props = find_peaks(roi, height=0.1)
        
        if len(peaks) == 0:
            return 0.0
            
        # 选择最高峰值
        best_idx = np.argmax(props["peak_heights"])
        period = min_period + peaks[best_idx]
        
        f0_hz = self.sample_rate / period
        
        # 转换为MIDI
        midi = 69.0 + 12.0 * np.log2(f0_hz / 440.0)
        
        return midi
    
    def extract_spl_fixed_window(self, signal):
        """固定窗口SPL提取 - 基于old_github的正确实现"""
        # 预处理：30Hz高通滤波（与old_github一致）
        nyquist = self.sample_rate / 2
        low_cutoff = 30 / nyquist
        b, a = butter(2, low_cutoff, btype='high')  # 2阶滤波器
        signal_filtered = filtfilt(b, a, signal)
        
        spl_values = []
        midi_values = []
        
        # 创建窗函数
        window_func = np.hanning(self.window_size)
        
        for start in range(0, len(signal_filtered) - self.window_size, self.hop_size):
            # 提取窗口
            segment = signal_filtered[start:start + self.window_size]
            windowed_segment = segment * window_func
            
            # 计算SPL (使用原始segment，不是windowed_segment)
            def find_SPL(signal, reference=20e-6):
                signal = signal * 20
                rms = np.sqrt(np.mean(signal**2))
                spl = 20 * np.log10(rms / reference)
                return spl
            
            spl = find_SPL(segment)  # 使用原始segment
            # 四舍五入：-0.5到0.5之间取整到0
            spl = round(spl)
            
            # 计算F0 (MIDI) - 使用windowed_segment
            midi = self.extract_f0(windowed_segment)
            # 四舍五入：-0.5到0.5之间取整到0
            midi = round(midi)
            
            spl_values.append(spl)
            midi_values.append(midi)
        
        return midi_values, spl_values
    
    def process_audio(self, file_path):
        """处理音频文件，返回MIDI-SPL对"""
        print(f"处理音频文件: {file_path}")
        
        # 1. 加载音频
        voice, egg, sr = self.load_audio(file_path)
        print(f"音频长度: {len(voice)/sr:.2f}秒")
        
        # 2. 周期检测（可选，这里我们直接使用固定窗口）
        print("使用固定窗口方法...")
        
        # 3. 提取F0和SPL
        midi_values, spl_values = self.extract_spl_fixed_window(voice)
        
        print(f"提取到 {len(midi_values)} 个数据点")
        
        # 过滤有效数据 - 使用FonaDyn标准范围
        valid_data = [(m, s) for m, s in zip(midi_values, spl_values) 
                     if (m >= self.MIDI_MIN and m <= self.MIDI_MAX and 
                         s >= self.SPL_MIN and s <= self.SPL_MAX)]
        
        midi_valid = [m for m, s in valid_data]
        spl_valid = [s for m, s in valid_data]
        
        print(f"有效数据点: {len(valid_data)}")
        print(f"MIDI范围: {min(midi_valid):.1f} - {max(midi_valid):.1f}")
        print(f"SPL范围: {min(spl_valid):.1f} - {max(spl_valid):.1f}")
        
        return midi_valid, spl_valid

def main():
    """主函数"""
    print("=== 基础VRP提取 ===")
    
    # 创建VRP处理器
    vrp = BasicVRP()
    
    # 使用old_github的正确SPL计算方法
    print("使用old_github的正确SPL计算方法（30Hz高通滤波，信号放大20倍，参考值20e-6）...")
    
    # 处理测试音频
    audio_file = "../audio/test_Voice_EGG.wav"
    midi_values, spl_values = vrp.process_audio(audio_file)
    
    # 保存结果
    df = pd.DataFrame({
        'MIDI': midi_values,
        'SPL': spl_values
    })
    
    output_file = "result/midi_spl_VRP.csv"
    df.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")
    
    # 显示统计信息
    print(f"\n=== 结果统计 ===")
    print(f"数据点数量: {len(df)}")
    print(f"MIDI平均值: {df['MIDI'].mean():.1f}")
    print(f"SPL平均值: {df['SPL'].mean():.1f}")
    
    print(f"\nMIDI分布:")
    print(f"30-50: {len(df[(df['MIDI'] >= 30) & (df['MIDI'] < 50)])} 个")
    print(f"50-70: {len(df[(df['MIDI'] >= 50) & (df['MIDI'] < 70)])} 个")
    print(f"70+: {len(df[df['MIDI'] >= 70])} 个")
    
    print(f"\nSPL分布:")
    print(f"40-60: {len(df[(df['SPL'] >= 40) & (df['SPL'] < 60)])} 个")
    print(f"60-80: {len(df[(df['SPL'] >= 60) & (df['SPL'] < 80)])} 个")
    print(f"80+: {len(df[df['SPL'] >= 80])} 个")

if __name__ == "__main__":
    main()
