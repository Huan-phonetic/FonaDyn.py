"""
完整的VRP提取脚本 - 基于src中的metrics实现
包含：MIDI, SPL, Clarity, CPP, Spectrum Balance, Crest Factor, QCI, dEGGmax
"""

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, butter, filtfilt, sosfilt
from scipy.fftpack import fft, ifft
import pandas as pd
import math
import numpy.matlib

class CompleteVRP:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.window_size = 1024  # 约23ms at 44.1kHz
        self.hop_size = 512      # 50% overlap
        
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
    
    def preprocess_voice(self, signal):
        """语音信号预处理 - 30Hz高通滤波"""
        nyquist = self.sample_rate / 2
        low_cutoff = 30 / nyquist
        b, a = butter(2, low_cutoff, btype='high')
        return filtfilt(b, a, signal)
    
    def preprocess_egg(self, signal):
        """EGG信号预处理"""
        # 简单的低通滤波
        nyquist = self.sample_rate / 2
        high_cutoff = 2000 / nyquist
        b, a = butter(4, high_cutoff, btype='low')
        return filtfilt(b, a, signal)
    
    def autocorrelation(self, signal, n, k):
        """自相关计算"""
        extended_size = n + k
        fft_result = fft(signal, extended_size)
        power_spectrum = np.abs(fft_result)**2
        result = ifft(power_spectrum)
        return np.real(result)[:n]
    
    def find_f0(self, windowed_segment, threshold=0.0, midi=True, midi_min=30, midi_max=100):
        """F0提取"""
        x = np.asarray(windowed_segment, dtype=float)
        if len(x) < 8:
            return 0.0, 0.0
        
        # 预处理
        x = x - np.mean(x)
        x = x / (np.max(np.abs(x)) + 1e-8)
        
        # 自相关
        n = len(x)
        k = n // 2
        acorr = self.autocorrelation(x, n, k)
        nac = acorr / (acorr[0] + 1e-8)
        
        # 寻找峰值
        min_period = self.sample_rate // 500  # 500Hz max
        max_period = self.sample_rate // 50   # 50Hz min
        
        if max_period >= len(nac):
            max_period = len(nac) - 1
            
        roi = nac[min_period:max_period]
        peaks, props = find_peaks(roi, height=0.1, distance=max(1, len(roi)//20))
        
        if len(peaks) == 0:
            return 0.0, 0.0
            
        # 选择最高峰值
        best_idx = np.argmax(props["peak_heights"])
        period = min_period + peaks[best_idx]
        
        f0_hz = self.sample_rate / period
        
        # 转换为MIDI
        if midi:
            midi_val = 69.0 + 12.0 * np.log2(f0_hz / 440.0)
            if midi_val < midi_min or midi_val > midi_max:
                return 0.0, 0.0
            return midi_val, props["peak_heights"][best_idx]
        else:
            return f0_hz, props["peak_heights"][best_idx]
    
    def find_spl(self, signal, reference=20e-6):
        """SPL计算"""
        signal = signal * 20
        rms = np.sqrt(np.mean(signal**2))
        spl = 20 * np.log10(rms / reference)
        return spl
    
    def find_cpp(self, windowed_segment, pitch_range=[60, 880]):
        """CPP计算 - 使用FonaDyn方法"""
        if len(windowed_segment) < 1024:
            return 0.0
        
        try:
            # Apply Hanning window (FonaDyn uses Hanning)
            windowed = windowed_segment * np.hanning(len(windowed_segment))
            
            # Pad to 2048 points (FonaDyn standard)
            padded = np.zeros(2048)
            padded[:len(windowed)] = windowed
            
            # FFT
            fft = np.fft.fft(padded)
            magnitude = np.abs(fft)
            
            # Cepstrum (1024 points as in FonaDyn)
            log_magnitude = np.log(magnitude + 1e-10)
            cepstrum = np.fft.ifft(log_magnitude)
            cepstrum_magnitude = np.abs(cepstrum[:1024])  # Take first 1024 points
            
            # Convert to dB
            cepstrum_db = 20 * np.log10(cepstrum_magnitude + 1e-10)
            
            # PeakProminence: linear regression between lowBin and highBin
            # FonaDyn uses lowBin=25, highBin=367 for 60Hz-880Hz range
            lowBin = 25
            highBin = 367
            
            if highBin >= len(cepstrum_db):
                highBin = len(cepstrum_db) - 1
            
            # Linear regression
            x = np.arange(lowBin, highBin + 1)
            y = cepstrum_db[lowBin:highBin + 1]
            
            if len(x) < 2:
                return 0.0
            
            # Calculate regression line
            slope, intercept = np.polyfit(x, y, 1)
            regression_line = slope * x + intercept
            
            # Find maximum peak above regression line
            peak_height = np.max(y - regression_line)
            
            return peak_height
        except:
            return 0.0
    
    def find_spectrum_balance(self, windowed_segment):
        """频谱平衡计算"""
        # 定义滤波器截止频率
        low_cutoff = 1500  # 1.5 kHz
        high_cutoff = 2000  # 2 kHz
        
        # 设计4阶Butterworth滤波器
        sos_low = butter(4, low_cutoff, 'lp', fs=self.sample_rate, output='sos')
        sos_high = butter(4, high_cutoff, 'hp', fs=self.sample_rate, output='sos')
        
        # 滤波
        low_filtered = sosfilt(sos_low, windowed_segment)
        high_filtered = sosfilt(sos_high, windowed_segment)
        
        # 计算功率
        low_power = np.mean(low_filtered**2)
        high_power = np.mean(high_filtered**2)
        
        # 转换为dB
        low_power_db = 10 * np.log10(low_power + 1e-10)
        high_power_db = 10 * np.log10(high_power + 1e-10)
        
        # 计算差值
        sb = high_power_db - low_power_db
        return sb
    
    def find_crest_factor(self, signal):
        """峰值因子计算"""
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        return peak / rms if rms > 0 else 0
    
    def unit_egg(self, egg):
        """归一化EGG信号到单位幅度和时间"""
        egg_shifted = egg - np.min(egg)
        normalized_amplitude = egg_shifted / np.max(egg_shifted)
        normalized_time = np.linspace(0, 1, len(egg), endpoint=False)
        return normalized_time, normalized_amplitude
    
    def find_qci(self, egg_segment):
        """QCI计算 - 基于src的正确实现"""
        if len(egg_segment) < 4:
            return 0.0
        
        try:
            unit = self.unit_egg(egg_segment)
            qci = np.trapezoid(unit[1], unit[0])
            return qci
        except:
            return 0.0
    
    def find_cse(self, voice_segment):
        """Calculate CSE (Cycle-rate Sample Entropy) - very simplified version"""
        if len(voice_segment) < 50:
            return 0.0
        
        try:
            # Very simplified entropy calculation
            # Use variance as a proxy for entropy
            signal = voice_segment - np.mean(voice_segment)
            variance = np.var(signal)
            
            # Convert to a reasonable entropy-like value
            # This is a simplified approximation
            cse = np.log(variance + 1e-10) / 10.0  # Scale down
            
            return max(0.0, cse)
        except:
            return 0.0
    
    def find_deggmax(self, egg_segment):
        """Calculate dEGGmax using exact FonaDyn method - differentiated EGG signal"""
        if len(egg_segment) < 4:
            return 0.0
        
        try:
            # Calculate peak-to-peak amplitude (min - max as in FonaDyn)
            peak2peak = np.min(egg_segment) - np.max(egg_segment)
            
            if peak2peak == 0:
                return 0.0
            
            # Calculate ticks (sample rate, not window length)
            ticks = self.sample_rate
            
            # Calculate differentiated EGG signal (first derivative)
            # This is equivalent to sig - Delay1.ar(sig) in FonaDyn
            differentiated_egg = np.diff(egg_segment)
            
            # Find maximum derivative (delta) - this is the key metric
            delta = np.max(np.abs(differentiated_egg))
            
            # Calculate amplitude scale factor using exact FonaDyn formula
            # ampScale = (peak2peak*(-0.5)*sin(2pi/ticks)).reciprocal
            ampScale = 1.0 / (peak2peak * (-0.5) * np.sin(2 * np.pi / ticks))
            
            # Calculate dEGGmax using FonaDyn formula
            dEGGmax = delta * ampScale
            
            # Apply log10 transformation as in FonaDyn (iContact calculation)
            # dEGGmax.max(1.0).log10
            dEGGmax_log = np.log10(max(1.0, dEGGmax))
            
            return dEGGmax_log
        except:
            return 0.0
    
    def extract_all_metrics(self, voice, egg=None):
        """提取所有metrics"""
        # 预处理
        voice_proc = self.preprocess_voice(voice)
        if egg is not None:
            egg_proc = self.preprocess_egg(egg)
        else:
            egg_proc = None
        
        # 初始化结果列表
        results = {
            'MIDI': [],
            'SPL': [],
            'Clarity': [],
            'CPP': [],
            'SpectrumBalance': [],
            'CrestFactor': [],
            'CSE': [],
            'QCI': [],
            'dEGGmax': []
        }
        
        # 创建窗函数
        window_func = np.hanning(self.window_size)
        
        for start in range(0, len(voice_proc) - self.window_size, self.hop_size):
            # 提取窗口
            voice_segment = voice_proc[start:start + self.window_size]
            voice_windowed = voice_segment * window_func
            
            # 帧级metrics
            midi, clarity = self.find_f0(voice_windowed, threshold=0.0, midi=True)
            spl = self.find_spl(voice_segment)  # 使用原始segment
            cpp = self.find_cpp(voice_windowed)
            sb = self.find_spectrum_balance(voice_windowed)
            
            # 周期级metrics (简化处理)
            crest = self.find_crest_factor(voice_segment)
            cse = 0.0  # 暂时跳过CSE计算
            
            if egg_proc is not None:
                egg_segment = egg_proc[start:start + self.window_size]
                qci = self.find_qci(egg_segment)
                deggmax = self.find_deggmax(egg_segment)
            else:
                qci = 0.0
                deggmax = 0.0
            
            # 四舍五入
            midi = round(midi) if midi > 0 else 0
            spl = round(spl) if spl > 0 else 0
            
            # 存储结果
            results['MIDI'].append(midi)
            results['SPL'].append(spl)
            results['Clarity'].append(clarity)
            results['CPP'].append(cpp)
            results['SpectrumBalance'].append(sb)
            results['CrestFactor'].append(crest)
            results['CSE'].append(cse)
            results['QCI'].append(qci)
            results['dEGGmax'].append(deggmax)
        
        return results
    
    def remove_outliers(self, results, midi_range=(30, 80), spl_range=(40, 100)):
        """移除离群值"""
        # 创建有效数据掩码
        valid_mask = (
            (np.array(results['MIDI']) >= midi_range[0]) & 
            (np.array(results['MIDI']) <= midi_range[1]) &
            (np.array(results['SPL']) >= spl_range[0]) & 
            (np.array(results['SPL']) <= spl_range[1]) &
            (np.array(results['MIDI']) > 0) &
            (np.array(results['SPL']) > 0)
        )
        
        # 过滤数据
        filtered_results = {}
        for key in results:
            filtered_results[key] = np.array(results[key])[valid_mask]
        
        return filtered_results
    
    def process_audio(self, file_path):
        """处理音频文件"""
        print(f"处理音频文件: {file_path}")
        
        # 加载音频
        voice, egg, sr = self.load_audio(file_path)
        print(f"音频长度: {len(voice)/sr:.2f}秒")
        
        # 提取所有metrics
        print("提取所有metrics...")
        results = self.extract_all_metrics(voice, egg)
        
        print(f"提取到 {len(results['MIDI'])} 个数据点")
        
        # 移除离群值
        print("移除离群值...")
        filtered_results = self.remove_outliers(results)
        
        print(f"过滤后有效数据点: {len(filtered_results['MIDI'])}")
        print(f"MIDI范围: {filtered_results['MIDI'].min()} - {filtered_results['MIDI'].max()}")
        print(f"SPL范围: {filtered_results['SPL'].min()} - {filtered_results['SPL'].max()}")
        
        return filtered_results

def main():
    """主函数"""
    print("=== 完整VRP提取 ===")
    
    # 创建VRP处理器
    vrp = CompleteVRP()
    
    # 处理测试音频
    audio_file = "../audio/test_Voice_EGG.wav"
    results = vrp.process_audio(audio_file)
    
    # 保存结果
    df = pd.DataFrame(results)
    output_file = "complete_vrp_results.csv"
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
    
    print(f"\n其他metrics范围:")
    print(f"Clarity: {df['Clarity'].min():.3f} - {df['Clarity'].max():.3f}")
    print(f"CPP: {df['CPP'].min():.3f} - {df['CPP'].max():.3f}")
    print(f"SpectrumBalance: {df['SpectrumBalance'].min():.3f} - {df['SpectrumBalance'].max():.3f}")
    print(f"CrestFactor: {df['CrestFactor'].min():.3f} - {df['CrestFactor'].max():.3f}")

if __name__ == "__main__":
    main()
