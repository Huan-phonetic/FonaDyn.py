import numpy as np
import soundfile as sf

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
def debug_deggmax(egg_segment):
    print(f"Input segment length: {len(egg_segment)}")
    print(f"Input segment range: {np.min(egg_segment)} to {np.max(egg_segment)}")
    
    if len(egg_segment) < 4:
        print("Segment too short")
        return 0.0
    
    try:
        # Calculate peak-to-peak amplitude
        peak2peak = np.max(egg_segment) - np.min(egg_segment)
        print(f"Peak-to-peak: {peak2peak}")
        
        if peak2peak == 0:
            print("Peak-to-peak is 0")
            return 0.0
        
        # Calculate delta (maximum derivative)
        delta = np.max(np.abs(np.diff(egg_segment)))
        print(f"delta: {delta}")
        
        # Simplified dEGGmax calculation
        dEGGmax = delta / peak2peak
        print(f"dEGGmax before log: {dEGGmax}")
        
        # Apply log transformation with minimum value
        dEGGmax_log = np.log(np.maximum(1.0, dEGGmax))
        print(f"dEGGmax after log: {dEGGmax_log}")
        
        return dEGGmax_log
    except Exception as e:
        print(f"Error: {e}")
        return 0.0

# 测试几个窗口
window_size = 1024
for i in range(3):
    start = i * 1024
    end = start + window_size
    segment = egg_proc[start:end]
    print(f"\n=== Window {i} ===")
    result = debug_deggmax(segment)
    print(f"Final result: {result}")
