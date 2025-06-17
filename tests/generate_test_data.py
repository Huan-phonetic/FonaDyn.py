import numpy as np
import soundfile as sf
import os

def generate_test_audio():
    # 创建测试数据目录
    test_data_dir = 'tests/test_data'
    os.makedirs(test_data_dir, exist_ok=True)
    
    # 生成测试音频
    sr = 44100
    duration = 1.0  # 1秒
    t = np.linspace(0, duration, int(sr * duration))
    
    # 生成语音信号（220Hz的正弦波）
    f0 = 220
    voice_signal = np.sin(2 * np.pi * f0 * t)
    
    # 生成EGG信号
    egg_signal = np.sin(2 * np.pi * f0 * t) * 0.5
    
    # 合并为双通道信号
    signal = np.vstack((voice_signal, egg_signal))
    
    # 保存为WAV文件
    output_path = os.path.join(test_data_dir, 'test_audio.wav')
    sf.write(output_path, signal.T, sr)
    print(f"Generated test audio file at: {output_path}")

if __name__ == '__main__':
    generate_test_audio() 