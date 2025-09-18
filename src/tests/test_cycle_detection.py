import numpy as np
import soundfile as sf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.cycle_picker import peak_cycles, validate_segments_with_audio
from src.preprocessing.voice_preprocessor import preprocess_voice_signal
from src.preprocessing.egg_preprocessor import preprocess_egg_signal
from src.metrics.voice_metrics import find_spl


def load_reference_cycles(reference_file):
    """Load reference signal (ch 2) and its cycle boundaries (peaks)."""
    ref_signal_stereo, ref_samplerate = sf.read(reference_file)
    ref_signal = ref_signal_stereo[:, 1]
    peaks, _ = find_peaks(ref_signal, height=0.1, distance=32, prominence=0.05)
    return peaks, ref_samplerate, ref_signal


def compare_cycles(detected_cycles, reference_cycles, tolerance=10):
    """Compare detected cycles with reference cycles"""
    correct = 0
    total = len(reference_cycles)
    if len(detected_cycles) == 0 or total == 0:
        return 0.0
    for ref_cycle in reference_cycles:
        distances = np.abs(detected_cycles - ref_cycle)
        if len(distances) == 0:
            continue
        if np.min(distances) <= tolerance:
            correct += 1
    accuracy = (correct / total) * 100
    return accuracy


def plot_comparison(signal_main, signal_ref, detected_cycles, reference_cycles, samplerate, title):
    """Overlay both audio waveforms and cycle points for comparison."""
    n = min(len(signal_main), len(signal_ref))
    signal_main = signal_main[:n]
    signal_ref = signal_ref[:n]
    time = np.arange(n) / samplerate

    sig_main = signal_main / (np.max(np.abs(signal_main)) + 1e-8)
    sig_ref = signal_ref / (np.max(np.abs(signal_ref)) + 1e-8)

    detected_cycles = np.asarray(detected_cycles, dtype=int)
    reference_cycles = np.asarray(reference_cycles, dtype=int)
    detected_cycles = detected_cycles[(detected_cycles >= 0) & (detected_cycles < n)]
    reference_cycles = reference_cycles[(reference_cycles >= 0) & (reference_cycles < n)]

    plt.figure(figsize=(15, 5))
    plt.plot(time, sig_main, label='EGG (main)', color='C0', linewidth=1.0)
    plt.plot(time, sig_ref, label='Peak follower ref', color='C1', linewidth=0.8, alpha=0.7)
    plt.plot(detected_cycles / samplerate, sig_main[detected_cycles], 'ro', markersize=4, label='Zero-crossing cycles')
    plt.plot(reference_cycles / samplerate, sig_ref[reference_cycles], 'gx', markersize=5, label='Peak-follower cycles')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (normalized)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def test_cycle_metrics(signal, sample_rate, save_csv_path=None, midi_min=30, midi_max=100, unique=False):
    """Print and optionally save all raw MIDI and SPL pairs before flooring.
       改进点：
       - NACF + find_peaks 取自相关峰
       - 抛物线插值提升精度
       - 可选 unique 模式：对 MIDI 和 SPL round 后去重
    """
    voice = signal[0, :]
    egg = signal[1, :]

    voice = preprocess_voice_signal(voice, sample_rate)
    egg = preprocess_egg_signal(egg, sample_rate)

    egg_segments, _ = peak_cycles(egg, sample_rate)
    segments = validate_segments_with_audio(
        egg_segments=egg_segments,
        samplerate=sample_rate,
        voice_signal=voice,
        midi_min=midi_min,
        midi_max=midi_max,
        ac_threshold=0.25,
    )

    def midi_to_hz(m): 
        return 440.0 * (2.0 ** ((m - 69.0) / 12.0))

    fmin, fmax = midi_to_hz(midi_min), midi_to_hz(midi_max)

    f0_per_cycle_midi, spl_per_cycle = [], []

    for start, end in segments:
        voice_segment = voice[start:end]
        x = voice_segment.astype(float)
        x -= np.mean(x)
        peak_abs = np.max(np.abs(x))
        if peak_abs == 0 or len(x) < 8:
            f0_per_cycle_midi.append(0.0)
            spl_per_cycle.append(find_spl(voice_segment))
            continue
        x /= peak_abs

        ac = np.correlate(x, x, mode='full')
        ac = ac[len(x)-1:]
        nac = ac / (ac[0] + 1e-8)

        lo = max(1, int(sample_rate / max(fmax, 1e-6)))
        hi = min(int(sample_rate / max(fmin, 1e-6)), len(nac) - 3)
        if hi <= lo:
            f0_per_cycle_midi.append(0.0)
            spl_per_cycle.append(find_spl(voice_segment))
            continue

        roi = nac[lo:hi+1]
        peaks, _ = find_peaks(roi)
        if len(peaks) == 0:
            k = int(np.argmax(roi))
        else:
            k = int(peaks[np.argmax(roi[peaks])])
        i = lo + k

        if 1 <= i < len(nac) - 1:
            a, b, c = nac[i-1], nac[i], nac[i+1]
            denom = (a - 2*b + c)
            delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (a - c) / denom
        else:
            delta = 0.0
        lag = i + delta

        if lag <= 0:
            f0_hz = 0.0
        else:
            f0_hz = sample_rate / lag

        if not (fmin <= f0_hz <= fmax) and f0_hz > 0:
            if 2*fmin <= f0_hz <= 2*fmax:
                f0_hz *= 0.5
            elif 0.5*fmin <= f0_hz <= 0.5*fmax:
                f0_hz *= 2.0

        if f0_hz <= 0 or not (fmin <= f0_hz <= fmax):
            midi = 0.0
        else:
            midi = 69.0 + 12.0 * np.log2(f0_hz / 440.0)

        f0_per_cycle_midi.append(midi)
        spl_per_cycle.append(find_spl(voice_segment))

    df = pd.DataFrame({"MIDI": f0_per_cycle_midi, "SPL": spl_per_cycle})

    # 如果要求 unique，则 round 后去重
    if unique:
        df["MIDI"] = df["MIDI"].round(0)
        df["SPL"] = df["SPL"].round(0)
        df = df.drop_duplicates().reset_index(drop=True)

    # 打印
    print("========== MIDI-SPL pairs ==========")
    for _, row in df.iterrows():
        print(f"MIDI: {row['MIDI']:.2f}, SPL: {row['SPL']:.2f}")
    print("Total pairs:", len(df))

    # 保存 CSV
    if save_csv_path:
        import os
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        df.to_csv(save_csv_path, index=False)
        print(f"Saved MIDI-SPL pairs to {save_csv_path}")

    return df



def main():
    # 加载语音+EGG信号（必须是双声道）
    test_file = 'audio/test_Voice_EGG.wav'
    signal_stereo, samplerate = sf.read(test_file)

    # 确保是双声道
    if signal_stereo.ndim == 1 or signal_stereo.shape[1] < 2:
        raise ValueError("测试文件必须是双声道 (voice, EGG)")

    # main 信号（EGG 通道）
    signal = signal_stereo[:, 1]

    # 加载参考信号
    reference_file = 'audio/test_CycleDetection_peak_follower.wav'
    ref_stereo, ref_samplerate = sf.read(reference_file)
    ref_signal = ref_stereo[:, 1]

    if samplerate != ref_samplerate:
        print(f"Warning: Sampling rates don't match. Test: {samplerate}, Reference: {ref_samplerate}")

    # 获取参考周期
    reference_cycles, _, _ = load_reference_cycles(reference_file)

    # 零交叉检测
    segments_zc, _ = peak_cycles(signal, samplerate)
    starts_zc = np.array([int(s) for s, _ in segments_zc], dtype=int)

    # 准确率对比
    zc_accuracy = compare_cycles(starts_zc, reference_cycles)
    print(f"Zero-crossing accuracy: {zc_accuracy:.2f}%")

    # 绘制对比图
    # plot_comparison(signal, ref_signal, starts_zc, reference_cycles, samplerate, 'Methods Comparison')

    # 🚀 在这里跑 MIDI–SPL pair 分布
    print("\nRunning cycle-based MIDI–SPL analysis...")
    test_cycle_metrics(signal_stereo.T, samplerate, save_csv_path="results/midi_spl_unique.csv", unique=True)


if __name__ == '__main__':
    main()
