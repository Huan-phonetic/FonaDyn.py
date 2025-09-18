import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import soundfile as sf


def highpass_filter(x, sr, cutoff=50, order=3):
    b, a = butter(order, cutoff/(0.5*sr), btype="high")
    return filtfilt(b, a, x)


def _lowpass(x, sr, cutoff_hz, order=3):
    Wn = min(0.99, cutoff_hz/(0.5*sr))
    if Wn <= 0:
        return x
    b, a = butter(order, Wn, btype="low")
    return filtfilt(b, a, x)


def peak_cycles(signal, samplerate,
                    min_frequency=50, max_frequency=600,
                    smooth_hz=200,
                    plot=False):
    """
    基于 dEGG 正峰 (最大闭合速度) 的 EGG 周期检测
    返回:
      segments: [(start_i, start_{i+1}), ...]  # 采样点
      time_segments: [(t_i, t_{i+1}), ...]     # 秒
    """
    # 高通去漂移
    signal = highpass_filter(signal, samplerate, cutoff=30)
    signal = signal - np.mean(signal)

    # dEGG
    degg = np.gradient(signal)
    degg = _lowpass(degg, samplerate, smooth_hz)

    # 找正峰（不加阈值）
    min_period = max(1, int(samplerate / max_frequency))
    peaks, _ = find_peaks(degg, distance=min_period)

    # 构造周期区间
    segments = [(peaks[i], peaks[i+1]) for i in range(len(peaks)-1)]
    time_segments = [(s/samplerate, e/samplerate) for s, e in segments]

    # 可选绘图
    if plot:
        t = np.arange(len(signal)) / samplerate
        plt.figure(figsize=(14, 5))
        plt.plot(t, signal, label="EGG")
        scale = np.max(np.abs(degg)) + 1e-12
        plt.plot(t, degg/scale, label="dEGG (scaled)")
        for s in peaks:
            ts = s / samplerate
            plt.axvline(ts, color="red", linestyle="--", alpha=0.6)
            plt.plot(ts, signal[s], "ro", markersize=4)
        plt.legend()
        plt.title("Cycle detection by dEGG positive peaks (full signal)")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    return segments, time_segments


def validate_segments_with_audio(egg_segments, samplerate, voice_signal,
                                 midi_min=30, midi_max=100,
                                 ac_threshold=0.25) -> list:
    """
    用 EGG 分段作为候选：
    1) 频率范围校验（30–100 MIDI）
    2) 在对应 voice 片段做自相关，验证为周期（相关峰 > ac_threshold）
    返回通过校验的 segments 列表（采样点）
    """
    validated = []
    # MIDI -> Hz: f = 440 * 2^((m-69)/12)
    def midi_to_hz(m):
        return 440.0 * (2.0 ** ((m - 69.0) / 12.0))
    fmin = midi_to_hz(midi_min)
    fmax = midi_to_hz(midi_max)

    min_period = int(samplerate / max(fmin, 1e-6))
    max_period = int(samplerate / max(fmax, 1e-6))
    if min_period < 1:
        min_period = 1
    if max_period < 1:
        max_period = 1

    for start, end in egg_segments:
        start_i = int(max(0, start))
        end_i = int(min(len(voice_signal), end))
        if end_i - start_i < 3:
            continue
        period = end_i - start_i
        # 周期范围：min_period 对应 fmin，max_period 对应 fmax
        if not (max_period <= period <= min_period):
            continue

        seg = voice_signal[start_i:end_i]
        seg = seg - np.mean(seg)
        if np.max(np.abs(seg)) < 1e-8:
            continue
        seg = seg / np.max(np.abs(seg))
        # 自相关
        ac = np.correlate(seg, seg, mode='full')
        ac = ac[ac.size//2:]
        # 只看 30–100 MIDI 对应的滞后
        lo = max(1, max_period)
        hi = min(len(ac)-1, min_period)
        if hi <= lo:
            continue
        ac_slice = ac[lo:hi+1]
        if ac_slice.size == 0:
            continue
        peak = np.max(ac_slice)
        norm = ac[0] if ac[0] != 0 else 1.0
        score = peak / norm
        if score >= ac_threshold:
            validated.append((start_i, end_i))

    return validated


if __name__ == "__main__":
    # ==== 测试语音信号 ====
    audio_file = 'audio/test_Voice_EGG.wav'
    sig, sr = sf.read(audio_file)
    if sig.ndim > 1:
        mic_signal = sig[:, 1]  # 第二通道
    else:
        mic_signal = sig
    # 保留前5s
    mic_signal = mic_signal[:sr*5]
    segs_v, timesegs_v, peaks_v, degg_v = peak_cycles(mic_signal, sr, plot=True)
    print(f"Detected cycles (voice): {len(segs_v)}")
    print("First 5 time segments (voice):", timesegs_v[:5])
