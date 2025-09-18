import unittest
import numpy as np
import librosa
import os
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessing.voice_preprocessor import preprocess_voice_signal
from src.preprocessing.egg_preprocessor import preprocess_egg_signal
from src.metrics.voice_metrics import find_f0, find_spl
from src.metrics.egg_metrics import find_qci, find_deggmax

import numpy as np
import pandas as pd
import os
import soundfile as sf
from src.preprocessing.cycle_picker import peak_cycles

def export_midi_spl(signal, sample_rate, segments, save_path="midi_spl_pairs.csv"):
    """
    简化版: 只导出 (MIDI, SPL) 对到 CSV 文件.
    signal: np.ndarray shape (2, N): [voice, egg]
    sample_rate: int
    segments: list of (start, end) indices (周期切分结果)
    save_path: 输出 CSV 路径
    """
    from src.metrics.voice_metrics import find_f0, find_spl

    voice_raw  = np.asarray(signal[0, :])
    # 使用正确的预处理信号进行F0计算
    from src.preprocessing.voice_preprocessor import preprocess_voice_signal
    voice_proc = preprocess_voice_signal(voice_raw, sample_rate)

    f0_midi_vals = []
    spl_vals     = []

    for start, end in segments:
        v_seg_proc = voice_proc[start:end]
        v_seg_raw  = voice_raw[start:end]

        # F0 (MIDI)
        f0_m, _ = find_f0(v_seg_proc, sample_rate, midi=True)
        f0_midi_vals.append(f0_m)

        # SPL (原始幅度)
        spl_vals.append(find_spl(v_seg_raw))

    # 保存成 CSV
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df = pd.DataFrame({"MIDI": f0_midi_vals, "SPL": spl_vals})
    df.to_csv(save_path, index=False)
    print(f"[debug] saved (MIDI, SPL) pairs -> {save_path} (rows={len(df)})")
    return df


if __name__ == '__main__':
    test_file = 'audio/test_Voice_EGG.wav'
    signal, samplerate = sf.read(test_file)  # (N, 2)

    # 确保是 (2, N) 格式
    if signal.ndim == 1:
        signal = np.expand_dims(signal, axis=1)   # (N, 1)
    signal = signal.T   # (channels, N)
    egg_signal = signal[1, :]   # (N, )
    segments, _ = peak_cycles(egg_signal, samplerate)
    df = export_midi_spl(signal, samplerate, segments, "test_midi_spl.csv")
    print(df.head())
