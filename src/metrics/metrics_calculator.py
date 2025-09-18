'''
Calculate and process voice and EGG metrics.
'''
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.voice_preprocessor import preprocess_voice_signal
from src.preprocessing.egg_preprocessor import preprocess_egg_signal
from src.preprocessing.cycle_picker import peak_cycles, validate_segments_with_audio
from src.metrics.voice_metrics import (
    find_f0, find_cpps, find_spectrum_balance, find_spl,
    find_crest_factor, period_downsampling
)
from src.metrics.egg_metrics import find_qci, find_deggmax


def _hz_to_midi(f0_hz: float) -> float:
    return 0.0 if f0_hz <= 0 else 69.0 + 12.0 * np.log2(f0_hz / 440.0)


def get_metrics(
    signal,
    sample_rate,
    n=8096,
    overlap=6072,
    *,
    debug_save_pairs_csv: str | None = None,
    debug_round_unique: bool = False
):
    """
    Calculate voice and EGG metrics.

    Args:
        signal: np.ndarray shape (2, N): [voice, egg]
        sample_rate: int
        n: frame window size
        overlap: frame overlap
        debug_save_pairs_csv: optional path to save raw cycle (MIDI, SPL) pairs
        debug_round_unique: if True, round to ints and drop duplicates before saving CSV

    Returns:
        dict with keys:
            'frequencies' (MIDI, floored to int, frame-aligned),
            'SPLs' (dB, floored to int, frame-aligned),
            'Total' (period counts per frame),
            'clarities', 'crests', 'SBs', 'CPPs', 'qdeltas', 'qcis'
    """
    # -------- split channels --------
    voice_raw = np.asarray(signal[0, :])  # 用于 SPL（保持原振幅）
    egg_raw   = np.asarray(signal[1, :])

    # -------- preprocess (pitch/CPP/SB 用) --------
    voice_proc = preprocess_voice_signal(voice_raw, sample_rate)
    egg_proc   = preprocess_egg_signal(egg_raw, sample_rate)

    # -------- EGG 周期 + 音频校验 --------
    egg_segments, _ = peak_cycles(egg_proc, sample_rate)
    segments = validate_segments_with_audio(
        egg_segments=egg_segments,
        samplerate=sample_rate,
        voice_signal=voice_proc,
        midi_min=30, midi_max=100, ac_threshold=0.25,
    )

    # -------- 帧级特征（CPP / SB / clarity / SPL）--------
    step = n - overlap                    # 与 period_downsampling 对齐
    window = np.hanning(n)
    k = n // 2

    times = []
    clarities = []
    cpps = []
    spectrum_balances = []
    spls = []

    for start in range(0, max(0, len(voice_proc) - n), step):
        seg_proc = voice_proc[start:start + n]
        seg_raw = voice_raw[start:start + n]  # 用于SPL计算
        win = seg_proc * window

        # 新版 find_f0：返回 (freq, confidence)，这里要 MIDI
        f0_midi, clarity = find_f0(win, sample_rate, n, k, threshold=0.0, midi=True, midi_min=30, midi_max=80)
        cpp = find_cpps(win, sample_rate, [60, 880])
        sb  = find_spectrum_balance(win, sample_rate)
        spl = find_spl(seg_raw)  # SPL使用固定窗口

        times.append(start / sample_rate)
        clarities.append(clarity)
        cpps.append(cpp)
        spectrum_balances.append(sb)
        spls.append(spl)

    # -------- 周期级特征（crest / qci / qdelta / F0）--------
    crests = []
    qcis = []
    qdeltas = []
    f0_per_cycle_midi = []

    for start, end in segments:
        v_seg_proc = voice_proc[start:end]   # 用于 F0/crest
        e_seg      = egg_proc[start:end]

        # crest / EGG 指标
        crests.append(find_crest_factor(v_seg_proc))
        qcis.append(find_qci(e_seg))
        qdeltas.append(find_deggmax(e_seg))

        # F0 (MIDI) —— 用新版 find_f0，并解包；周期段通常较短
        f0_m, _ = find_f0(v_seg_proc, sample_rate, midi=True, midi_min=30, midi_max=80)
        f0_per_cycle_midi.append(f0_m)

    # 可选：导出调试 CSV（MIDI,SPL）
    if debug_save_pairs_csv:
        os.makedirs(os.path.dirname(debug_save_pairs_csv), exist_ok=True)
        # 使用帧级的MIDI和SPL
        midi_vals = np.array([find_f0(voice_proc[start:start + n] * np.hanning(n), 
                                    sample_rate, midi=True, midi_min=30, midi_max=80)[0] 
                            for start in range(0, max(0, len(voice_proc) - n), step)])
        spl_vals = np.array(spls)
        if debug_round_unique:
            midi_vals = np.round(midi_vals, 0)
            spl_vals  = np.round(spl_vals, 0)
            pairs = pd.DataFrame({"MIDI": midi_vals, "SPL": spl_vals}).drop_duplicates()
        else:
            pairs = pd.DataFrame({"MIDI": midi_vals, "SPL": spl_vals})
        pairs.to_csv(debug_save_pairs_csv, index=False)
        print(f"[debug] saved frame MIDI-SPL pairs -> {debug_save_pairs_csv} "
              f"(rows={len(pairs)})")

    # -------- 将周期级指标下采样到帧栅格（与帧特征对齐）--------
    # 注意 frame_size 要与 step 对齐
    frame_size = step if step > 0 else n

    sampled_crests, period_counts = period_downsampling(
        crests, segments, times, frame_size=frame_size, sample_rate=sample_rate
    )
    sampled_qcis, _ = period_downsampling(
        qcis, segments, times, frame_size=frame_size, sample_rate=sample_rate
    )
    sampled_qdeltas, _ = period_downsampling(
        qdeltas, segments, times, frame_size=frame_size, sample_rate=sample_rate
    )
    sampled_f0_midi, _ = period_downsampling(
        f0_per_cycle_midi, segments, times, frame_size=frame_size, sample_rate=sample_rate
    )
    # SPL已经是帧级的，不需要下采样
    sampled_spl = spls

    # ---- 量化前的安全裁剪（适配 FonaDyn 的网格）----
    MIDI_MIN_BIN = 30
    MIDI_MAX_BIN = 100
    DB_MIN_BIN   = 40
    DB_MAX_BIN   = 120

    # 下采样后的连续值
    f0_vals  = np.array(sampled_f0_midi)   # MIDI（连续）
    spl_vals = np.array(sampled_spl)       # dB  （连续）

    # 裁剪，确保不会产生负的行索引或超界
    f0_vals  = np.clip(f0_vals,  MIDI_MIN_BIN, MIDI_MAX_BIN)
    spl_vals = np.clip(spl_vals, DB_MIN_BIN,  DB_MAX_BIN)

    # ---- 最终量化到整格（VRP 的离散 bin）----
    frequencies = np.floor(f0_vals).astype(int)   # MIDI bins
    spls        = np.floor(spl_vals).astype(int)  # dB bins

    # 长度一致性检查
    assert (len(spls) == len(sampled_crests) == len(sampled_qcis) ==
            len(sampled_qdeltas) == len(frequencies) ==
            len(clarities) == len(cpps) == len(spectrum_balances)), \
           "Frame-aligned arrays have mismatched lengths."

    return {
        'frequencies' : frequencies,
        'SPLs'        : spls,
        'Total'       : period_counts,
        'clarities'   : np.array(clarities),
        'crests'      : np.array(sampled_crests),
        'SBs'         : np.array(spectrum_balances),
        'CPPs'        : np.array(cpps),
        'qdeltas'     : np.array(sampled_qdeltas),
        'qcis'        : np.array(sampled_qcis)
    }


def post_process_metrics(metrics):
    """Post-process calculated metrics (same API,稍作整理)."""
    for k in metrics:
        metrics[k] = np.asarray(metrics[k])

    valid = (metrics['frequencies'] != 0) & (metrics['SPLs'] != 0) & (metrics['Total'] != 0)
    for k in metrics:
        metrics[k] = metrics[k][valid]

    merged = {}
    for f in range(0, 200):
        for spl in range(0, 200):
            mask = ((metrics['frequencies'] >= f - 0.5) & (metrics['frequencies'] < f + 0.5) &
                    (metrics['SPLs']       >= spl - 0.5) & (metrics['SPLs']       < spl + 0.5))
            if np.any(mask):
                for k in metrics:
                    merged.setdefault(k, [])
                    vals = metrics[k][mask]
                    merged[k].append(np.sum(vals) if k == 'Total' else np.mean(vals))
                # 不要覆盖实际的频率和SPL值，保持原始值
                # merged['frequencies'][-1] = f
                # merged['SPLs'][-1] = spl
    return merged
