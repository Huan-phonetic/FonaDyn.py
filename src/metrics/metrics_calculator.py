'''
Calculate and process voice and EGG metrics.
'''

import numpy as np
import pandas as pd
from ..preprocessing.voice_preprocessor import preprocess_voice_signal
from ..preprocessing.egg_preprocessor import preprocess_egg_signal
from ..preprocessing.cycle_picker import phase_tracker
from .voice_metrics import (
    find_f0, find_cpps, find_spectrum_balance, find_spl,
    find_crest_factor, period_downsampling
)
from .egg_metrics import (
    find_qci, find_deggmax
)

def get_metrics(signal, sample_rate, n=8096, overlap=6072):
    """
    Calculate voice and EGG metrics.
    
    Args:
        signal (np.ndarray): Input signal (2 channels: voice and EGG)
        sample_rate (int): Sampling rate
        n (int): Window size
        overlap (int): Overlap size
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    # Split channels
    voice = signal[0, :]
    egg = signal[1, :]

    # Preprocess signals
    voice = preprocess_voice_signal(voice, sample_rate)
    egg = preprocess_egg_signal(egg, sample_rate)

    # Get cycle segments
    segments = phase_tracker(egg, sample_rate)

    # Calculate frame-based metrics
    step = n - overlap
    window = np.hanning(n)
    k = n // 2
    
    frequencies = []
    clarities = []
    cpps = []
    spectrum_balances = []
    times = []
    spls = []

    for start in range(0, len(voice) - n, step):
        segment = voice[start:start + n]
        windowed_segment = segment * window

        f0, clarity = find_f0(windowed_segment, sample_rate, n, k, threshold=0.96, midi=True)
        cpp = find_cpps(windowed_segment, sample_rate, [60, 880])
        sb = find_spectrum_balance(windowed_segment, sample_rate)
        spl = find_spl(segment)

        frequencies.append(f0)
        spls.append(spl)
        times.append(start / sample_rate)
        clarities.append(clarity)
        cpps.append(cpp)
        spectrum_balances.append(sb)

    # Calculate cycle-based metrics
    crests = []
    qcis = []
    qdeltas = []

    for start, end in segments:
        voice_segment = voice[start:end]
        egg_segment = egg[start:end]
        
        crest = find_crest_factor(voice_segment)
        qci = find_qci(egg_segment)
        qdelta = find_deggmax(egg_segment)

        crests.append(crest)
        qcis.append(qci)
        qdeltas.append(qdelta)

    # Downsample frame-based metrics
    sampled_crests, period_counts = period_downsampling(crests, segments, times, frame_size=2024, sample_rate=sample_rate)
    sampled_qcis, _ = period_downsampling(qcis, segments, times, frame_size=2024, sample_rate=sample_rate)
    sampled_qdeltas, _ = period_downsampling(qdeltas, segments, times, frame_size=2024, sample_rate=sample_rate)

    # Verify all metrics have the same length
    assert (len(spls) == len(sampled_crests) == len(sampled_qcis) ==
            len(sampled_qdeltas) == len(frequencies) == len(clarities) == len(cpps) == len(spectrum_balances))

    return {
        'frequencies': frequencies,
        'SPLs': spls,
        'Total': period_counts,
        'clarities': clarities,
        'crests': sampled_crests,
        'SBs': spectrum_balances,
        'CPPs': cpps,
        'qdeltas': sampled_qdeltas,
        'qcis': sampled_qcis
    }

def post_process_metrics(metrics):
    """
    Post-process calculated metrics.
    
    Args:
        metrics (dict): Dictionary of metrics
        
    Returns:
        dict: Processed metrics
    """
    # Convert lists to numpy arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key])

    # Remove invalid data points
    valid_mask = (metrics['frequencies'] != 0) & (metrics['SPLs'] != 0) & (metrics['Total'] != 0)
    for key in metrics:
        metrics[key] = metrics[key][valid_mask]

    # Merge metrics by frequency and SPL bins
    merged_metrics = {}
    for f in range(0, 200):
        for spl in range(0, 200):
            mask = (metrics['frequencies'] >= f-0.5) & (metrics['frequencies'] < f + 0.5) & \
                   (metrics['SPLs'] >= spl-0.5) & (metrics['SPLs'] < spl + 0.5)
            if np.sum(mask) > 0:
                for key in metrics:
                    if key not in merged_metrics:
                        merged_metrics[key] = []
                    masked_metrics = metrics[key][mask]
                    if key == 'Total':
                        merged_metrics[key].append(np.sum(masked_metrics))
                    else:
                        merged_metrics[key].append(np.mean(masked_metrics))
                merged_metrics['frequencies'][-1] = f
                merged_metrics['SPLs'][-1] = spl

    return merged_metrics 