'''
Two methods to pick a cycle:
1. double-sided peak-following, after Dolanský, marked as 'D' in the code
2. phase portrait, marked as 'P' in the code
'''

import numpy as np
from scipy.signal import find_peaks, hilbert

def get_cycles(signal, samplerate, EGG=True):
    """
    Get cycle boundaries using double-sided peak-following method.
    
    Args:
        signal (np.ndarray): Input signal
        samplerate (int): Sampling rate
        EGG (bool): Whether the signal is EGG (True) or voice (False)
        
    Returns:
        tuple: (segments, starts) where segments is a list of cycle boundaries and starts is a list of cycle start points
    """
    # Differentiate signal
    signal = np.diff(signal)
    
    # Set parameters based on signal type
    if EGG:
        height_p = 0.0001
        distance_p = 64
        prominence_p = 0.0005
        height_n = 0.0001
        distance_n = 64
        prominence_n = 0.0005
    else:
        height_p = 3
        distance_p = 140
        prominence_p = 5
        height_n = 3
        distance_n = 140
        prominence_n = 2

    # Find peaks
    positive_peaks, _ = find_peaks(signal, height=height_p, distance=distance_p, prominence=prominence_p)
    negative_peaks, _ = find_peaks(-signal, height=height_n, distance=distance_n, prominence=prominence_n)

    # Find peak pairs
    peaks = []
    for p in positive_peaks:
        following_negatives = negative_peaks[negative_peaks > p]
        if following_negatives.size > 0:
            n = following_negatives[0]
            peaks.append((p, n))
    
    # Find cycle boundaries
    starts = []
    ends = []
    segments = []
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    
    for p, n in peaks:
        # Find the zero crossing before the positive peak
        previous_zero = zero_crossings[zero_crossings < p]
        if previous_zero.size > 0:
            start = previous_zero[-1]
            starts.append(start)
            # Find the zero crossing after the negative peak
            following_zero = zero_crossings[zero_crossings > n]
            if following_zero.size > 0:
                end = following_zero[0]
                ends.append(end)
    
    # Filter cycles based on duration
    for i in range(len(starts)-1):
        start = starts[i]
        end = starts[i+1]
        if end - start < 0.02 * samplerate and end - start > 0.00023 * samplerate:
            segments.append((start, end))
            
    return segments, starts

def phase_tracker(signal, fs):
    """
    Track cycles using phase portrait method.
    
    Args:
        signal (np.ndarray): Input signal
        fs (int): Sampling rate
        
    Returns:
        list: List of cycle boundaries
    """
    def detect_zero_crossings(signal):
        """Detect zero-crossings in the signal."""
        return np.where(np.diff(np.signbit(signal)))[0]

    def extract_instantaneous_features(signal):
        """Extract amplitude envelope and normalized phase using the Hilbert transform."""
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        normalized_phase = instantaneous_phase / np.max(np.abs(instantaneous_phase))
        return amplitude_envelope, normalized_phase

    def validate_cycles(phase, zero_crossings, fs):
        """Validate cycles based on phase characteristics."""
        valid_boundaries = []
        for i in range(1, len(zero_crossings)):
            start_idx = zero_crossings[i - 1]
            end_idx = zero_crossings[i]
            
            # Extract the phase segment for the current cycle
            cycle_phase = phase[start_idx:end_idx]
            
            # Find positive and negative peaks within the cycle
            positive_peaks, _ = find_peaks(cycle_phase, height=0.5, prominence=0.2)
            negative_peaks, _ = find_peaks(-cycle_phase, height=0.5, prominence=0.2)
            
            # Validate the cycle
            if positive_peaks.size == 1 and negative_peaks.size == 1 and any(cycle_phase > 0.9) and any(cycle_phase < -0.9):
                valid_boundaries.append(end_idx)
                i += 1
        
        # Filter cycles based on duration
        periods = []
        for i in range(len(valid_boundaries)-1):
            start = valid_boundaries[i]
            end = valid_boundaries[i+1]
            if end - start < 0.02 * fs and end - start > 0.00023 * fs:
                periods.append((start, end))

        return periods
    
    # Extract features and get cycle boundaries
    amplitude_envelope, phase = extract_instantaneous_features(signal)
    zero_crossings = detect_zero_crossings(signal)
    cycle_boundaries = validate_cycles(phase, zero_crossings, fs)

    return cycle_boundaries 