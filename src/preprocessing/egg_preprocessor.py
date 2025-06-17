'''
Input: EGG data
Output: preprocessed EGG data
'''

import numpy as np
from scipy.signal import firwin, lfilter

def apply_high_pass_filter(signal, sample_rate=44100, numtaps=1025, cutoff=80):
    """
    Apply a high-pass FIR filter to the signal.
    
    Args:
        signal (np.ndarray): Input signal
        sample_rate (int): Sampling rate
        numtaps (int): Number of filter taps
        cutoff (int): Cutoff frequency in Hz
        
    Returns:
        np.ndarray: Filtered signal
    """
    fir_coeff = firwin(numtaps, cutoff, pass_zero=False, fs=sample_rate, window='hamming')
    filtered_signal = lfilter(fir_coeff, 1.0, signal)
    return filtered_signal

def apply_low_pass_filter(signal, sample_rate=44100, cutoff_hz=10000, numtaps=1025):
    """
    Apply a low-pass FIR filter to the signal.
    
    Args:
        signal (np.ndarray): Input signal
        sample_rate (int): Sampling rate
        cutoff_hz (int): Cutoff frequency in Hz
        numtaps (int): Number of filter taps
        
    Returns:
        np.ndarray: Filtered signal
    """
    fir_coeff = firwin(numtaps, cutoff_hz, fs=sample_rate, window='hamming', pass_zero=True)
    filtered_signal = lfilter(fir_coeff, 1.0, signal)
    return filtered_signal

def preprocess_egg_signal(egg_signal, sample_rate, threshold_dB=-40, expansion_ratio=1/4):
    """
    Preprocess the EGG signal using high-pass and low-pass filters.
    
    Args:
        egg_signal (np.ndarray): Input EGG signal
        sample_rate (int): Sampling rate
        threshold_dB (float): Threshold in dB
        expansion_ratio (float): Expansion ratio
        
    Returns:
        np.ndarray: Preprocessed EGG signal
    """
    # Apply high-pass filter
    filtered_signal = apply_high_pass_filter(egg_signal, sample_rate)
    
    # Apply low-pass filter
    filtered_signal = apply_low_pass_filter(filtered_signal, sample_rate)
    
    # TODO: Implement nine-point running median filter
    
    return filtered_signal 