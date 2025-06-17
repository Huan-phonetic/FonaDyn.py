'''
Input: Voice channel
Output: preprocessed voice channel

1. A second-order high-pass Butterworth filter at 30 Hz.
2. Scaled that a peak-to-peak full scale sine wave represents ±20 Pa peak amplitude, which corresponds to 117 dB SPL RMS.
'''

import numpy as np
from scipy.signal import butter, lfilter
import librosa

def butter_highpass(cutoff, sample_rate, order=2):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, sample_rate, order=2):
    b, a = butter_highpass(cutoff, sample_rate, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_voice_signal(voice, fs):
    """
    Preprocess the voice signal using a high-pass filter.
    
    Args:
        voice (np.ndarray): Input voice signal
        fs (int): Sampling frequency
        
    Returns:
        np.ndarray: Preprocessed voice signal
    """
    voice = highpass_filter(voice, 30, fs)
    return voice 