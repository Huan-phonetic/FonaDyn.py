'''
Input: Voice channel
Output: preprocessed voice channel

1. A second-order high-pass Butterworth filter at 30 Hz.
2. Scaled that a peak-to-peak full scale sine wave represents ±20 Pa peak amplitude, which corresponds to 117 dB SPL RMS.
'''

import numpy as np
from scipy.signal import butter, lfilter
import librosa
from matplotlib.pyplot import plot

sample_rate = 44100


def butter_highpass(cutoff, sample_rate, order=2):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, sample_rate, order=2):
    b, a = butter_highpass(cutoff, sample_rate, order=order)
    y = lfilter(b, a, data)
    return y


def preprocess_Voice_signal(voice, fs):
    voice = highpass_filter(voice, 30, sample_rate)
    return voice

# Example usage:
def main():
    voice_path = 'audio/test_Voice_EGG.wav'
    # Load voice data
    voice = librosa.load(voice_path, sr=sample_rate)[0]
    voice = preprocess_Voice_signal(voice, sample_rate)

    print(f'File: {voice_path} of length {len(voice)} samples has been preprocessed')  

if __name__ == '__main__':
    main()
