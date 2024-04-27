'''
Input: EGG data
Output: preprocessed EGG data


'''

import numpy as np
from scipy.signal import firwin, lfilter, freqz
from scipy.fft import fft, ifft
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter


def process_EGG_signal(egg_signal, sample_rate, threshold_dB=-40, expansion_ratio=1/4):
    # high pass filter
    filtered_signal = apply_high_pass_filter(egg_signal)
    
    # low pass filter
    filtered_signal = apply_low_pass_filter(filtered_signal)
    
    # a nine-point running median filter

    return filtered_signal

def apply_high_pass_filter(signal, sample_rate=44100, numtaps=1025, cutoff=80):
    # Design the FIR filter
    fir_coeff = firwin(numtaps, cutoff, pass_zero=False, fs=sample_rate, window='hamming')

    # Apply the filter to the signal using lfilter, which applies the filter in a linear-phase manner
    filtered_signal = lfilter(fir_coeff, 1.0, signal)
    
    return filtered_signal

def apply_low_pass_filter(signal, sample_rate=44100, cutoff_hz=10000, numtaps=1025):
    # Design the low-pass FIR filter with a cutoff of 10 kHz
    fir_coeff = firwin(numtaps, cutoff_hz, fs=sample_rate, window='hamming', pass_zero=True)

    # Apply the filter to the signal
    filtered_signal = lfilter(fir_coeff, 1.0, signal)

    return filtered_signal

# Example usage
def main():
    audio_file = 'audio/test_Voice_EGG.wav'
    sr, signal = wavfile.read(audio_file)
    signal = signal[:, 1]
    processed_signal = process_EGG_signal(signal, sr)
    # plot raw and processed signal in two subplots
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title('Raw EGG Signal')
    plt.subplot(2, 1, 2)
    plt.plot(processed_signal)
    plt.title('Processed EGG Signal')
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':

    main()