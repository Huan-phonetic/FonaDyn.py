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
scaling_factor = 0.707


def butter_highpass(cutoff, sample_rate, order=2):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, sample_rate, order=2):
    b, a = butter_highpass(cutoff, sample_rate, order=order)
    y = lfilter(b, a, data)
    return y


def voice_preprocess(voice, fs):
    voice = highpass_filter(voice, 30, sample_rate)
    voice = voice * scaling_factor
    return voice

# Example usage:
def main():
    voice_path = 'audio/test_Voice_EGG.wav'
    # Load voice data
    voice = librosa.load(voice_path, sr=sample_rate)[0]
    print(f'Before processing, min: {np.min(voice)}, max: {np.max(voice)}')
    voice = voice_preprocess(voice, sample_rate)
    print(f'After processing, min: {np.min(voice)}, max: {np.max(voice)}')
    # Convert to dB SPL (117 dB SPL RMS for the peak amplitude of 20 Pa)
    # Using 20 µPa as reference pressure
    P_ref = 20e-6
    dB_SPL = 20 * np.log10(np.abs(voice) / P_ref)
    print(f'{voice_path} of length {len(voice)/sample_rate}s successfully pre-processed.')
    print(f'Max dB SPL: {np.max(dB_SPL)}, Min dB SPL: {np.min(dB_SPL)}')
    save_path = 'audio/test_Voice_EGG_preprocessed.wav'
    librosa.output.write_wav(save_path, voice, sample_rate)    

if __name__ == '__main__':
    main()
