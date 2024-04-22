'''
Get the sound pressure level of a voice signal

Input: voice signal (processed at a cycle level)
Output: sound pressure level (SPL) in dB
'''


import numpy as np
import librosa
from scipy.signal import find_peaks
from matplotlib.pyplot import plot

RMS_ref = 0.05 # for a sine wave at 1 Pa
sr = 44100

def double_sided_peak_picking(signal, height=None, distance=None):
    # Find peaks using scipy's find_peaks function
    peaks, _ = find_peaks(signal, height=height, distance=distance)
    negative_peaks, _ = find_peaks(-signal, height=height, distance=distance)
    return np.sort(np.concatenate((peaks, negative_peaks)))

def compute_SPL_periods(signal, peaks):
    # Compute RMS for periods defined between peaks
    SPLs = []
    for i in range(len(peaks)-1):
        start, end = peaks[i], peaks[i+1]
        period_signal = signal[start:end]
        rms = np.sqrt(np.mean(period_signal**2))
        SPL = 20 * np.log10(rms / RMS_ref) + 94
        SPLs.append(SPL)
    return SPLs



# Example usage:
def main():
    voice_path = 'audio/test_Voice_EGG.wav'
    # Load voice data
    voice = librosa.load(voice_path, sr=sr)[0]
    peaks = double_sided_peak_picking(voice, height=0.02, distance=50)
    SPL = compute_SPL_periods(voice, peaks)
    plot(SPL)

if __name__ == '__main__':
    main()