'''
Two methods to pick a cycle:
1. double-sided peak-following, after Dolanský, marked as 'D' in the code
2. phase portrait, marked as 'P' in the code
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import librosa

def get_cycles(signal, height=None, distance=None, prominence=None):
    def find_zero_crossings(signal):
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        return zero_crossings
    
    # differenciated signal
    signal = np.diff(signal)

    # Find positive peaks
    positive_peaks, _ = find_peaks(signal, height=height, distance=distance, prominence=prominence)
    # Find negative peaks by inverting the signal
    negative_peaks, _ = find_peaks(-signal, height=height, distance=distance, prominence=prominence)

    # plt.figure(figsize=(10, 4))
    # plt.plot(signal, label='Signal')
    # plt.scatter(positive_peaks, signal[positive_peaks], color='red', label='Positive peaks')
    # plt.scatter(negative_peaks, signal[negative_peaks], color='green', label='Negative peaks')
    # plt.legend()
    # plt.show()

    # period accepts only one positive peak followed by one negative peak, the boundaries are defined by the three crossings
    peaks = []
    for p in positive_peaks:
        following_negatives = negative_peaks[negative_peaks > p]
        if following_negatives.size > 0:
            n = following_negatives[0]
            peaks.append((p, n))
    
    starts = []
    ends = []
    segments = []
    zero_crossings = find_zero_crossings(signal)
    
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
    # Cycles longer than 20 ms (50 Hz) or shorter than 0.23 ms (4410 Hz) are rejected at this stage.
    for i in range(len(starts)-1):
        start = starts[i]
        end = starts[i+1]
        if end - start < 0.02 * 44100 and end - start > 0.00023 * 44100:
            segments.append((start, end))
    return segments


# Example usage:
def main():
    
    audio_file = 'audio/test_Voice_EGG.wav'
    signal, sr = librosa.load(audio_file, sr=44100, mono=True)
    segments = get_cycles(signal, height=0.004, distance=64, prominence=0.005)

    # Plotting the results
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label='Signal')
    for start, end in segments:
        plt.axvline(start, color='red', linestyle='--')
        # plt.axvline(end, color='green', linestyle='--')
    plt.legend()

    plt.show()

if __name__ == '__main__':

    main()