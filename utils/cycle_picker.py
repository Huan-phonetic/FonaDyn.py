'''
Two methods to pick a cycle:
1. double-sided peak-following, after Dolanský, marked as 'D' in the code
2. phase portrait, marked as 'P' in the code
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import librosa
from scipy.io import wavfile

def get_cycles(signal, samplerate, EGG=True): 
    '''
    Input: signal, EGG=True if the signal is EGG, False if the signal is voice
    Output: segments, a list of tuples, each tuple contains the start and end index of a cycle

    Note: for audio signal, the best parameters have not been found yet, the current parameters are for EGG signal
    Note2: the current parameters are for Librosa.load
    '''

    # differenciated signal
    signal = np.diff(signal)
    if EGG:
        height_p=0.004
        distance_p=64
        prominence_p=0.005
        height_n=0.004
        distance_n=64
        prominence_n=0.005
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
    # Cycles longer than 20 ms (50 Hz) or shorter than 0.23 ms (4410 Hz) are rejected at this stage.
    for i in range(len(starts)-1):
        start = starts[i]
        end = starts[i+1]
        if end - start < 0.02 * samplerate and end - start > 0.00023 * samplerate:
            segments.append((start, end))
    return segments, starts


# Example usage:
def main():
    
    audio_file = 'audio/test_Voice_EGG.wav'
    signal, sr = librosa.load(audio_file, sr=44100, mono=False)
    signal = signal[0]
    signal = signal[:44100]
    # EGG picker parameters, using librosa.load
    segments = get_cycles(signal, EGG=True)
    # sr, signal = wavfile.read(audio_file)
    # signal = signal[:, 0]
    # # choose the last second of the signal
    # signal = signal[:44100]
    # segments = get_cycles(signal, EGG=False)

    # Plotting the results
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label='Signal')
    for start, end in segments:
        plt.axvline(start, color='red', linestyle='--')
        # plt.axvline(end, color='blue', linestyle='--')
    plt.legend()

    plt.show()

if __name__ == '__main__':

    main()