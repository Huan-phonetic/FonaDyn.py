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
from EGG_process import process_EGG_signal
import numpy as np
from scipy.integrate import cumtrapz
from scipy.signal import find_peaks
from scipy.signal import hilbert

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
        height_p=0.0001
        distance_p=64
        prominence_p=0.0005
        height_n=0.0001
        distance_n=64
        prominence_n=0.0005
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
    # Cycles longer than 20 ms (882 samples) or shorter than 0.23 ms (10 samples) are rejected at this stage.
    for i in range(len(starts)-1):
        start = starts[i]
        end = starts[i+1]
        if end - start < 0.02 * samplerate and end - start > 0.00023 * samplerate:
            segments.append((start, end))
    return segments, starts

def phase_tracker(signal, fs):
    def detect_zero_crossings(signal):
        # Detects zero-crossings in the signal
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        return zero_crossings

    def extract_instantaneous_features(signal):
        """Extracts amplitude envelope and normalized phase using the Hilbert transform."""
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        normalized_phase = instantaneous_phase / np.max(np.abs(instantaneous_phase))
        return amplitude_envelope, normalized_phase


    def validate_cycles(phase, zero_crossings, fs):
        valid_boundaries = []
        for i in range(1, len(zero_crossings)):  # Start from 1 to safely use zero_crossings[i - 1]
            start_idx = zero_crossings[i - 1]
            end_idx = zero_crossings[i]
            
            # Extract the phase segment for the current cycle
            cycle_phase = phase[start_idx:end_idx]
            
            # Find positive and negative peaks within the cycle
            positive_peaks, _ = find_peaks(cycle_phase, height=0.5, prominence=0.2)
            negative_peaks, _ = find_peaks(-cycle_phase, height=0.5, prominence=0.2)

            
            # Validate the cycle by checking for exactly one positive and one negative peak
            if positive_peaks.size == 1 and negative_peaks.size == 1 and any(cycle_phase > 0.9) and any(cycle_phase < -0.9):
                valid_boundaries.append(end_idx)  # Append the start index of the cycle
                i += 1  # Skip the next cycle since it is already validated
        
        # Cycles longer than 20 ms (882 samples) or shorter than 0.23 ms (10 samples) are rejected at this stage.
        periods = []
        for i in range(len(valid_boundaries)-1):
            start = valid_boundaries[i]
            end = valid_boundaries[i+1]
            if end - start < 0.02 * fs and end - start > 0.00023 * fs:
                periods.append((start, end))

        return  periods
    
    # Extract instantaneous features
    amplitude_envelope, phase = extract_instantaneous_features(signal)
    zero_crossings = detect_zero_crossings(signal)
    cycle_boundaries = validate_cycles(phase, zero_crossings, fs)

    return cycle_boundaries

# Example usage:
def main():
    
    audio_file = 'audio/test_Voice_EGG.wav'
    signal, sr = librosa.load(audio_file, sr=44100, mono=False)
    signal = signal[1]
    sample_rate = 44100
    signal = process_EGG_signal(signal, 44100)
    # EGG picker parameters, using librosa.load
    # segments, starts = get_cycles(signal, 44100, EGG=True)
    # Create a sample signal (sine wave)
    
    # Get cycle boundaries
    cycle_boundaries = phase_tracker(signal, sample_rate)

    # Plot the signal, amplitude envelope, and phase
    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Signal')
    plt.scatter(cycle_boundaries, signal[cycle_boundaries], color='red', label='Cycle boundaries')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Signal, Amplitude Envelope, and Phase')

    plt.show(block=True)
    

if __name__ == '__main__':

    main()