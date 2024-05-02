import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import librosa
from EGG_process import process_EGG_signal

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

        return np.array(valid_boundaries)
    
    # Extract instantaneous features
    amplitude_envelope, phase = extract_instantaneous_features(signal)
    zero_crossings = detect_zero_crossings(signal)
    cycle_boundaries = validate_cycles(phase, zero_crossings, fs)

    return cycle_boundaries

audio_file = 'audio/test_Voice_EGG.wav'
signal, sr = librosa.load(audio_file, sr=44100, mono=False)
signal = signal[1]
signal = process_EGG_signal(signal, 44100)
signal = signal[-20100:]

# Sampling rate
sample_rate = 44100

# Time axis
t = np.arange(0, len(signal)) / sample_rate

# Get cycle boundaries and phase
amplitude_envelope, phase = extract_instantaneous_features(signal)
zero_crossings = detect_zero_crossings(signal)
cycle_boundaries = validate_cycles(phase, zero_crossings, sample_rate)

# Plot the signal, phase, zero-crossings, and cycle boundaries
plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Signal')
plt.plot(t, phase, label='Phase')
plt.plot(t[cycle_boundaries], signal[cycle_boundaries], 'go', label='Cycle boundaries')
plt.xlabel('Time (s)')
plt.show(block = True)