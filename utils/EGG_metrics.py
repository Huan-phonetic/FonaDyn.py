'''
EGG metrics cycle by cycle.
'''

import numpy as np
from scipy.signal import correlate
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import librosa
from scipy.signal import butter, sosfilt
from scipy import stats
from cycle_picker import get_cycles
from EGG_process import process_EGG_signal
from scipy.fft import fft, fftfreq
import nolds

def find_CSE(EGG):
    CSE = nolds.sampen(EGG, emb_dim=2)
    return CSE

def find_qci(EGG):
    unit = unit_EGG(EGG)
    # qci is the area under the curve of the unit EGG signal
    qci = np.trapz(unit[1], unit[0])
    return qci

def find_dEGGmax(signal):
    scaled_signal = signal * 32767
    rounded_signal = np.round(scaled_signal).astype(np.int16)

    # Assuming signal is a 1D numpy array containing the discretized a(t)
    period_length_T = len(rounded_signal)
    # Find the largest positive difference dmax over the period
    # We use np.diff to compute the difference between consecutive samples
    # and then find the maximum
    dmax = np.max(np.abs(np.diff(rounded_signal)))
    
    # Assuming a sinusoidal waveform with peak-to-peak amplitude Ap-p = 2
    # and the given formula for QD
    Ap_p = np.max(rounded_signal) - np.min(rounded_signal)
    QD = 2 * dmax / (Ap_p * np.sin(2 * np.pi / period_length_T))
    
    return QD

def find_Ic(Qci, QDelta):
    # Check for positive QDelta to avoid mathematical error in logarithm
    if QDelta <= 0:
        raise ValueError("QDelta must be positive to compute the logarithm.")
    
    # Calculate Ic
    Ic = Qci * np.log10(QDelta)
    return Ic

def find_speedquotient(EGG):
    '''
    Input: EGG signal
    Output: speed quotient of EGG signal
    '''
    EGG_diff = np.diff(EGG)
    speedquotient = np.sum(np.abs(EGG_diff)) / np.sum(np.abs(EGG))
    return speedquotient

def find_openquotient(EGG):
    '''
    Input: EGG signal
    Output: open quotient of EGG signal
    '''
    EGG_diff = np.diff(EGG)
    openquotient = np.sum(EGG_diff[EGG_diff > 0]) / np.sum(EGG)
    return openquotient

def find_closedquotient(EGG):
    '''
    Input: EGG signal
    Output: closed quotient of EGG signal
    '''
    EGG_diff = np.diff(EGG)
    closedquotient = np.sum(EGG_diff[EGG_diff < 0]) / np.sum(EGG)
    return closedquotient

def find_contactquotient(EGG):
    '''
    Input: EGG signal
    Output: contact quotient of EGG signal
    '''
    EGG_diff = np.diff(EGG)
    contactquotient = np.sum(np.abs(EGG_diff)) / np.sum(EGG)
    return contactquotient

def find_dEGG(EGG):
    '''
    Input: EGG signal
    Output: dEGG of EGG signal
    '''
    EGG_diff = np.diff(EGG)
    dEGG = np.sum(np.abs(EGG_diff))
    return dEGG

def find_HRF(EGG, srnum_harmonics=None):
    # Compute the FFT of the signal
    n = len(EGG)
    egg_fft = fft(EGG)
    frequencies = fftfreq(n, 1/EGG)
    
    # Calculate the magnitude of the FFT components
    magnitudes = np.abs(egg_fft)
    
    # Calculate the power of each component (magnitude squared)
    powers = magnitudes**2
    
    # Find the fundamental frequency
    # Ignore the zero frequency component for fundamental frequency detection
    fundamental_idx = np.argmax(powers[1:]) + 1
    fundamental_freq = frequencies[fundamental_idx]
    fundamental_power = powers[fundamental_idx]
    
    # Determine the number of harmonics based on the fundamental frequency
    max_harmonic = int(frequencies[-1] / fundamental_freq)
    if num_harmonics is None or num_harmonics > max_harmonic:
        num_harmonics = max_harmonic
    
    # Sum the powers of the harmonics from 2 to N
    harmonic_powers = sum(powers[fundamental_idx * 2: fundamental_idx * num_harmonics + 1])
    
    # Calculate the power ratio in dB
    hrfegg = 10 * np.log10(harmonic_powers / fundamental_power)
    
    return hrfegg

def unit_EGG(EGG):
    '''
    Input: each cycle of EGG signal
    Output: unit EGG signal for computing qci
    '''
    EGG_shifted = EGG - np.min(EGG)
    
    # Normalize the amplitude to have a maximum of 1
    normalized_amplitude = EGG_shifted / np.max(EGG_shifted)
    
    # Normalize the time axis
    num_samples = len(EGG)
    normalized_time = np.linspace(0, 1, num_samples, endpoint=False)
    
    return normalized_time, normalized_amplitude

def main():
    audio_file = 'audio/test_Voice_EGG.wav'
    sr, signal = wavfile.read(audio_file)
    signal = signal[:, 1]
    signal = process_EGG_signal(signal, sr)
    # segments = get_cycles(signal, EGG=True)
    CSE = find_CSE(signal)
    print(CSE)
    plt.plot(signal)
    plt.show()




if __name__ == '__main__':

    main()