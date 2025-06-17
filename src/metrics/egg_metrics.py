'''
EGG metrics cycle by cycle.
'''

import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import nolds

def find_cse(egg):
    """
    Calculate sample entropy of EGG signal.
    
    Args:
        egg (np.ndarray): Input EGG signal
        
    Returns:
        float: Sample entropy
    """
    return nolds.sampen(egg, emb_dim=2)

def find_qci(egg):
    """
    Calculate QCI (Quasi-Closed Interval) of EGG signal.
    
    Args:
        egg (np.ndarray): Input EGG signal
        
    Returns:
        float: QCI value
    """
    unit = unit_egg(egg)
    return np.trapz(unit[1], unit[0])

def find_deggmax(signal):
    """
    Calculate dEGGmax (maximum derivative of EGG).
    
    Args:
        signal (np.ndarray): Input EGG signal
        
    Returns:
        float: dEGGmax value
    """
    scaled_signal = signal * 32767
    rounded_signal = np.round(scaled_signal).astype(np.int16)

    period_length_T = len(rounded_signal)
    dmax = np.max(np.abs(np.diff(rounded_signal)))
    
    Ap_p = np.max(rounded_signal) - np.min(rounded_signal)
    QD = 2 * dmax / (Ap_p * np.sin(2 * np.pi / period_length_T))
    
    return QD

def find_ic(qci, qdelta):
    """
    Calculate Ic (Instability Coefficient).
    
    Args:
        qci (float): QCI value
        qdelta (float): QDelta value
        
    Returns:
        float: Ic value
    """
    if qdelta <= 0:
        raise ValueError("QDelta must be positive to compute the logarithm.")
    
    return qci * np.log10(qdelta)

def find_speed_quotient(egg):
    """
    Calculate speed quotient of EGG signal.
    
    Args:
        egg (np.ndarray): Input EGG signal
        
    Returns:
        float: Speed quotient
    """
    egg_diff = np.diff(egg)
    return np.sum(np.abs(egg_diff)) / np.sum(np.abs(egg))

def find_open_quotient(egg):
    """
    Calculate open quotient of EGG signal.
    
    Args:
        egg (np.ndarray): Input EGG signal
        
    Returns:
        float: Open quotient
    """
    egg_diff = np.diff(egg)
    return np.sum(egg_diff[egg_diff > 0]) / np.sum(egg)

def find_closed_quotient(egg):
    """
    Calculate closed quotient of EGG signal.
    
    Args:
        egg (np.ndarray): Input EGG signal
        
    Returns:
        float: Closed quotient
    """
    egg_diff = np.diff(egg)
    return np.sum(egg_diff[egg_diff < 0]) / np.sum(egg)

def find_contact_quotient(egg):
    """
    Calculate contact quotient of EGG signal.
    
    Args:
        egg (np.ndarray): Input EGG signal
        
    Returns:
        float: Contact quotient
    """
    egg_diff = np.diff(egg)
    return np.sum(np.abs(egg_diff)) / np.sum(egg)

def find_degg(egg):
    """
    Calculate dEGG (derivative of EGG).
    
    Args:
        egg (np.ndarray): Input EGG signal
        
    Returns:
        float: dEGG value
    """
    egg_diff = np.diff(egg)
    return np.sum(np.abs(egg_diff))

def find_hrf(egg, num_harmonics=None):
    """
    Calculate Harmonic-to-Fundamental Ratio (HRF) of EGG signal.
    
    Args:
        egg (np.ndarray): Input EGG signal
        num_harmonics (int, optional): Number of harmonics to consider
        
    Returns:
        float: HRF value in dB
    """
    n = len(egg)
    egg_fft = fft(egg)
    frequencies = fftfreq(n, 1/len(egg))
    
    magnitudes = np.abs(np.asarray(egg_fft))
    powers = magnitudes**2
    
    fundamental_idx = np.argmax(powers[1:]) + 1
    fundamental_freq = frequencies[fundamental_idx]
    fundamental_power = powers[fundamental_idx]
    
    max_harmonic = int(frequencies[-1] / fundamental_freq)
    if num_harmonics is None or num_harmonics > max_harmonic:
        num_harmonics = max_harmonic
    
    harmonic_powers = sum(powers[fundamental_idx * 2: fundamental_idx * num_harmonics + 1])
    
    return 10 * np.log10(harmonic_powers / fundamental_power)

def unit_egg(egg):
    """
    Normalize EGG signal to unit amplitude and time.
    
    Args:
        egg (np.ndarray): Input EGG signal
        
    Returns:
        tuple: (normalized_time, normalized_amplitude)
    """
    egg_shifted = egg - np.min(egg)
    normalized_amplitude = egg_shifted / np.max(egg_shifted)
    normalized_time = np.linspace(0, 1, len(egg), endpoint=False)
    
    return normalized_time, normalized_amplitude 