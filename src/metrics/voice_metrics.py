'''
Fixed time frame(23ms): F0/clarity, CPPs, Spectrum Balance
Cycle: SPL, crest factors
'''

import numpy as np
from scipy.signal import correlate, butter, sosfilt, find_peaks
from scipy.fftpack import fft, ifft
import math
import numpy.matlib

def autocorrelation(signal, n, k):
    """
    Compute autocorrelation using FFT for efficiency.
    
    Args:
        signal (np.ndarray): Input signal
        n (int): Number of lags to compute
        k (int): Maximum lag
        
    Returns:
        np.ndarray: Autocorrelation values
    """
    extended_size = n + k
    fft_result = fft(signal, extended_size)
    power_spectrum = np.abs(fft_result)**2
    result = ifft(power_spectrum)
    return np.real(result)[:n]

def find_f0(windowed_segment, sr, n, k, threshold=0.96, midi=False):
    """
    Find fundamental frequency using autocorrelation.
    
    Args:
        windowed_segment (np.ndarray): Windowed signal segment
        sr (int): Sampling rate
        n (int): Window size
        k (int): Maximum lag
        threshold (float): Confidence threshold
        midi (bool): Whether to return MIDI note number
        
    Returns:
        tuple: (frequency, confidence)
    """
    acorr = autocorrelation(windowed_segment, n, k)
    acorr /= acorr[0]  # Normalize by zero-lag

    peaks, properties = find_peaks(acorr, height=0)

    if peaks.size > 0:
        heights = properties['peak_heights']
        max_index = np.argmax(heights)
        peak = peaks[max_index]
        confidence = heights[max_index]
        if confidence > threshold:
            frequency = sr / peak
            if frequency > 1000:
                frequency = 0
                confidence = 0
            elif midi:
                frequency = 69 + 12 * np.log2(frequency / 440)
            return frequency, confidence
        else:
            return 0, 0
    else:
        return 0, 0

def find_spl(signal, reference=20e-6):
    """
    Calculate sound pressure level.
    
    Args:
        signal (np.ndarray): Input signal
        reference (float): Reference pressure
        
    Returns:
        float: Sound pressure level in dB
    """
    signal = signal * 20
    rms = np.sqrt(np.mean(signal**2))
    spl = 20 * np.log10(rms / reference)
    return spl

def find_crest_factor(signal):
    """
    Calculate crest factor.
    
    Args:
        signal (np.ndarray): Input signal
        
    Returns:
        float: Crest factor
    """
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    return peak / rms

def period_downsampling(metric, periods, times, frame_size=2024, sample_rate=44100):
    """
    Calculate mean metric values for each time frame.
    
    Args:
        metric (np.ndarray): Metric values for each period
        periods (np.ndarray): Start and end samples of each period
        times (np.ndarray): Start times of each frame
        frame_size (int): Number of samples in each frame
        sample_rate (int): Sampling rate
        
    Returns:
        tuple: (sampled_metrics, period_counts)
    """
    frames_start = np.array(times) * sample_rate
    frames_end = frames_start + frame_size

    periods = np.array(periods)
    metric = np.array(metric)

    sampled_metrics = np.zeros(len(times))
    period_counts = np.zeros(len(times), dtype=int)

    period_indices = np.logical_and(periods[:, None, 0] >= frames_start,
                                  periods[:, None, 0] < frames_end)

    for i in range(len(times)):
        in_frame_periods = metric[period_indices[:, i]]
        period_counts[i] = np.sum(period_indices[:, i])
        if period_counts[i] > 0:
            sampled_metrics[i] = np.mean(in_frame_periods)

    return sampled_metrics, period_counts

def find_cpps(x, fs, pitch_range):
    """
    Compute cepstral peak prominence.
    
    Args:
        x (np.ndarray): Input signal
        fs (int): Sampling frequency
        pitch_range (list): [min_pitch, max_pitch] in Hz
        
    Returns:
        float: Cepstral peak prominence
    """
    frameLen = len(x)
    NFFT = 2**(math.ceil(np.log(frameLen)/np.log(2)))
    quef = np.linspace(0, frameLen/1000, NFFT)

    quef_lim = [int(np.round(fs/pitch_range[1])), int(np.round(fs/pitch_range[0]))]
    quef_seq = range(quef_lim[0]-1, quef_lim[1])
    
    frameMat = np.zeros(NFFT)
    frameMat[0: frameLen] = x

    def hanning(N):
        x = np.array([i/(N+1) for i in range(1,int(np.ceil(N/2))+1)])
        w = 0.5-0.5*np.cos(2*np.pi*x)
        w_rev = w[::-1]
        return np.concatenate((w, w_rev[int((np.ceil(N%2))):]))
        
    win = hanning(frameLen)
    winmat = numpy.matlib.repmat(win, 1, 1)
    frameMat = frameMat[0:frameLen]*winmat
    frameMat = frameMat[0]
    
    SpecMat = np.abs(np.fft.fft(frameMat))
    SpecdB = 20*np.log10(SpecMat)
    ceps = 20*np.log10(np.abs(np.fft.fft(SpecdB)))
    
    ceps_lim = ceps[quef_seq]
    ceps_max = np.max(ceps_lim)
    max_index = np.argmax(ceps_lim)

    ceps_mean = np.mean(ceps_lim)
    p = np.polyfit(quef_seq, ceps_lim,1)
    ceps_norm = np.polyval(p, quef_seq[max_index])

    cpp = ceps_max-ceps_norm
    
    return cpp

def find_spectrum_balance(windowed_segment, sr):
    """
    Calculate spectrum balance.
    
    Args:
        windowed_segment (np.ndarray): Windowed signal segment
        sr (int): Sampling rate
        
    Returns:
        float: Spectrum balance in dB
    """
    low_cutoff = 1500  # 1.5 kHz
    high_cutoff = 2000  # 2 kHz

    sos_low = butter(4, low_cutoff, 'lp', fs=sr, output='sos')
    sos_high = butter(4, high_cutoff, 'hp', fs=sr, output='sos')

    low_filtered = np.asarray(sosfilt(sos_low, windowed_segment))
    high_filtered = np.asarray(sosfilt(sos_high, windowed_segment))

    low_power = np.mean(low_filtered**2)
    high_power = np.mean(high_filtered**2)

    low_power_db = 10 * np.log10(low_power + 1e-10)
    high_power_db = 10 * np.log10(high_power + 1e-10)

    return high_power_db - low_power_db

def filter_out_zeros(frequencies, spls):
    """
    Filter out zero values from frequency and SPL arrays.
    
    Args:
        frequencies (np.ndarray): Frequency values
        spls (np.ndarray): SPL values
        
    Returns:
        tuple: (filtered_frequencies, filtered_spls)
    """
    valid_mask = (frequencies != 0) & (spls != 0)
    return frequencies[valid_mask], spls[valid_mask] 