'''
Fixed time frame(23ms): F0/clarity, CPPs, Spectrum Balance
Cycle: SPL, crest factors
'''

import numpy as np
from scipy.signal import correlate
from scipy.fftpack import fft, ifft
from scipy.signal import find_peaks
import librosa
from scipy.signal import butter, sosfilt
from cycle_picker import get_cycles
from utils.preprocess_Voice import preprocess_Voice_signal
import numpy as np
import math
import numpy.matlib


def autocorrelation(signal, n, k):
    """ Compute autocorrelation using FFT for efficiency """
    extended_size = n + k  # Extend the signal for autocorrelation lags
    fft_result = fft(signal, extended_size)
    power_spectrum = np.abs(fft_result)**2
    result = ifft(power_spectrum)
    return np.real(result)[:n]  # Return only the lags that are needed

def find_f0(windowed_segment, sr, n, k,threshold=0.96, midi=False):
    acorr = autocorrelation(windowed_segment, n, k)
    # Normalize the autocorrelation
    acorr /= acorr[0]  # Normalize by zero-lag

    # Find all peaks in the autocorrelation and retrieve their heights
    peaks, properties = find_peaks(acorr, height=0)

    if peaks.size > 0:
        heights = properties['peak_heights']
        # Find the peak with the highest value which also serves as the confidence
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

def find_SPL(signal, reference=20e-6):
    signal = signal * 20
    rms = np.sqrt(np.mean(signal**2))
    spl = 20 * np.log10(rms / reference)
    return spl

    
def find_crest_factor(signal):
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    return peak / rms

def period_downsampling(metric, periods, times, frame_size=2024, sample_rate=44100):
    """
    Calculate the mean sound pressure level (SPL) for each time frame based on the input periods.

    Args:
        metric (list or np.array): SPL values for each period.
        periods (list of tuples or np.array): Start and end samples of each period.
        times (list or np.array): Start times of each frame.
        frame_size (int): Number of samples in each frame.
        sample_rate (int): Number of samples per second.

    Returns:
        tuple: A tuple containing:
            - np.array: Mean SPL values for each frame.
            - np.array: Count of periods within each frame.
    """
    # Calculate the start and end samples for each time frame
    frames_start = np.array(times) * sample_rate
    frames_end = frames_start + frame_size

    # Convert periods to a NumPy array for vectorized operations
    periods = np.array(periods)
    metric = np.array(metric)

    # Initialize arrays to store the mean SPL values and period counts for each time frame
    sampled_metrics = np.zeros(len(times))
    period_counts = np.zeros(len(times), dtype=int)

    # Compute indices where periods start within the time frames
    period_indices = np.logical_and(periods[:, None, 0] >= frames_start,
                                    periods[:, None, 0] < frames_end)

    # Calculate the mean SPL for each time frame using the identified indices
    for i in range(len(times)):
        in_frame_periods = metric[period_indices[:, i]]
        period_counts[i] = np.sum(period_indices[:, i])
        if period_counts[i] > 0:
            sampled_metrics[i] = np.mean(in_frame_periods)

    return sampled_metrics, period_counts

def find_CPPs(x, fs, pitch_range): 
    """
    Computes cepstral peak prominence for a given signal 

    Parameters
    -----------
    x: ndarray
        The audio signal
    fs: integer
        The sampling frequency
    pitch_range: list of 2 elements
        The pitch range where a peak is searched for

    Returns
    -----------
    float
        The cepstral peak prominence of the audio signal
    """
    # Quefrency
    frameLen = len(x)
    NFFT = 2**(math.ceil(np.log(frameLen)/np.log(2)))
    quef = np.linspace(0, frameLen/1000, NFFT)

    # Allowed quefrency range
    quef_lim = [int(np.round(fs/pitch_range[1])), int(np.round(fs/pitch_range[0]))]
    quef_seq = range(quef_lim[0]-1, quef_lim[1])
    
    # FrameMat
    frameMat = np.zeros(NFFT)
    frameMat[0: frameLen] = x

    # Hanning
    def hanning(N):
        x = np.array([i/(N+1) for i in range(1,int(np.ceil(N/2))+1)])
        w = 0.5-0.5*np.cos(2*np.pi*x)
        w_rev = w[::-1]
        return np.concatenate((w, w_rev[int((np.ceil(N%2))):]))
    win = hanning(frameLen)
    winmat = numpy.matlib.repmat(win, 1, 1)
    frameMat = frameMat[0:frameLen]*winmat
    frameMat = frameMat[0]
    
    # Cepstrum
    SpecMat = np.abs(np.fft.fft(frameMat))
    SpecdB = 20*np.log10(SpecMat)
    ceps = 20*np.log10(np.abs(np.fft.fft(SpecdB)))
    
    # Finding the peak
    ceps_lim = ceps[quef_seq]
    ceps_max = np.max(ceps_lim)
    max_index = np.argmax(ceps_lim)

    # Normalisation
    ceps_mean = np.mean(ceps_lim)
    p = np.polyfit(quef_seq, ceps_lim,1)
    ceps_norm = np.polyval(p, quef_seq[max_index])

    cpp = ceps_max-ceps_norm
    
    return cpp
    

def find_SB(windowed_segment, sr):
    # Define filter cutoff frequencies
    low_cutoff = 1500  # 1.5 kHz
    high_cutoff = 2000  # 2 kHz

    # Design a 4th-order Butterworth low-pass filter (24 dB/octave with a 4th order)
    sos_low = butter(4, low_cutoff, 'lp', fs=sr, output='sos')
    # Design a 4th-order Butterworth high-pass filter (24 dB/octave with a 4th order)
    sos_high = butter(4, high_cutoff, 'hp', fs=sr, output='sos')

    # Filter the segment
    low_filtered = np.asarray(sosfilt(sos_low, windowed_segment))
    high_filtered = np.asarray(sosfilt(sos_high, windowed_segment))

    # Calculate power in the bands, power is proportional to square of amplitude
    low_power = np.mean(low_filtered**2)
    high_power = np.mean(high_filtered**2)

    # Convert powers to dB
    low_power_db = 10 * np.log10(low_power + 1e-10)  # Adding a small constant to avoid log(0)
    high_power_db = 10 * np.log10(high_power + 1e-10)

    # Compute the level difference: High - Low
    SB = high_power_db - low_power_db

    return SB

def filter_out_zeros(frequencies, SPLs):
    # Create a mask for valid data points
    valid_mask = (frequencies != 0) & (SPLs != 0)

    # Filter data using the valid_mask
    frequencies = frequencies[valid_mask]
    SPLs = SPLs[valid_mask]

    return frequencies, SPLs


# Example usage:
def main():
    audio_file = 'audio/test_Voice_EGG.wav'
    # sr, signal = wavfile.read(audio_file)
    # signal = signal[:, 0]
    sr = 44100
    voice = librosa.load(audio_file, sr=44100)[0]

    # Preprocess the signal
    signal = preprocess_Voice_signal(voice, 44100)

    n = 2048  # Window size
    overlap = 1024  # Overlap size
    step = n - overlap  # Step size
    window = np.hanning(n)  # Hanning window

    for start in range(0, len(signal) - n, step):
        segment = signal[start:start + n]
        windowed_segment = segment * window
        # f, clarity = find_f0(windowed_segment, sr, n, n//2, threshold=0.90, midi=True)
        CPP = find_CPPs(windowed_segment, sr, [60, 880])


if __name__ == '__main__':
    main()