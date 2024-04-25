'''
Fixed time frame(23ms): F0/clarity, CPPs, Spectrum Balance
Cycle: SPL, crest factors
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


def autocorrelation(signal, n, k):
    """ Compute autocorrelation using FFT for efficiency """
    extended_size = n + k  # Extend the signal for autocorrelation lags
    fft_result = fft(signal, extended_size)
    power_spectrum = np.abs(fft_result)**2
    result = ifft(power_spectrum)
    return np.real(result)[:n]  # Return only the lags that are needed

def get_audio_metrics(signal, sr, n=4096, overlap=2048):
    # A second-order high-pass Butterworth filter at 30 Hz is applied to the signal
    # to remove the DC component and low-frequency noise
    sos = butter(2, 30, 'hp', fs=sr, output='sos')
    signal = sosfilt(sos, signal)

    step = n - overlap
    window = np.hanning(n)
    k = n // 2  # k as n/2
    frequencies = []
    clarities = []
    CPPs = []
    SBs = []
    times = []


    for start in range(0, len(signal) - n, step):
        segment = signal[start:start + n]
        windowed_segment = segment * window

        f0, clarity = find_f0(windowed_segment, sr, n, k, threshold=0.93, midi=True)
        CPP = find_CPPs(windowed_segment, sr)
        SB = find_SB(windowed_segment, sr)

        frequencies.append(f0)
        times.append(start / sr)
        clarities.append(clarity)
        CPPs.append(CPP)
        SBs.append(SB)
    
    periods = get_cycles(signal, height=0.004, distance=64, prominence=0.005)
    # SPL and crest factor are calculated for each period
    SPLs = []
    crests = []
    for start, end in periods:
        segment = signal[start:end]
        SPL = find_SPL(segment)
        crest = find_crest_factor(segment)
        SPLs.append(SPL)
        crests.append(crest)

    sampled_SPLs = period_downsampling(SPLs, periods, times)
    sampled_clarities = period_downsampling(crests, periods, times)

    # Convert lists to numpy arrays
    frequencies = np.array(frequencies)
    sampled_SPLs = np.array(sampled_SPLs)
    times = np.array(times)
    clarities = np.array(clarities)
    sampled_clarities = np.array(sampled_clarities)
    CPPs = np.array(CPPs)
    SBs = np.array(SBs)

    # Create a mask for valid data points
    valid_mask = (frequencies != 0) & (sampled_SPLs != 0)

    # Filter data using the valid_mask
    frequencies = frequencies[valid_mask]
    sampled_SPLs = sampled_SPLs[valid_mask]
    times = times[valid_mask]
    clarities = clarities[valid_mask]
    sampled_clarities = sampled_clarities[valid_mask]
    CPPs = CPPs[valid_mask]
    SBs = SBs[valid_mask]

    # return a dictionary of the metrics
    return {
        'frequencies': frequencies,
        'SPLs': sampled_SPLs,
        'times': times,
        'periods': periods,
        'clarities': clarities,
        'crests': sampled_clarities,
        'CPPs': CPPs,
        'SBs': SBs        
    }

def find_f0(windowed_segment, sr, n, k,threshold=0.93, midi=False):
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
    signal = signal/32767 * 20
    rms = np.sqrt(np.mean(signal**2))
    spl = 20 * np.log10(rms / reference)
    return spl

    
def find_crest_factor(signal):
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    return peak / rms

# def period_downsampling(metric, periods, times, frame_size=2024):
#     # Initialize an array to store the mean SPL values for each time frame
#     sampled_metrics = np.zeros(len(times))
    
#     # Iterate over each time frame based on the times array
#     for i, time in enumerate(times):
#         # Define the start and end samples for the current time frame
#         frame_start = time * 44100
#         frame_end = time * 44100 + frame_size
        
#         # List to hold SPL values for periods within the current frame
#         period_metrics = []

#         for j, [start, end] in enumerate(periods):
#             if start >= frame_start and end <= frame_end:
#                 period_metrics.append(metric[j])

#         # Calculate the mean SPL for the current time frame
#         if len(period_metrics) > 0:
#             sampled_metrics[i] = np.mean(period_metrics)
#         else:
#             sampled_metrics[i] = 0

#     return sampled_metrics

def period_downsampling(metric, periods, times, frame_size=2024, sample_rate=44100):
    # Calculate the start and end samples for each time frame
    frames_start = np.array(times) * sample_rate
    frames_end = frames_start + frame_size

    # Convert periods to a NumPy array for vectorized operations
    periods = np.array(periods)
    metric = np.array(metric)

    # Initialize an array to store the mean SPL values for each time frame
    sampled_metrics = np.zeros(len(times))

    # Find indices where periods fit within the time frames
    period_indices = np.logical_and(
        periods[:, None, 0] >= frames_start,
        periods[:, None, 1] <= frames_end
    )

    # Calculate the mean SPL for each time frame
    # using the identified indices
    for i, (start, end) in enumerate(zip(frames_start, frames_end)):
        in_frame_periods = metric[period_indices[:, i]]
        if in_frame_periods.size > 0:
            sampled_metrics[i] = np.mean(in_frame_periods)

    return sampled_metrics

def find_CPPs(windowed_segment, sr):
    # Step 1: Perform initial FFT on the windowed segment
    spectrum = fft(windowed_segment)

    # Step 2: Compute the power spectrum and convert to dB
    power_spectrum = np.abs(spectrum) ** 2
    power_spectrum_db = 10 * np.log10(power_spectrum + 1e-10)  # Avoid log of zero

    # Step 3: Compute the cepstrum from the power spectrum in dB
    cepstrum = ifft(power_spectrum_db).real

    # Step 4: Select the first 1024 points of the cepstrum, resulting in 1024 quefrency bins
    cepstrum = cepstrum[:2048]

    # Step 5: Compute linear regression to detrend the cepstrum
    quefrency_bins = np.arange(2048)
    slope, intercept, _, _, _ = stats.linregress(quefrency_bins, cepstrum)

    # Detrend cepstrum
    trend_line = slope * quefrency_bins + intercept
    detrended_cepstrum = cepstrum - trend_line

    # Step 6: Find the peak in the detrended cepstrum
    # Considering vocal fundamental frequency range from 60 Hz to 880 Hz
    # Convert these frequencies to quefrency indices
    min_quefrency_index = int(sr / 880)
    max_quefrency_index = int(sr / 60)
    peak_cpp = np.max(detrended_cepstrum[min_quefrency_index:max_quefrency_index])

    return peak_cpp
    

def find_SB(windowed_segment, sr):
    # Define filter cutoff frequencies
    low_cutoff = 1500  # 1.5 kHz
    high_cutoff = 2000  # 2 kHz

    # Design a 4th-order Butterworth low-pass filter (24 dB/octave with a 4th order)
    sos_low = butter(4, low_cutoff, 'lp', fs=sr, output='sos')
    # Design a 4th-order Butterworth high-pass filter (24 dB/octave with a 4th order)
    sos_high = butter(4, high_cutoff, 'hp', fs=sr, output='sos')

    # Filter the segment
    low_filtered = sosfilt(sos_low, windowed_segment)
    high_filtered = sosfilt(sos_high, windowed_segment)

    # Calculate power in the bands, power is proportional to square of amplitude
    low_power = np.mean(low_filtered**2)
    high_power = np.mean(high_filtered**2)

    # Convert powers to dB
    low_power_db = 10 * np.log10(low_power + 1e-10)  # Adding a small constant to avoid log(0)
    high_power_db = 10 * np.log10(high_power + 1e-10)

    # Compute the level difference: High - Low
    SB = high_power_db - low_power_db

    return SB


# Example usage:
def main():
    audio_file = 'audio/test_Voice_EGG.wav'
    sr, signal = wavfile.read(audio_file)
    signal = signal[:, 0]

    audio_metrics = get_audio_metrics(signal, sr)
    print('Successfully extracted audio metrics!')
    # plot scatter frequency at x-axis and SPL at y-axis
    plt.scatter(audio_metrics['frequencies'], audio_metrics['SPLs'])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL (dB)')
    plt.show()
if __name__ == '__main__':
    main()