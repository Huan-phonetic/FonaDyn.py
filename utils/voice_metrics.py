'''
For each time segment in audio, compute CPPs, crest factors, Spectrum Balance, SPL and F0
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
    SPLs = []
    clarities = []
    crests = []
    CPPs = []
    SBs = []
    times = []


    for start in range(0, len(signal) - n, step):
        segment = signal[start:start + n]
        windowed_segment = segment * window

        f0, clarity = find_f0(windowed_segment, sr, n, k)
        SPL = find_SPL(windowed_segment, sr)
        crest = find_crest_factor(windowed_segment)
        CPP = find_CPPs(windowed_segment, sr)
        SB = find_SB(windowed_segment, sr)

        frequencies.append(f0)
        SPLs.append(SPL)
        times.append(start / sr)
        clarities.append(clarity)
        crests.append(crest)
        CPPs.append(CPP)
        SBs.append(SB)
    
    # return a dictionary of the metrics
    return {
        'frequencies': frequencies,
        'SPLs': SPLs,
        'times': times,
        'clarities': clarities,
        'crests': crests,
        'CPPs': CPPs,
        'SBs': SBs        
    }

def find_f0(windowed_segment, sr, n, k):
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
        if confidence > 0.93:
            frequency = sr / peak
            if frequency > 1000:
                frequency = 0
            return frequency, confidence
        else:
            return 0, 0
    else:
        return 0, 0
    
def find_SPL(windowed_segment, sr):
    RMS_ref = 0.05  # for a sine wave at 1 Pa
    rms = np.sqrt(np.mean(windowed_segment**2))
    SPL = 20 * np.log10(rms / RMS_ref) + 94
    return SPL
    
def find_crest_factor(windowed_segment):
    rms = np.sqrt(np.mean(windowed_segment**2))
    peak = np.max(np.abs(windowed_segment))
    return peak / rms

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
    signal, sr = librosa.load(audio_file, sr=44100, mono=True)

    audio_metrics = get_audio_metrics(signal, sr)
    # plot scatter frequency at x-axis and SPL at y-axis
    plt.scatter(audio_metrics['frequencies'], audio_metrics['SPLs'])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL (dB)')
    plt.show()
if __name__ == '__main__':
    main()