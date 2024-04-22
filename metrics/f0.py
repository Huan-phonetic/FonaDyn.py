import numpy as np
from scipy.signal import correlate
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import librosa


def autocorrelation(signal, n, k):
    """ Compute autocorrelation using FFT for efficiency """
    extended_size = n + k  # Extend the signal for autocorrelation lags
    fft_result = fft(signal, extended_size)
    power_spectrum = np.abs(fft_result)**2
    result = ifft(power_spectrum)
    return np.real(result)[:n]  # Return only the lags that are needed

def find_fundamental_frequency(signal, sr, n=4096, overlap=2048):
    """ Estimate the fundamental frequency using autocorrelation """
    step = n - overlap
    window = np.hanning(n)
    k = n // 2  # k as n/2
    frequencies = []
    confidences = []

    for start in range(0, len(signal) - n, step):
        segment = signal[start:start + n]
        windowed_segment = segment * window
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
                frequencies.append(frequency)
                confidences.append(confidence)
                    
            else:
                frequencies.append(0)
                confidences.append(0)
        else:
            frequencies.append(0)
            confidences.append(0)

    return frequencies, confidences
# Example usage:
def main():
    audio_file = 'audio/test_Voice_EGG.wav'
    signal, sr = librosa.load(audio_file, sr=44100, mono=True)

    frequencies, confidences = find_fundamental_frequency(signal, sr=sr)
    plt.plot(frequencies)
    plt.title("Fundamental Frequency")
    plt.xlabel("Frame")
    plt.ylabel("Frequency (Hz)")
    plt.show()

if __name__ == '__main__':
    main()