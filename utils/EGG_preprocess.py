'''
Input: EGG data
Output: preprocessed EGG data


'''

import numpy as np
from scipy.signal import firwin, lfilter, freqz
from scipy.fft import fft, ifft
from scipy.io import wavfile
import matplotlib.pyplot as plt

def process_EGG_signal(egg_signal, sample_rate, threshold_dB=-40, expansion_ratio=1/4):
    filtered_signal, _ = create_high_pass_filter(egg_signal, sample_rate)
    
    # # 2. FFT transformation with 50% overlap
    # num_frames = int((len(filtered_signal) - fft_size) / (fft_size / 2)) + 1
    # fft_frames = np.array([fft(filtered_signal[i*int(fft_size/2):i*int(fft_size/2)+fft_size]) for i in range(num_frames)])
    
    # # 3. Threshold and expansion in frequency domain
    # expanded_fft_frames = []
    # for frame in fft_frames:
    #     frame_dB = 20 * np.log10(np.abs(frame) + np.finfo(float).eps)  # Convert amplitude to dB, avoid log(0)
    #     expanded_dB = np.where(frame_dB < threshold_dB, frame_dB + expansion_ratio * (threshold_dB - frame_dB), frame_dB)
    #     expanded_fft_frame = np.abs(frame) * 10**(expanded_dB / 20) * np.exp(1j * np.angle(frame))  # Convert back from dB
    #     expanded_fft_frames.append(expanded_fft_frame)
    
    # # 4. Inverse FFT and reconstruct the time-domain signal with 50% overlap
    # reconstructed_signal = np.zeros_like(filtered_signal)
    # window = np.hanning(fft_size)
    # overlap_add_weight = np.zeros_like(filtered_signal)
    # for i, frame in enumerate(expanded_fft_frames):
    #     start_index = i * int(fft_size / 2)
    #     reconstructed_signal[start_index:start_index+fft_size] += np.real(ifft(frame)) * window
    #     overlap_add_weight[start_index:start_index+fft_size] += window
    
    # # Avoid division by zero
    # overlap_add_weight[overlap_add_weight == 0] = 1
    # reconstructed_signal /= overlap_add_weight  # Normalize by the window sums

    return filtered_signal

def create_high_pass_filter(signal, sample_rate, num_taps=1025, cutoff_hz=100, width_hz=20):
    # Calculate the Nyquist frequency
    nyquist = sample_rate / 2
    
    # Calculate the transition width in normalized frequency
    width_normalized = width_hz / nyquist
    
    # Calculate the desired cutoff frequency in normalized frequency
    cutoff_normalized = cutoff_hz / nyquist
    
    # Design the high-pass FIR filter using a Hamming window
    fir_coefficients = firwin(num_taps, cutoff=cutoff_normalized, window='hamming', pass_zero=False, width=width_normalized)
    
    # Apply the FIR filter to the signal
    filtered_signal = lfilter(fir_coefficients, 1.0, signal)
    
    return filtered_signal, fir_coefficients


# Example usage
def main():
    audio_file = 'audio/test_Voice_EGG.wav'
    sr, signal = wavfile.read(audio_file)
    signal = signal[:, 1]
    processed_signal = process_EGG_signal(signal, sr)
    # plot raw and processed signal in two subplots
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title('Raw EGG Signal')
    plt.subplot(2, 1, 2)
    plt.plot(processed_signal)
    plt.title('Processed EGG Signal')
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':

    main()