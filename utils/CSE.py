# import numpy as np

# def sample_entropy(u, m, r):
#     """
#     Calculate the Sample Entropy (SampEn) of a time series.
    
#     Parameters:
#     - u: Input time series
#     - m: Length of sequences to compare
#     - r: Tolerance for considering two sequences as similar
    
#     Returns:
#     - SampEn value as a float
#     """
#     n = len(u)
#     A = 0
#     B = 0

#     # Loop over the time series to get all possible subsequences of length m
#     for i in range(n - m):
#         for j in range(i + 1, n - m):
#             match = True
#             # Compare the subsequences
#             for k in range(m):
#                 if abs(u[i + k] - u[j + k]) > r:
#                     match = False
#                     break
#             if match:
#                 B += 1
#                 # Check for the full length m+1
#                 if abs(u[i + m] - u[j + m]) <= r:
#                     A += 1

#     if A == 0 or B == 0:
#         return 0
#     else:
#         return -np.log(A / B)

# # Example usage:
# # Generate a random time series
# data = np.random.rand(100)
# # Parameters for the Sample Entropy calculation
# m = 2  # Length of sequences to compare
# r = 0.2 * np.std(data)  # Tolerance set as a fraction of the standard deviation

# # Calculate the Sample Entropy
# entropy = sample_entropy(data, m, r)
# print("Sample Entropy:", entropy)


import numpy as np
import nolds
from scipy.io import wavfile
from EGG_process import process_EGG_signal
from cycle_picker import get_cycles

def segment_signal(signal, segment_lengths):
    """Yields segments of the signal based on provided lengths."""
    start = 0
    for length in segment_lengths:
        end = start + length
        yield signal[start:end]
        start = end

def compute_entropy_of_segments(segments, n=2, r=None):
    """Compute Sample Entropy for each segment."""
    entropies = []
    for segment in segments:
        if r is None:  # Calculate r as 0.2 times the standard deviation of the segment
            r = 0.2 * np.std(segment)
        entropy = nolds.sampen(segment, emb_dim=n, tolerance=r)
        entropies.append(entropy)
    return entropies

# Example usage
signal = np.random.randn(10000)  # This would be your long EGG signal
segment_lengths = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]  # Example segmentation

segments = list(segment_signal(signal, segment_lengths))
entropies = compute_entropy_of_segments(segments)

# Analyze entropy values to detect abrupt changes
for i, entropy in enumerate(entropies):
    print(f"Entropy of segment {i+1}: {entropy}")


audio_file = 'audio/test_Voice_EGG.wav'
sr, signal = wavfile.read(audio_file)
signal = signal[:, 1]
signal = process_EGG_signal(signal, sr)
segments, starts = get_cycles(signal, 44100, EGG=True)
signal_segments = list(segment_signal(signal, [end - start for start, end in segments]))
entropies = compute_entropy_of_segments(signal_segments)

# Analyze entropy values to detect abrupt changes
for i, entropy in enumerate(entropies):
    print(f"Entropy of segment {i+1}: {entropy}")