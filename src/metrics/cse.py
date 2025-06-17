'''
Calculate Sample Entropy (CSE) for EGG signals.
'''

import numpy as np
import nolds

def segment_signal(signal, segment_lengths):
    """
    Yield segments of the signal based on provided lengths.
    
    Args:
        signal (np.ndarray): Input signal
        segment_lengths (list): List of segment lengths
        
    Yields:
        np.ndarray: Signal segment
    """
    start = 0
    for length in segment_lengths:
        end = start + length
        yield signal[start:end]
        start = end

def compute_entropy_of_segments(segments, n=2, r=None):
    """
    Compute Sample Entropy for each segment.
    
    Args:
        segments (list): List of signal segments
        n (int): Embedding dimension
        r (float, optional): Tolerance for considering two sequences as similar
        
    Returns:
        list: List of entropy values for each segment
    """
    entropies = []
    for segment in segments:
        if r is None:
            r = 0.2 * np.std(segment)
        entropy = nolds.sampen(segment, emb_dim=n, tolerance=r)
        entropies.append(entropy)
    return entropies

def calculate_cse(signal, segment_lengths):
    """
    Calculate CSE (Sample Entropy) for a signal.
    
    Args:
        signal (np.ndarray): Input signal
        segment_lengths (list): List of segment lengths
        
    Returns:
        list: List of entropy values for each segment
    """
    segments = list(segment_signal(signal, segment_lengths))
    return compute_entropy_of_segments(segments) 