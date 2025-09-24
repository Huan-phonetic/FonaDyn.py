#!/usr/bin/env python3
"""
VoiceMap Metrics Module
Individual metric calculation functions with type annotations and error handling
"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt, medfilt, welch
from scipy.fft import fft, ifft
from scipy.stats import linregress
from typing import Tuple, Optional
import logging

from config import VoiceMapConfig

logger = logging.getLogger(__name__)


class MetricCalculator:
    """Base class for metric calculations with common utilities"""
    
    def __init__(self, config: VoiceMapConfig):
        """
        Initialize metric calculator with configuration.
        
        Args:
            config: VoiceMap configuration object
        """
        self.config = config
        self.sample_rate = config.sample_rate
        self.min_samples = config.min_samples
        self.min_frequency = config.min_frequency
        self.max_period_samples = config.max_period_samples
        
        # SPL sliding window parameters
        self.spl_window_size = config.spl_window_size
        self.spl_hop_size = config.spl_hop_size
        
        # CPP parameters
        self.cpp_fft_size = config.cpp_fft_size
        self.cpp_ceps_size = config.cpp_ceps_size
        self.cpp_low_bin = config.cpp_low_bin
        self.cpp_high_bin = config.cpp_high_bin
        self.cpp_dither_amp = config.cpp_dither_amp
        
        # SpecBal parameters
        self.specbal_cutoff_low = config.specbal_cutoff_low
        self.specbal_cutoff_high = config.specbal_cutoff_high
        self.specbal_rms_window = config.specbal_rms_window
    
    def assign_metric_to_cycles(
        self, 
        cycle_indices: np.ndarray, 
        metric_values: np.ndarray, 
        hop_size: int
    ) -> np.ndarray:
        """
        Assign metric values to cycles based on window positions.
        
        Args:
            cycle_indices: Array of cycle trigger indices
            metric_values: Array of metric values from sliding windows
            hop_size: Hop size for window calculation
        
        Returns:
            Array of metric values assigned to each cycle
        """
        cycle_metric = np.zeros(len(cycle_indices))
        
        for i, cycle_idx in enumerate(cycle_indices):
            window_idx = cycle_idx // hop_size
            
            if window_idx < len(metric_values):
                cycle_metric[i] = metric_values[window_idx]
            else:
                cycle_metric[i] = metric_values[-1] if len(metric_values) > 0 else 0.0
        
        return cycle_metric
    
    def calculate_sliding_rms(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate sliding window RMS values.
        
        Args:
            signal: Input signal
        
        Returns:
            Array of RMS values for each window
        """
        n_windows = (len(signal) - self.spl_window_size) // self.spl_hop_size + 1
        rms_values = np.zeros(n_windows)
        
        for i in range(n_windows):
            start_idx = i * self.spl_hop_size
            end_idx = start_idx + self.spl_window_size
            
            if end_idx <= len(signal):
                window_data = signal[start_idx:end_idx]
                rms_values[i] = np.sqrt(np.mean(window_data**2))
        
        return rms_values
    
    def rms_to_spl(self, rms_values: np.ndarray) -> np.ndarray:
        """
        Convert RMS values to SPL (Sound Pressure Level) in dB.
        
        Args:
            rms_values: RMS values
        
        Returns:
            SPL values in dB
        """
        spl_values = np.zeros_like(rms_values)
        
        for i in range(len(rms_values)):
            if rms_values[i] > 0:
                spl_values[i] = 20 * np.log10(rms_values[i])
            else:
                spl_values[i] = -100  # Protection value
        
        return spl_values


class SPLCalculator(MetricCalculator):
    """SPL (Sound Pressure Level) Calculator"""
    
    def calculate(
        self, 
        voice_signal: np.ndarray, 
        cycle_triggers: np.ndarray
    ) -> np.ndarray:
        """
        Calculate SPL using sliding window approach.
        
        Args:
            voice_signal: Preprocessed voice signal
            cycle_triggers: Cycle trigger array
        
        Returns:
            Array of SPL values for each cycle
        """
        logger.info("Calculating SPL...")
        
        try:
            # 20ms delay alignment
            delay_samples = int(0.02 * self.sample_rate)
            voice_delayed = np.concatenate([
                np.zeros(delay_samples), 
                voice_signal[:-delay_samples]
            ])
            
            # Calculate sliding window RMS
            window_rms = self.calculate_sliding_rms(voice_delayed)
            
            # Convert to SPL
            window_spl = self.rms_to_spl(window_rms)
            
            # Assign SPL values to cycles
            cycle_indices = np.where(cycle_triggers > 0.5)[0]
            cycle_spl = np.zeros(len(cycle_indices))
            
            for i, cycle_idx in enumerate(cycle_indices):
                window_idx = cycle_idx // self.spl_hop_size
                
                if window_idx < len(window_spl):
                    cycle_spl[i] = window_spl[window_idx]
                else:
                    cycle_spl[i] = window_spl[-1] if len(window_spl) > 0 else -100
            
            logger.info(f"  Calculated {len(window_spl)} SPL windows")
            logger.info(f"  Assigned SPL values to {len(cycle_indices)} cycles")
            logger.info(f"  SPL range: {cycle_spl.min():.1f} - {cycle_spl.max():.1f} dB")
            
            return cycle_spl
            
        except Exception as e:
            logger.error(f"Error calculating SPL: {e}")
            raise


class ClarityCalculator(MetricCalculator):
    """Clarity Calculator using cross-correlation between consecutive cycles"""
    
    def calculate(
        self, 
        voice_signal: np.ndarray, 
        cycle_triggers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Clarity using cross-correlation between consecutive cycles.
        
        Args:
            voice_signal: Preprocessed voice signal
            cycle_triggers: Cycle trigger array
        
        Returns:
            Tuple of (midi_values, clarity_values)
        """
        logger.info("Calculating Clarity...")
        
        try:
            cycle_indices = np.where(cycle_triggers > 0.5)[0]
            
            if len(cycle_indices) < 2:
                return np.array([]), np.array([])
            
            # Extract cycles
            cycles = []
            cycle_f0 = []
            
            for i in range(len(cycle_indices) - 1):
                start_idx = cycle_indices[i]
                end_idx = cycle_indices[i + 1]
                
                if end_idx - start_idx >= self.min_samples:
                    cycle_data = voice_signal[start_idx:end_idx]
                    
                    # Normalization
                    if np.std(cycle_data) > 0:
                        cycle_normalized = (cycle_data - np.mean(cycle_data)) / np.std(cycle_data)
                        cycles.append(cycle_normalized)
                        
                        # Calculate F0
                        period_samples = end_idx - start_idx
                        f0_hz = self.sample_rate / period_samples
                        cycle_f0.append(f0_hz)
                    else:
                        cycle_f0.append(np.nan)
            
            # Calculate Clarity (cross-correlation between consecutive cycles)
            clarity_values = []
            midi_values = []
            
            for i in range(len(cycles) - 1):
                current_cycle = cycles[i]
                next_cycle = cycles[i + 1]
                
                # Ensure same length
                min_length = min(len(current_cycle), len(next_cycle))
                if min_length > 0:
                    current_cycle = current_cycle[:min_length]
                    next_cycle = next_cycle[:min_length]
                    
                    # Calculate normalized cross-correlation
                    correlation = np.corrcoef(current_cycle, next_cycle)[0, 1]
                    
                    # Handle NaN values
                    if np.isnan(correlation):
                        clarity_values.append(0.0)
                    else:
                        clarity_values.append(max(0.0, correlation))
                    
                    # MIDI value
                    if not np.isnan(cycle_f0[i]):
                        midi_values.append(librosa.hz_to_midi(cycle_f0[i]))
                    else:
                        midi_values.append(np.nan)
            
            # Convert to numpy arrays
            clarity_values = np.array(clarity_values)
            midi_values = np.array(midi_values)
            
            logger.info(f"  Calculated {len(clarity_values)} Clarity values")
            logger.info(f"  Clarity range: {clarity_values.min():.3f} - {clarity_values.max():.3f}")
            
            return midi_values, clarity_values
            
        except Exception as e:
            logger.error(f"Error calculating Clarity: {e}")
            raise


class CPPCalculator(MetricCalculator):
    """CPP (Cepstral Peak Prominence) Calculator"""
    
    def calculate(
        self, 
        voice_signal: np.ndarray, 
        cycle_triggers: np.ndarray
    ) -> np.ndarray:
        """
        Calculate CPP using cepstral analysis.
        
        Args:
            voice_signal: Preprocessed voice signal
            cycle_triggers: Cycle trigger array
        
        Returns:
            Array of CPP values for each cycle
        """
        logger.info("Calculating CPP...")
        
        try:
            cycle_indices = np.where(cycle_triggers > 0.5)[0]
            cpp_values = []
            
            # Calculate sliding window CPP
            window_size = self.spl_window_size
            hop_size = self.spl_hop_size
            
            for i in range(0, len(voice_signal) - window_size + 1, hop_size):
                window_data = voice_signal[i:i + window_size]
                
                # Add white noise dither
                dither = np.random.normal(0, self.cpp_dither_amp, len(window_data))
                window_with_dither = window_data + dither
                
                # FFT (Hanning window, 2048 points)
                windowed = window_with_dither * np.hanning(len(window_with_dither))
                
                # Zero-pad to 2048 points
                if len(windowed) < self.cpp_fft_size:
                    windowed = np.pad(
                        windowed, 
                        (0, self.cpp_fft_size - len(windowed)), 
                        'constant'
                    )
                else:
                    windowed = windowed[:self.cpp_fft_size]
                
                # FFT
                fft_result = fft(windowed)
                magnitude = np.abs(fft_result)
                
                # Calculate cepstrum
                log_magnitude = np.log(magnitude + 1e-10)
                cepstrum_result = np.real(ifft(log_magnitude))
                
                # Take first 1024 points
                cepstrum_result = cepstrum_result[:self.cpp_ceps_size]
                
                # PeakProminence algorithm
                cpp_value = self.peak_prominence(
                    cepstrum_result, 
                    self.cpp_low_bin, 
                    self.cpp_high_bin
                )
                cpp_values.append(cpp_value)
            
            # Assign CPP values to cycles
            cycle_cpp = self.assign_metric_to_cycles(cycle_indices, cpp_values, hop_size)
            
            logger.info(f"  Calculated {len(cpp_values)} CPP windows")
            logger.info(f"  Assigned CPP values to {len(cycle_indices)} cycles")
            logger.info(f"  CPP range: {cycle_cpp.min():.3f} - {cycle_cpp.max():.3f}")
            
            return cycle_cpp
            
        except Exception as e:
            logger.error(f"Error calculating CPP: {e}")
            raise
    
    def peak_prominence(
        self, 
        cepstrum_data: np.ndarray, 
        low_bin: int, 
        high_bin: int
    ) -> float:
        """
        PeakProminence algorithm implementation.
        
        Args:
            cepstrum_data: Cepstrum data
            low_bin: Low bin index
            high_bin: High bin index
        
        Returns:
            Peak prominence value
        """
        # Convert to dB
        cepstrum_db = 20 * np.log10(np.abs(cepstrum_data) + 1e-10)
        
        # Linear regression
        x = np.arange(low_bin, high_bin + 1)
        y = cepstrum_db[low_bin:high_bin + 1]
        
        if len(x) < 2:
            return 0.0
        
        slope, intercept, _, _, _ = linregress(x, y)
        
        # Calculate regression line
        regression_line = slope * x + intercept
        
        # Find maximum peak
        peak_idx = np.argmax(y)
        peak_value = y[peak_idx]
        regression_value = regression_line[peak_idx]
        
        # CPP = peak value minus regression line
        cpp = peak_value - regression_value
        
        return max(0.0, cpp)


class SpecBalCalculator(MetricCalculator):
    """SpecBal (Spectral Balance) Calculator"""
    
    def calculate(
        self, 
        voice_signal: np.ndarray, 
        cycle_triggers: np.ndarray
    ) -> np.ndarray:
        """
        Calculate SpecBal using spectral filtering.
        
        Args:
            voice_signal: Preprocessed voice signal
            cycle_triggers: Cycle trigger array
        
        Returns:
            Array of SpecBal values for each cycle
        """
        logger.info("Calculating SpecBal...")
        
        try:
            cycle_indices = np.where(cycle_triggers > 0.5)[0]
            specbal_values = []
            
            # Calculate sliding window SpecBal
            window_size = self.spl_window_size
            hop_size = self.spl_hop_size
            
            for i in range(0, len(voice_signal) - window_size + 1, hop_size):
                window_data = voice_signal[i:i + window_size]
                
                # Low-pass filter (1500 Hz, 4th order Butterworth)
                nyquist = self.sample_rate / 2
                low_cutoff = self.specbal_cutoff_low / nyquist
                b, a = butter(4, low_cutoff, btype='low')
                level_lo = filtfilt(b, a, window_data)
                
                # High-pass filter (2000 Hz, 4th order Butterworth)
                high_cutoff = self.specbal_cutoff_high / nyquist
                b, a = butter(4, high_cutoff, btype='high')
                level_hi = filtfilt(b, a, window_data)
                
                # RMS calculation
                rms_lo = np.sqrt(np.mean(level_lo**2))
                rms_hi = np.sqrt(np.mean(level_hi**2))
                
                # Convert to dB
                level_lo_db = 20 * np.log10(rms_lo + 1e-10)
                level_hi_db = 20 * np.log10(rms_hi + 1e-10)
                
                # SpecBal = levelHi - levelLo
                specbal = level_hi_db - level_lo_db
                
                # Sanitize (limit to -50dB)
                specbal = max(specbal, -50.0)
                specbal_values.append(specbal)
            
            # Assign SpecBal values to cycles
            cycle_specbal = self.assign_metric_to_cycles(cycle_indices, specbal_values, hop_size)
            
            logger.info(f"  Calculated {len(specbal_values)} SpecBal windows")
            logger.info(f"  Assigned SpecBal values to {len(cycle_indices)} cycles")
            logger.info(f"  SpecBal range: {cycle_specbal.min():.3f} - {cycle_specbal.max():.3f}")
            
            return cycle_specbal
            
        except Exception as e:
            logger.error(f"Error calculating SpecBal: {e}")
            raise


class CrestCalculator(MetricCalculator):
    """Crest (Crest Factor) Calculator"""
    
    def calculate(
        self, 
        voice_signal: np.ndarray, 
        cycle_triggers: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Crest Factor per cycle.
        
        Args:
            voice_signal: Preprocessed voice signal
            cycle_triggers: Cycle trigger array
        
        Returns:
            Array of Crest values for each cycle
        """
        logger.info("Calculating Crest...")
        
        try:
            cycle_indices = np.where(cycle_triggers > 0.5)[0]
            crest_values = []
            
            # 20ms delay alignment
            delay_samples = int(0.02 * self.sample_rate)
            voice_delayed = np.concatenate([
                np.zeros(delay_samples), 
                voice_signal[:-delay_samples]
            ])
            
            for i in range(len(cycle_indices) - 1):
                start_idx = cycle_indices[i]
                end_idx = cycle_indices[i + 1]
                
                if end_idx - start_idx >= self.min_samples:
                    cycle_data = voice_delayed[start_idx:end_idx]
                    
                    if len(cycle_data) > 0:
                        # RMS calculation
                        rms = np.sqrt(np.mean(cycle_data**2))
                        
                        # Peak calculation
                        peak = np.max(np.abs(cycle_data))
                        
                        # Crest Factor = Peak / RMS
                        if rms > 0:
                            crest = peak / rms
                        else:
                            crest = 0.0
                        
                        crest_values.append(crest)
            
            crest_values = np.array(crest_values)
            
            logger.info(f"  Calculated {len(crest_values)} Crest values")
            logger.info(f"  Crest range: {crest_values.min():.3f} - {crest_values.max():.3f}")
            
            return crest_values
            
        except Exception as e:
            logger.error(f"Error calculating Crest: {e}")
            raise


class QcontactCalculator(MetricCalculator):
    """Qcontact, dEGGmax and Icontact Calculator"""
    
    def calculate(
        self, 
        egg_signal: np.ndarray, 
        cycle_triggers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Qcontact, dEGGmax and Icontact.
        
        Args:
            egg_signal: Preprocessed EGG signal
            cycle_triggers: Cycle trigger array
        
        Returns:
            Tuple of (qcontact_values, deggmax_values, icontact_values)
        """
        logger.info("Calculating Qcontact, dEGGmax and Icontact...")
        
        try:
            cycle_indices = np.where(cycle_triggers > 0.5)[0]
            qcontact_values = []
            deggmax_values = []
            icontact_values = []
            
            for i in range(len(cycle_indices) - 1):
                start_idx = cycle_indices[i]
                end_idx = cycle_indices[i + 1]
                
                if end_idx - start_idx >= self.min_samples:
                    cycle_data = egg_signal[start_idx:end_idx]
                    
                    if len(cycle_data) > 0:
                        # Calculate max and min within cycle
                        cycle_max = np.max(cycle_data)
                        cycle_min = np.min(cycle_data)
                        peak2peak = cycle_min - cycle_max
                        
                        if peak2peak != 0:
                            # Qcontact calculation
                            integral = (1.0 / peak2peak) * cycle_min
                            qcontact_values.append(integral)
                            
                            # dEGGmax calculation
                            ticks = len(cycle_data)  # Cycle length in samples
                            
                            sin_term = np.sin(2 * np.pi / ticks) if ticks > 0 else 0
                            denominator = peak2peak * (-0.5) * sin_term
                            
                            if abs(denominator) > 1e-10:
                                amp_scale = 1.0 / denominator
                            else:
                                amp_scale = 0.0
                            
                            degg = np.diff(cycle_data)
                            delta = np.max(degg) if len(degg) > 0 else 0
                            
                            deggmax = delta * amp_scale
                            deggmax_values.append(deggmax)
                            
                            # Icontact calculation
                            icontact = np.log10(max(deggmax, 1.0)) * integral
                            icontact_values.append(icontact)
                        else:
                            qcontact_values.append(0.0)
                            deggmax_values.append(0.0)
                            icontact_values.append(0.0)
            
            qcontact_values = np.array(qcontact_values)
            deggmax_values = np.array(deggmax_values)
            icontact_values = np.array(icontact_values)
            
            logger.info(f"  Calculated {len(qcontact_values)} Qcontact values")
            logger.info(f"  Qcontact range: {qcontact_values.min():.3f} - {qcontact_values.max():.3f}")
            logger.info(f"  Calculated {len(deggmax_values)} dEGGmax values")
            logger.info(f"  dEGGmax range: {deggmax_values.min():.3f} - {deggmax_values.max():.3f}")
            logger.info(f"  Calculated {len(icontact_values)} Icontact values")
            logger.info(f"  Icontact range: {icontact_values.min():.3f} - {icontact_values.max():.3f}")
            
            return qcontact_values, deggmax_values, icontact_values
            
        except Exception as e:
            logger.error(f"Error calculating Qcontact/dEGGmax/Icontact: {e}")
            raise


class EntropyCalculator(MetricCalculator):
    """Entropy Calculator (placeholder for future implementation)"""
    
    def calculate(
        self, 
        voice_signal: np.ndarray, 
        cycle_triggers: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Entropy - currently returns zeros.
        
        Args:
            voice_signal: Preprocessed voice signal
            cycle_triggers: Cycle trigger array
        
        Returns:
            Array of Entropy values (all zeros)
        """
        logger.info("Calculating Entropy (placeholder)...")
        
        cycle_indices = np.where(cycle_triggers > 0.5)[0]
        entropy_values = np.zeros(len(cycle_indices) - 1)
        
        logger.info(f"  Calculated {len(entropy_values)} Entropy values (all zeros)")
        logger.info(f"  Entropy range: {entropy_values.min():.3f} - {entropy_values.max():.3f}")
        
        return entropy_values


class HRFCalculator(MetricCalculator):
    """HRF Calculator (placeholder for future implementation)"""
    
    def calculate(
        self, 
        voice_signal: np.ndarray, 
        cycle_triggers: np.ndarray
    ) -> np.ndarray:
        """
        Calculate HRF - currently returns zeros.
        
        Args:
            voice_signal: Preprocessed voice signal
            cycle_triggers: Cycle trigger array
        
        Returns:
            Array of HRF values (all zeros)
        """
        logger.info("Calculating HRF (placeholder)...")
        
        cycle_indices = np.where(cycle_triggers > 0.5)[0]
        hrf_values = np.zeros(len(cycle_indices) - 1)
        
        logger.info(f"  Calculated {len(hrf_values)} HRF values (all zeros)")
        logger.info(f"  HRF range: {hrf_values.min():.3f} - {hrf_values.max():.3f}")
        
        return hrf_values
