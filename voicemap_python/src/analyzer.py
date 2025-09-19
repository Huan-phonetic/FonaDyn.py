#!/usr/bin/env python3
"""
VoiceMap Voice Range Profile Analyzer
Complete implementation of VoiceMap algorithms for VRP analysis
"""

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, filtfilt, medfilt
import os
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

from config import VoiceMapConfig, DEFAULT_CONFIG
from logger import setup_logger, get_logger
from metrics import (
    SPLCalculator, ClarityCalculator, CPPCalculator, SpecBalCalculator,
    CrestCalculator, QcontactCalculator, EntropyCalculator, HRFCalculator
)

logger = get_logger(__name__)


class VoiceMapAnalyzer:
    """VoiceMap Voice Range Profile Analyzer"""
    
    def __init__(self, config: Optional[VoiceMapConfig] = None):
        """
        Initialize VoiceMap analyzer.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or DEFAULT_CONFIG
        self.logger = get_logger(__name__)
        
        # Initialize metric calculators
        self.spl_calculator = SPLCalculator(self.config)
        self.clarity_calculator = ClarityCalculator(self.config)
        self.cpp_calculator = CPPCalculator(self.config)
        self.specbal_calculator = SpecBalCalculator(self.config)
        self.crest_calculator = CrestCalculator(self.config)
        self.qcontact_calculator = QcontactCalculator(self.config)
        self.entropy_calculator = EntropyCalculator(self.config)
        self.hrf_calculator = HRFCalculator(self.config)
        
        self.logger.info("VoiceMap analyzer initialized")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        Load audio file and extract voice and EGG channels.
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Tuple of (voice_signal, egg_signal, sample_rate, duration)
        """
        self.logger.info(f"Loading audio file: {file_path}")
        
        try:
            signal, sr = sf.read(file_path)
            
            if signal.ndim == 2:
                voice = signal[:, 0]
                egg = signal[:, 1]
            else:
                voice = signal
                egg = None
                
            duration = len(signal) / sr
            
            self.logger.info(f"Audio duration: {duration:.1f} seconds")
            self.logger.info(f"Sample rate: {sr} Hz")
            self.logger.info(f"Total samples: {len(signal):,}")
            
            return voice, egg, sr, duration
            
        except Exception as e:
            self.logger.error(f"Error loading audio file: {e}")
            raise
    
    def preprocess_voice(self, voice_signal: np.ndarray) -> np.ndarray:
        """
        Preprocess voice signal with 30Hz high-pass filter.
        
        Args:
            voice_signal: Raw voice signal
        
        Returns:
            Preprocessed voice signal
        """
        try:
            nyquist = self.config.sample_rate / 2
            low_cutoff = 30 / nyquist
            b, a = butter(2, low_cutoff, btype='high')
            voice_hp = filtfilt(b, a, voice_signal)
            return voice_hp
        except Exception as e:
            self.logger.error(f"Error preprocessing voice signal: {e}")
            raise
    
    def preprocess_egg(self, egg_signal: np.ndarray) -> np.ndarray:
        """
        Preprocess EGG signal with 100Hz HPF + 10kHz LPF + median filter.
        
        Args:
            egg_signal: Raw EGG signal
        
        Returns:
            Preprocessed EGG signal
        """
        try:
            # 100Hz high-pass filter
            nyquist = self.config.sample_rate / 2
            low_cutoff = 100 / nyquist
            b, a = butter(2, low_cutoff, btype='high')
            egg_hp = filtfilt(b, a, egg_signal)
            
            # 10kHz low-pass filter
            high_cutoff = 10000 / nyquist
            b, a = butter(2, high_cutoff, btype='low')
            egg_lp = filtfilt(b, a, egg_hp)
            
            # Median filter (9-point median filter)
            egg_median = medfilt(egg_lp, kernel_size=9)
            
            return egg_median
        except Exception as e:
            self.logger.error(f"Error preprocessing EGG signal: {e}")
            raise
    
    def peak_follower_cycle_detection(self, egg_signal: np.ndarray) -> np.ndarray:
        """
        Perform cycle detection using PeakFollower method.
        
        Args:
            egg_signal: Preprocessed EGG signal
        
        Returns:
            Array of cycle triggers
        """
        self.logger.info("Using PeakFollower method for cycle detection...")
        
        try:
            # Calculate dEGG (derivative of EGG)
            degg = np.diff(egg_signal)
            
            # Dolansky algorithm
            cycle_triggers = self.dolansky_algorithm(
                degg, 
                self.config.dolansky_decay, 
                self.config.dolansky_coeff
            )
            
            # Ensure length match
            cycle_triggers = np.concatenate([cycle_triggers, [0]])
            
            # Filter cycles
            filtered_triggers = self.filter_cycles(cycle_triggers)
            
            return filtered_triggers
        except Exception as e:
            self.logger.error(f"Error in cycle detection: {e}")
            raise
    
    def filter_cycles(self, cycle_triggers: np.ndarray) -> np.ndarray:
        """
        Filter cycles according to FonaDyn standards.
        
        Args:
            cycle_triggers: Raw cycle triggers
        
        Returns:
            Filtered cycle triggers
        """
        self.logger.info("  Filtering cycles...")
        
        try:
            trigger_indices = np.where(cycle_triggers > 0.5)[0]
            
            if len(trigger_indices) < 2:
                return cycle_triggers
            
            periods = np.diff(trigger_indices)
            
            valid_periods = []
            valid_indices = [trigger_indices[0]]
            
            for i, period in enumerate(periods):
                if (period >= self.config.min_samples and 
                    period <= self.config.max_period_samples):
                    valid_periods.append(period)
                    valid_indices.append(trigger_indices[i + 1])
            
            filtered_triggers = np.zeros_like(cycle_triggers)
            filtered_triggers[valid_indices] = 1
            
            self.logger.info(f"  Original cycles: {len(trigger_indices):,}")
            self.logger.info(f"  Filtered cycles: {len(valid_indices):,}")
            self.logger.info(f"  Filter ratio: {len(valid_indices)/len(trigger_indices)*100:.1f}%")
            
            return filtered_triggers
        except Exception as e:
            self.logger.error(f"Error filtering cycles: {e}")
            raise
    
    def dolansky_algorithm(
        self, 
        signal: np.ndarray, 
        decay: float, 
        coeff: float
    ) -> np.ndarray:
        """
        Implement Dolansky algorithm for cycle detection.
        
        Args:
            signal: Input signal
            decay: Decay parameter
            coeff: Coefficient parameter
        
        Returns:
            Cycle triggers
        """
        try:
            peak_plus = self.peak_follower(np.maximum(signal, 0), decay)
            peak_minus = self.peak_follower(np.maximum(-signal, 0), decay)
            
            peak_plus_fos = self.fos_filter(peak_plus, coeff)
            peak_minus_fos = self.fos_filter(peak_minus, coeff)
            
            cycle_triggers = self.set_reset_ff(peak_plus_fos, peak_minus_fos)
            
            return cycle_triggers
        except Exception as e:
            self.logger.error(f"Error in Dolansky algorithm: {e}")
            raise
    
    def peak_follower(self, signal: np.ndarray, decay: float) -> np.ndarray:
        """
        Implement PeakFollower algorithm.
        
        Args:
            signal: Input signal
            decay: Decay parameter
        
        Returns:
            Peak follower output
        """
        output = np.zeros_like(signal)
        peak = 0
        
        for i in range(len(signal)):
            if signal[i] > peak:
                peak = signal[i]
            else:
                peak *= decay
            output[i] = peak
        
        return output
    
    def fos_filter(self, signal: np.ndarray, coeff: float) -> np.ndarray:
        """
        Implement FOS filter.
        
        Args:
            signal: Input signal
            coeff: Coefficient parameter
        
        Returns:
            Filtered signal
        """
        output = np.zeros_like(signal)
        
        for i in range(1, len(signal)):
            output[i] = coeff * output[i-1] + coeff * signal[i] - coeff * signal[i-1]
        
        return output
    
    def set_reset_ff(
        self, 
        set_signal: np.ndarray, 
        reset_signal: np.ndarray
    ) -> np.ndarray:
        """
        Implement SetResetFF logic.
        
        Args:
            set_signal: Set signal
            reset_signal: Reset signal
        
        Returns:
            Set-reset flip-flop output
        """
        output = np.zeros_like(set_signal)
        state = 0
        
        for i in range(len(set_signal)):
            if set_signal[i] > reset_signal[i]:
                state = 1
            elif reset_signal[i] > set_signal[i]:
                state = 0
            output[i] = state
        
        return output
    
    def calculate_all_metrics(
        self, 
        voice_signal: np.ndarray, 
        egg_signal: np.ndarray, 
        cycle_triggers: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all metrics using modular calculators.
        
        Args:
            voice_signal: Preprocessed voice signal
            egg_signal: Preprocessed EGG signal
            cycle_triggers: Cycle trigger array
        
        Returns:
            Dictionary of metric arrays
        """
        self.logger.info("Calculating all metrics...")
        
        try:
            # Calculate each metric using its calculator
            spl_values = self.spl_calculator.calculate(voice_signal, cycle_triggers)
            midi_values, clarity_values = self.clarity_calculator.calculate(voice_signal, cycle_triggers)
            cpp_values = self.cpp_calculator.calculate(voice_signal, cycle_triggers)
            specbal_values = self.specbal_calculator.calculate(voice_signal, cycle_triggers)
            crest_values = self.crest_calculator.calculate(voice_signal, cycle_triggers)
            qcontact_values, deggmax_values, icontact_values = self.qcontact_calculator.calculate(egg_signal, cycle_triggers)
            entropy_values = self.entropy_calculator.calculate(voice_signal, cycle_triggers)
            hrf_values = self.hrf_calculator.calculate(voice_signal, cycle_triggers)
            
            return {
                'midi': midi_values,
                'spl': spl_values,
                'clarity': clarity_values,
                'cpp': cpp_values,
                'specbal': specbal_values,
                'crest': crest_values,
                'qcontact': qcontact_values,
                'deggmax': deggmax_values,
                'icontact': icontact_values,
                'entropy': entropy_values,
                'hrf': hrf_values
            }
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            raise
    
    def apply_clarity_filtering(self, metrics: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply Clarity threshold filtering to metrics.
        
        Args:
            metrics: Dictionary of metric arrays
        
        Returns:
            Filtered metrics dictionary
        """
        self.logger.info(f"Applying Clarity threshold (threshold: {self.config.clarity_threshold})...")
        
        try:
            clarity_mask = metrics['clarity'] >= self.config.clarity_threshold
            
            total_points = len(metrics['midi'])
            filtered_points = np.sum(clarity_mask)
            filter_ratio = filtered_points / total_points * 100 if total_points > 0 else 0
            
            self.logger.info(f"  Total data points: {total_points:,}")
            self.logger.info(f"  Clarity >= {self.config.clarity_threshold} data points: {filtered_points:,}")
            self.logger.info(f"  Filter ratio: {filter_ratio:.1f}%")
            
            # Apply filtering to all metrics
            filtered_metrics = {}
            for key, values in metrics.items():
                if key == 'midi':
                    # Filter NaN values first
                    valid_mask = ~np.isnan(values)
                    filtered_values = values[valid_mask]
                    # Then apply clarity filtering
                    filtered_metrics[key] = filtered_values[clarity_mask[valid_mask]]
                else:
                    # Apply clarity filtering directly
                    # Ensure clarity_mask matches the length of values
                    if len(clarity_mask) == len(values):
                        filtered_metrics[key] = values[clarity_mask]
                    else:
                        # If lengths don't match, use the shorter length
                        min_length = min(len(clarity_mask), len(values))
                        filtered_metrics[key] = values[:min_length][clarity_mask[:min_length]]
            
            return filtered_metrics
        except Exception as e:
            self.logger.error(f"Error applying clarity filtering: {e}")
            raise
    
    def output_vrp_csv(
        self, 
        metrics: Dict[str, np.ndarray], 
        cycle_count: int, 
        duration: float
    ) -> str:
        """
        Output standard VRP CSV file.
        
        Args:
            metrics: Dictionary of filtered metric arrays
            cycle_count: Total number of detected cycles
            duration: Audio duration in seconds
        
        Returns:
            Path to output CSV file
        """
        self.logger.info("Outputting VRP CSV file...")
        
        try:
            # SPL correction: +120dB
            spl_corrected = metrics['spl'] + self.config.spl_correction_db
            
            # Create DataFrame
            df = pd.DataFrame({
                'MIDI': metrics['midi'],
                'dB': spl_corrected,
                'Total': 1,  # Each data point counts as 1
                'Clarity': metrics['clarity'],
                'CPP': metrics['cpp'],
                'SpecBal': metrics['specbal'],
                'Crest': metrics['crest'],
                'Entropy': metrics['entropy'],  # Currently zeros
                'Qcontact': metrics['qcontact'],
                'dEGGmax': metrics['deggmax'],
                'Icontact': metrics['icontact'],
                'HRFegg': metrics['hrf']  # Currently zeros
            })
            
            # Apply rounding: round to nearest integer
            df['MIDI'] = df['MIDI'].apply(lambda x: round(x) if x > 0 else 0)
            df['dB'] = df['dB'].apply(lambda x: round(x) if x > 0 else 0)
            
            self.logger.info(f"  Original data points: {len(df):,}")
            
            # Apply FonaDyn standard range filtering
            range_mask = (
                (df['MIDI'] >= self.config.n_min_midi) & (df['MIDI'] <= self.config.n_max_midi) &
                (df['dB'] >= self.config.n_min_spl) & (df['dB'] <= self.config.n_max_spl)
            )
            
            df_filtered = df[range_mask].copy()
            self.logger.info(f"  Range filtered: {len(df_filtered):,}")
            self.logger.info(f"  Filtered out: {len(df) - len(df_filtered):,} data points")
            
            # Group by (MIDI, dB), aggregate other metrics and sum Total
            grouped = df_filtered.groupby(['MIDI', 'dB']).agg({
                'Clarity': 'mean',
                'CPP': 'mean', 
                'SpecBal': 'mean',
                'Crest': 'mean',
                'Entropy': 'mean',
                'Qcontact': 'mean',
                'dEGGmax': 'mean',
                'Icontact': 'mean',
                'HRFegg': 'mean',
                'Total': 'sum'  # Sum Total column
            }).reset_index()
            
            # Add standard VRP format complete column structure
            standard_columns = [
                'MIDI','dB','Total','Clarity','Crest','SpecBal','CPP','Entropy',
                'dEGGmax','Qcontact','Icontact','HRFegg','maxCluster',
                'Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5',
                'maxCPhon','cPhon 1','cPhon 2','cPhon 3','cPhon 4','cPhon 5'
            ]
            
            # Add 0 values for missing columns
            for col in standard_columns:
                if col not in grouped.columns:
                    grouped[col] = 0
            
            # Reorder columns to match standard format
            grouped = grouped[standard_columns]
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create result directory (if doesn't exist)
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Save aggregated results to result folder
            output_file = f"{self.config.output_dir}/complete_vrp_results_{timestamp}_VRP.csv"
            grouped.to_csv(output_file, index=False)
            self.logger.info(f"  Results saved to: {output_file}")
            
            # Display statistics
            self.logger.info("=== VRP Results Statistics ===")
            self.logger.info(f"Unique (MIDI,dB) pairs: {len(grouped):,}")
            self.logger.info(f"Total data points: {grouped['Total'].sum():,}")
            self.logger.info(f"MIDI average: {grouped['MIDI'].mean():.1f}")
            self.logger.info(f"SPL average: {grouped['dB'].mean():.1f} dB")
            self.logger.info(f"Clarity average: {grouped['Clarity'].mean():.3f}")
            self.logger.info(f"CPP average: {grouped['CPP'].mean():.3f}")
            self.logger.info(f"SpecBal average: {grouped['SpecBal'].mean():.3f}")
            self.logger.info(f"Crest average: {grouped['Crest'].mean():.3f}")
            self.logger.info(f"Entropy average: {grouped['Entropy'].mean():.3f}")
            self.logger.info(f"Qcontact average: {grouped['Qcontact'].mean():.3f}")
            self.logger.info(f"dEGGmax average: {grouped['dEGGmax'].mean():.3f}")
            self.logger.info(f"Icontact average: {grouped['Icontact'].mean():.3f}")
            self.logger.info(f"HRFegg average: {grouped['HRFegg'].mean():.3f}")
            
            return output_file
        except Exception as e:
            self.logger.error(f"Error outputting VRP CSV: {e}")
            raise
    
    def analyze_and_output_vrp(self, file_path: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], str]:
        """
        Complete analysis and output VRP CSV.
        
        Args:
            file_path: Path to audio file. If None, uses config default.
        
        Returns:
            Tuple of (filtered_metrics, output_file_path)
        """
        self.logger.info("=" * 60)
        self.logger.info("VoiceMap Complete Analysis")
        self.logger.info("=" * 60)
        
        try:
            # Use provided file path or default from config
            audio_file = file_path or self.config.audio_file
            
            # 1. Load audio
            voice, egg, sr, duration = self.load_audio(audio_file)
            
            # 2. Preprocessing
            voice_processed = self.preprocess_voice(voice)
            egg_processed = self.preprocess_egg(egg)
            
            # 3. Cycle detection
            self.logger.info("Cycle Detection...")
            cycle_triggers = self.peak_follower_cycle_detection(egg_processed)
            cycle_count = np.sum(cycle_triggers > 0.5)
            self.logger.info(f"Detected cycles: {cycle_count:,}")
            
            # 4. Calculate all metrics
            metrics = self.calculate_all_metrics(voice_processed, egg_processed, cycle_triggers)
            
            # 5. Apply Clarity filtering
            filtered_metrics = self.apply_clarity_filtering(metrics)
            
            # 6. Display filtered results
            self.logger.info(f"Valid data points: {len(filtered_metrics['midi']):,}")
            self.logger.info(f"SPL range (original): {filtered_metrics['spl'].min():.1f} - {filtered_metrics['spl'].max():.1f} dB")
            self.logger.info(f"SPL range (corrected +{self.config.spl_correction_db}dB): {(filtered_metrics['spl'] + self.config.spl_correction_db).min():.1f} - {(filtered_metrics['spl'] + self.config.spl_correction_db).max():.1f} dB")
            self.logger.info(f"MIDI range: {filtered_metrics['midi'].min():.1f} - {filtered_metrics['midi'].max():.1f}")
            self.logger.info(f"Clarity range: {filtered_metrics['clarity'].min():.3f} - {filtered_metrics['clarity'].max():.3f}")
            self.logger.info(f"CPP range: {filtered_metrics['cpp'].min():.3f} - {filtered_metrics['cpp'].max():.3f}")
            self.logger.info(f"SpecBal range: {filtered_metrics['specbal'].min():.3f} - {filtered_metrics['specbal'].max():.3f}")
            self.logger.info(f"Crest range: {filtered_metrics['crest'].min():.3f} - {filtered_metrics['crest'].max():.3f}")
            self.logger.info(f"Qcontact range: {filtered_metrics['qcontact'].min():.3f} - {filtered_metrics['qcontact'].max():.3f}")
            self.logger.info(f"dEGGmax range: {filtered_metrics['deggmax'].min():.3f} - {filtered_metrics['deggmax'].max():.3f}")
            self.logger.info(f"Icontact range: {filtered_metrics['icontact'].min():.3f} - {filtered_metrics['icontact'].max():.3f}")
            self.logger.info(f"Entropy range: {filtered_metrics['entropy'].min():.3f} - {filtered_metrics['entropy'].max():.3f}")
            self.logger.info(f"HRF range: {filtered_metrics['hrf'].min():.3f} - {filtered_metrics['hrf'].max():.3f}")
            
            # 7. Output VRP CSV file
            output_file = self.output_vrp_csv(filtered_metrics, cycle_count, duration)
            
            return filtered_metrics, output_file
        except Exception as e:
            self.logger.error(f"Error in complete analysis: {e}")
            raise


def main():
    """Main function"""
    # Set up logging
    setup_logger("voicemap", level=logging.INFO)
    logger = get_logger(__name__)
    
    logger.info("VoiceMap Complete Analysis Script")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Create analyzer with default configuration
        analyzer = VoiceMapAnalyzer()
        
        # Analyze and output VRP
        data, output_file = analyzer.analyze_and_output_vrp()
        
        logger.info("=" * 60)
        logger.info("VoiceMap Complete Analysis Finished!")
        logger.info("=" * 60)
        logger.info(f"Output file: {output_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
