# VoiceMap Python Implementation Documentation

## Overview

VoiceMap Python is a complete implementation of VoiceMap algorithms for Voice Range Profile (VRP) analysis. It provides a Python interface to the sophisticated voice analysis algorithms originally implemented in SuperCollider.

## Architecture

### Core Components

1. **VoiceMapAnalyzer** - Main analyzer class that orchestrates the entire analysis process
2. **MetricCalculators** - Individual calculators for each voice metric
3. **VoiceMapConfig** - Configuration management system
4. **Logger** - Structured logging system

### Analysis Pipeline

1. **Audio Loading** - Load and validate audio files
2. **Preprocessing** - Apply filters and signal conditioning
3. **Cycle Detection** - Detect glottal cycles using Dolansky algorithm
4. **Metric Calculation** - Calculate all voice metrics
5. **Filtering** - Apply Clarity threshold filtering
6. **Output Generation** - Generate VRP CSV files

## API Reference

### VoiceMapAnalyzer

Main analyzer class for voice range profile analysis.

#### Methods

- `analyze_and_output_vrp(file_path=None)` - Complete analysis pipeline
- `load_audio(file_path)` - Load audio file
- `preprocess_voice(voice_signal)` - Preprocess voice signal
- `preprocess_egg(egg_signal)` - Preprocess EGG signal
- `peak_follower_cycle_detection(egg_signal)` - Detect cycles

### VoiceMapConfig

Configuration management for all analysis parameters.

#### Key Parameters

- `sample_rate` - Audio sample rate (default: 44100)
- `clarity_threshold` - Clarity filtering threshold (default: 0.96)
- `spl_correction_db` - SPL correction value (default: 120.0)
- `min_samples` - Minimum samples per cycle (default: 20)
- `min_frequency` - Minimum frequency in Hz (default: 50)

### Metric Calculators

Individual calculators for each voice metric:

- **SPLCalculator** - Sound Pressure Level calculation
- **ClarityCalculator** - Cycle-to-cycle correlation
- **CPPCalculator** - Cepstral Peak Prominence
- **SpecBalCalculator** - Spectral Balance
- **CrestCalculator** - Crest Factor
- **QcontactCalculator** - Contact Quotient and related metrics

## Algorithm Details

### Cycle Detection

Uses the Dolansky algorithm with PeakFollower method:

1. Calculate dEGG (derivative of EGG signal)
2. Apply PeakFollower with decay parameter
3. Apply FOS filter
4. Use SetResetFF logic for cycle triggers
5. Filter cycles based on minimum samples and frequency

### SPL Calculation

Sliding window approach:

1. Apply 20ms delay alignment
2. Calculate RMS in 100ms windows with 10ms steps
3. Convert RMS to SPL using 20*log10 formula
4. Assign SPL values to detected cycles

### Clarity Calculation

Cross-correlation based approach:

1. Extract individual cycles
2. Normalize each cycle
3. Calculate cross-correlation between consecutive cycles
4. Use maximum correlation as Clarity value

## Output Format

### VRP CSV Structure

Standard 25-column format:

- **MIDI** - Fundamental frequency in MIDI units
- **dB** - Sound pressure level in dB
- **Total** - Number of data points for this (MIDI, dB) pair
- **Clarity** - Cycle-to-cycle correlation
- **CPP** - Cepstral Peak Prominence
- **SpecBal** - Spectral Balance
- **Crest** - Crest Factor
- **Entropy** - Sample Entropy (currently 0)
- **Qcontact** - Contact Quotient
- **dEGGmax** - Maximum dEGG slope
- **Icontact** - Index of Contacting
- **HRFegg** - Harmonic Richness Factor (currently 0)
- **maxCluster** - Maximum cluster (currently 0)
- **Cluster 1-5** - Cluster values (currently 0)
- **maxCPhon** - Maximum cPhon (currently 0)
- **cPhon 1-5** - cPhon values (currently 0)

## Performance Considerations

- **Memory Usage** - Large audio files may require significant memory
- **Processing Time** - Analysis time scales with audio duration
- **Accuracy** - Results depend on audio quality and EGG signal quality

## Troubleshooting

### Common Issues

1. **Audio File Not Found** - Check file path and permissions
2. **No Cycles Detected** - Verify EGG signal quality
3. **Low Clarity Values** - Check audio quality and cycle detection
4. **Memory Errors** - Reduce audio file size or increase system memory

### Debug Mode

Enable debug logging for detailed analysis information:

```python
from logger import setup_logger
import logging

setup_logger("fonadyn", level=logging.DEBUG)
```
