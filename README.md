# VoiceMap Python

Python implementation of VoiceMap algorithms for Voice Range Profile (VRP) analysis.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python main.py

# Run with custom audio file
python main.py path/to/audio.wav
```

## Usage

```python
from analyzer import VoiceMapAnalyzer

# Basic usage
analyzer = VoiceMapAnalyzer()
data, output_file = analyzer.analyze_and_output_vrp("audio.wav")

# Custom configuration
from config import VoiceMapConfig
config = VoiceMapConfig(clarity_threshold=0.95, output_dir="results")
analyzer = VoiceMapAnalyzer(config)
```

## Analysis Logic

The VoiceMap analysis follows a 7-step process to extract voice metrics from audio recordings:

### 1. Audio Loading (`VoiceMapAnalyzer.load_audio()`)
- Loads stereo audio file using `soundfile.read()`
- Extracts voice channel (left) and EGG channel (right)
- Returns sample rate and duration information

### 2. Signal Preprocessing
- **Voice Signal** (`VoiceMapAnalyzer.preprocess_voice()`): Applies 30Hz high-pass filter using `scipy.signal.butter()` and `filtfilt()`
- **EGG Signal** (`VoiceMapAnalyzer.preprocess_egg()`): Applies 100Hz HPF + 10kHz LPF + 9-point median filter

### 3. Cycle Detection (`VoiceMapAnalyzer.peak_follower_cycle_detection()`)
- Uses PeakFollower method with Dolansky algorithm
- Calculates dEGG (derivative of EGG signal)
- Applies `dolansky_algorithm()` → `peak_follower()` → `fos_filter()` → `set_reset_ff()`
- Filters cycles based on minimum frequency (50Hz) and period constraints

### 4. Metric Calculation (`VoiceMapAnalyzer.calculate_all_metrics()`)
Calculates 9 voice metrics using specialized calculator classes:
- **MIDI & Clarity** (`ClarityCalculator.calculate()`): Fundamental frequency detection using autocorrelation
- **SPL** (`SPLCalculator.calculate()`): Sound pressure level using sliding window RMS
- **CPP** (`CPPCalculator.calculate()`): Cepstral peak prominence using FFT and cepstrum analysis
- **SpecBal** (`SpecBalCalculator.calculate()`): Spectral balance between 1500-2000Hz bands
- **Crest** (`CrestCalculator.calculate()`): Crest factor (peak-to-RMS ratio)
- **Qcontact, dEGGmax, Icontact** (`QcontactCalculator.calculate()`): EGG-based contact quotient metrics
- **Entropy** (`EntropyCalculator.calculate()`): Spectral entropy (currently placeholder)
- **HRFegg** (`HRFCalculator.calculate()`): Harmonic richness factor (currently placeholder)

### 5. Quality Filtering (`VoiceMapAnalyzer.apply_clarity_filtering()`)
- Filters out data points below clarity threshold (default: 0.96)
- Removes low-quality voice segments

### 6. Data Aggregation (`VoiceMapAnalyzer.output_vrp_csv()`)
- Applies SPL correction (+120dB)
- Rounds MIDI and dB values to integers
- Filters to VoiceMap standard ranges (MIDI: 30-96, SPL: 40-120dB)
- Groups by (MIDI, dB) pairs and averages other metrics
- Sums Total column for each group

### 7. Output Generation
- Creates 25-column VRP CSV format (with EGG clusters and phonation clusters on hold)
- Saves timestamped file to `result/` directory
- Displays comprehensive statistics

## Features

- **9 Voice Metrics**: MIDI, SPL, Clarity, CPP, SpecBal, Crest, Qcontact, dEGGmax, Icontact
- **Cycle Detection**: PeakFollower with Dolansky algorithm
- **Standard Output**: 25-column VRP CSV format
- **Configurable**: Adjustable thresholds and parameters

## Output

Results saved to `result/` directory as timestamped CSV files:
- `complete_vrp_results_YYYYMMDD_HHMMSS_VRP.csv`

## Testing

```bash
python -m unittest tests.test_voicemap
```

## Requirements

- Python 3.7+
- numpy, scipy, librosa, pandas, soundfile