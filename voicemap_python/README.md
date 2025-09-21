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