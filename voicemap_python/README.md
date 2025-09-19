# VoiceMap Python Implementation

Complete implementation of VoiceMap algorithms for Voice Range Profile (VRP) analysis.

## Project Structure

```
voicemap_python/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── analyzer.py        # Main analyzer class
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging configuration
│   └── metrics.py         # Metric calculators
├── tests/                  # Test files
├── docs/                   # Documentation
├── data/                   # Data files
├── examples/               # Usage examples
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.7 or higher

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- numpy>=1.21.0
- scipy>=1.7.0
- librosa>=0.9.0
- pandas>=1.3.0
- soundfile>=0.10.3

## Usage

### Quick Start
```bash
# Run analysis with default audio file
python main.py

# Run analysis with custom audio file
python main.py path/to/your/audio.wav
```

### Programmatic Usage

#### Basic Usage
```python
import sys
sys.path.insert(0, 'src')

from analyzer import FonaDynAnalyzer

# Create analyzer with default configuration
analyzer = FonaDynAnalyzer()

# Run analysis
data, output_file = analyzer.analyze_and_output_vrp("path/to/audio.wav")

print(f"Analysis complete! Output: {output_file}")
print(f"Data points: {len(data['midi']):,}")
```

#### Custom Configuration
```python
from analyzer import VoiceMapAnalyzer
from config import VoiceMapConfig

# Create custom configuration
config = VoiceMapConfig(
    clarity_threshold=0.95,      # Clarity threshold
    spl_correction_db=120.0,     # SPL correction
    output_dir="my_results"      # Output directory
)

# Create analyzer with custom config
analyzer = VoiceMapAnalyzer(config)

# Run analysis
data, output_file = analyzer.analyze_and_output_vrp()
```

#### Data Analysis
```python
# Analyze the results
print(f"MIDI range: {data['midi'].min():.1f} - {data['midi'].max():.1f}")
print(f"SPL range: {data['spl'].min():.1f} - {data['spl'].max():.1f} dB")
print(f"Clarity mean: {data['clarity'].mean():.3f}")
print(f"CPP mean: {data['cpp'].mean():.3f}")
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 44100 | Audio sample rate |
| `clarity_threshold` | 0.96 | Clarity filtering threshold |
| `spl_correction_db` | 120.0 | SPL correction value |
| `min_samples` | 20 | Minimum samples per cycle |
| `min_frequency` | 50 | Minimum frequency (Hz) |
| `dolansky_decay` | 0.99 | Dolansky algorithm decay |
| `dolansky_coeff` | 0.995 | Dolansky algorithm coefficient |
| `n_min_midi` | 30 | Minimum MIDI value |
| `n_max_midi` | 96 | Maximum MIDI value |
| `n_min_spl` | 40 | Minimum SPL value |
| `n_max_spl` | 120 | Maximum SPL value |

## Features

### Implemented Metrics (9 parameters)
1. **MIDI** - Fundamental frequency (F0)
2. **SPL** - Sound Pressure Level (+120dB correction)
3. **Clarity** - Cycle-to-cycle correlation
4. **CPP** - Cepstral Peak Prominence
5. **SpecBal** - Spectral Balance
6. **Crest** - Crest Factor
7. **Qcontact** - Contact Quotient
8. **dEGGmax** - Maximum dEGG slope
9. **Icontact** - Index of Contacting

### Key Features
- ✅ Strictly follows FonaDyn SuperCollider algorithms
- ✅ PeakFollower cycle detection with Dolansky algorithm
- ✅ Sliding window SPL calculation (100ms windows)
- ✅ Cross-correlation based Clarity calculation
- ✅ Standard VRP CSV output format
- ✅ Configurable Clarity threshold (default: 0.96)
- ✅ MIDI/SPL rounding and range filtering
- ✅ Modular design for easy metric extension
- ✅ Type annotations and comprehensive error handling
- ✅ Centralized configuration management
- ✅ Structured logging system

### Output
- Generates timestamped VRP CSV files in `result/` directory
- Standard 25-column VRP format compatible with FonaDyn
- Aggregated data grouped by (MIDI, dB) pairs
- Total count per (MIDI, dB) combination

## Examples

See the `examples/` directory for detailed usage examples:
- `examples.py` - Comprehensive usage examples

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
This project follows PEP 8 style guidelines.

## Notes
- Entropy and HRF are currently set to 0.0 (placeholders for future implementation)
- All algorithms based on VoiceMap SuperCollider source code analysis
- Tested with `test_Voice_EGG.wav` and compared against `test_Log.csv`
