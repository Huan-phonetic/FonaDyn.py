# VoiceMap Test Framework

Complete VRP analysis framework with all voice metrics.

## Quick Start

```bash
# Run complete VRP analysis
python full_vrp.py

# Run MIDI-SPL only
python only_midi_spl.py
```

## Features

- **Complete Metrics**: MIDI, SPL, Clarity, CPP, SpecBal, Crest, Qcontact, dEGGmax
- **Fixed Window Analysis**: 1024 samples (~23ms at 44.1kHz)
- **Standard Ranges**: MIDI 30-96, SPL 40-120
- **CSV Output**: Standard VRP format

## Output Files

- `complete_vrp_results_YYYYMMDD_HHMMSS_VRP.csv` - Complete analysis
- `midi_spl_VRP.csv` - MIDI-SPL only

## Key Parameters

- **Window Size**: 1024 samples
- **Hop Size**: 512 samples (50% overlap)
- **Preprocessing**: 30Hz high-pass filter
- **SPL Reference**: 20e-6 Pa
- **Outlier Filter**: MIDI 30-80, SPL 40-100

## Results Example

```
Data points: 5936
MIDI range: 45 - 71
SPL range: 40 - 95
Clarity range: 0.111 - 0.875
CPP range: 12.104 - 35.095
```