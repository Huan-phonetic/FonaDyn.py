# FonaDyn.py

A Python implementation of voice mapping functionality based on FonaDyn, developed and maintained by Sten Ternström.

## Project Structure

```
FonaDyn.py/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── vrp_creator.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── voice_preprocessor.py
│   │   ├── egg_preprocessor.py
│   │   └── cycle_picker.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── voice_metrics.py
│   │   ├── egg_metrics.py
│   │   ├── metrics_calculator.py
│   │   └── cse.py
│   └── __init__.py
├── tests/
├── examples/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── requirements.txt
└── README.md
```

## Features

- Voice signal preprocessing
- EGG signal preprocessing
- Cycle detection and analysis
- Voice metrics calculation:
  - Fundamental frequency (F0)
  - Sound pressure level (SPL)
  - Cepstral peak prominence (CPP)
  - Spectrum balance
  - Crest factor
- EGG metrics calculation:
  - Quasi-closed interval (QCI)
  - Maximum derivative of EGG (dEGGmax)
  - Speed quotient
  - Open quotient
  - Closed quotient
  - Contact quotient
  - Harmonic-to-fundamental ratio (HRF)
  - Sample entropy (CSE)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FonaDyn.py.git
cd FonaDyn.py
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
import librosa
from src.core.vrp_creator import create_VRP_from_Voice_EGG

# Load audio file
audio_file = 'path/to/your/audio.wav'
signal, sr = librosa.load(audio_file, sr=44100, mono=False)

# Create VRP
vrp = create_VRP_from_Voice_EGG(signal, sr)

# Save to CSV
import pandas as pd
pd.DataFrame(vrp).to_csv('output.csv', index=False)
```

## Input/Output

### Input
- `.wav` file with two channels:
  - Channel 1: Voice signal
  - Channel 2: EGG signal

### Output
- CSV file containing the following columns:
  - Midi: MIDI note number
  - dB: Sound pressure level
  - Total: Number of cycles
  - Clarity: F0 clarity
  - Crest: Crest factor
  - SpecBal: Spectrum balance
  - CPP: Cepstral peak prominence
  - Entropy: Sample entropy
  - dEGGmax: Maximum derivative of EGG
  - Qcontact: Contact quotient

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on FonaDyn, developed and maintained by Sten Ternström
- Paper: "FonaDyn — A system for real-time analysis of the electroglottogram, over the voice range" 