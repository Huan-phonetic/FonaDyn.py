"""
VRP creator: output long-form CSV per (frequency, SPL) bin with metric means.

- Input stereo file: channel 0 = voice, channel 1 = EGG
- Pipeline:
  1) compute metrics (frame + cycle)
  2) post-process and bin into frequency/SPL bins
  3) output long-form CSV with columns:
     frequencies,SPLs,Total,clarities,crests,SBs,CPPs,qdeltas,qcis
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf

# Support both `python -m src.core.vrp_creator` and direct script execution
try:
    from ..metrics.metrics_calculator import get_metrics, post_process_metrics
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.metrics.metrics_calculator import get_metrics, post_process_metrics


def build_vrp_long_dataframe(metrics_binned: dict) -> pd.DataFrame:
    """Create long-form dataframe aligned with expected output columns."""
    df = pd.DataFrame(metrics_binned)
    # Ensure column order
    cols = ['frequencies', 'SPLs', 'Total', 'clarities', 'crests', 'SBs', 'CPPs', 'qdeltas', 'qcis']
    df = df[cols]
    df = df.sort_values(['frequencies', 'SPLs']).reset_index(drop=True)
    return df


def create_vrp_long_from_file(audio_path: str, output_csv: str) -> pd.DataFrame:
    """Compute metrics and write long-form VRP CSV to output_csv. Returns the DataFrame."""
    stereo, sr = sf.read(audio_path)
    # Ensure shape is (2, N)
    if stereo.ndim != 2 or stereo.shape[1] < 2:
        raise ValueError('Expected a stereo file with voice+EGG channels')
    data = stereo.T[:2, :]

    metrics = get_metrics(data, sr)
    metrics_binned = post_process_metrics(metrics)
    table = build_vrp_long_dataframe(metrics_binned)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    table.to_csv(output_csv, index=False, float_format='%.6g')
    return table


if __name__ == '__main__':
    # Example run: write the expected long-form CSV
    audio_path = 'audio/test_Voice_EGG.wav'
    out_dir = 'results'
    out_long = os.path.join(out_dir, 'VRP_long.csv')

    table = create_vrp_long_from_file(audio_path, out_long)
    print(f'Wrote long-form VRP metrics to: {out_long} (rows={len(table)})')