from voice_preprocess import voice_preprocess
from EGG_process import process_EGG_signal
from cycle_picker import phase_tracker
import numpy as np
from voice_metrics import *
from EGG_metrics import *
import pandas as pd

def get_metrics(signal, sample_rate, n=8096, overlap=6072):
    # first channel is audio, second channel is EGG
    voice = signal[0, :]
    EGG = signal[1, :]

    # preprocess the voice signal
    voice = voice_preprocess(voice, sample_rate)

    # preprocess the EGG signal
    EGG = process_EGG_signal(EGG, sample_rate)

    # segment the EGG signal, this is also used for Audio signal since they are matched
    # segments, starts = get_cycles(EGG, sample_rate)
    segments = phase_tracker(EGG, sample_rate)

    # calculate frame based metrics
    step = n - overlap
    window = np.hanning(n)
    k = n // 2  # k as n/2
    frequencies = []
    clarities = []
    CPPs = []
    SBs = []
    times = []
    SPLs = []

    for start in range(0, len(voice) - n, step):
        segment = voice[start:start + n]
        windowed_segment = segment * window

        f0, clarity = find_f0(windowed_segment, sample_rate, n, k, threshold=0.96, midi=True)
        CPP = find_CPPs(windowed_segment, sample_rate, [60, 880])
        SB = find_SB(windowed_segment, sample_rate)
        SPL = find_SPL(segment)

        frequencies.append(f0)
        SPLs.append(SPL)
        times.append(start / sample_rate)
        clarities.append(clarity)
        CPPs.append(CPP)
        SBs.append(SB)

    # calculate cycle based metrics

    crests = []
    qcis = []
    qdeltas = []
    cses = []

    for start, end in segments:
        voice_segment = voice[start:end]
        EGG_segment = EGG[start:end]
        
        crest = find_crest_factor(voice_segment)
        qci = find_qci(EGG_segment)
        qdelta = find_dEGGmax(EGG_segment)
        # cse = find_CSE(EGG_segment)

        crests.append(crest)
        qcis.append(qci)
        qdeltas.append(qdelta)
        # cses.append(cse)

    # downsample the frame based metrics
    sampled_crests, period_counts = period_downsampling(crests, segments, times, frame_size=2024, sample_rate=sample_rate)
    sampled_qcis, _ = period_downsampling(qcis, segments, times, frame_size=2024, sample_rate=sample_rate)
    sampled_qdeltas, _ = period_downsampling(qdeltas, segments, times, frame_size=2024, sample_rate=sample_rate)

    

    # check all metircs shapes are the same
    assert (len(SPLs) == len(sampled_crests) == len(sampled_qcis) ==
            len(sampled_qdeltas) == len(frequencies) == len(clarities) == len(CPPs) == len(SBs))


    return {
        'frequencies': frequencies,
        'SPLs': SPLs,
        'Total': period_counts,
        'clarities': clarities,
        'crests': sampled_crests,
        'SBs': SBs,
        'CPPs': CPPs,
        'qdeltas': sampled_qdeltas,
        'qcis': sampled_qcis,
        # 'cses': sampled_cses
    }

def post_process_metrics(metrics):
    # Step 1: convert lists to numpy arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key])

    # Step 2: remove items with zero frequencies and zeor SPLs
    valid_mask = (metrics['frequencies'] != 0) & (metrics['SPLs'] != 0) & (metrics['Total'] != 0)

    for key in metrics:
        metrics[key] = metrics[key][valid_mask]

    # Step 3: Merge the metrics by integers
    merged_metrics = {}
    for f in range(0, 200):
        for spl in range(0, 200):
            mask = (metrics['frequencies'] >= f-0.5) & (metrics['frequencies'] < f + 0.5) & \
                   (metrics['SPLs'] >= spl-0.5) & (metrics['SPLs'] < spl + 0.5)
            if np.sum(mask) > 0:
                for key in metrics:
                    if key not in merged_metrics:
                        merged_metrics[key] = []
                    masked_metrics = metrics[key][mask]
                    if key == 'Total':
                        merged_metrics[key].append(np.sum(masked_metrics))  # Sum for 'Total'
                    else:
                        merged_metrics[key].append(np.mean(masked_metrics))  # Mean for other keys
                merged_metrics['frequencies'][-1] = f  # Store frequency bin
                merged_metrics['SPLs'][-1] = spl        # Store SPL level bin


    return merged_metrics

def main():
    audio_file = 'audio/test_Voice_EGG.wav'
    signal, sr = librosa.load(audio_file, sr=44100, mono=False)
    metrics = get_metrics(signal, sr)
    metrics = post_process_metrics(metrics)
    # save to a csv file
    df = pd.DataFrame(metrics)
    df.to_csv('FonaPy.csv', index=False)

if __name__ == '__main__':

    main()