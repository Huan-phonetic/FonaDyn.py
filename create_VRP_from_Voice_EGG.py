'''
This script is used to create a Voice Map from the Voice EGG data. The Voice Map method is based on FonaDyn, developed and maintained by Sten Ternström.
It's a simplified version of supercollider code. It would only produce a VRP.csv file in a format that can be used in FonaDyn.
The VRP.csv file contains the following columns:
Midi, dB, Total, Clarity, Crest, SpecBal, CPP, Entropy, dEGGmax, Qcontact, (the rest colomns though used in Fonadyn are not extracted in this script)

The script is based on the following paper:
FonaDyn — A system for real-time analysis of the electroglottogram, over the voice range

Input: Voice_EGG.wav
Output: VRP.csv
'''

import numpy as np
import librosa

def create_VRP_from_Voice_EGG(signal, sr):
    # first channel is audio, second channel is EGG
    voice = signal[:, 0]
    EGG = signal[:, 1]

    # process the voice signal
    voice_metrics = get_audio_metrics(voice, sr)

    # process the EGG signal
    EGG_metrics = get_EGG_metrics(EGG, sr)

    # combine the voice and EGG metrics, check


