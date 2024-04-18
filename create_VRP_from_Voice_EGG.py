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

