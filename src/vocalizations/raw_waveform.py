# This script gets the audio files and gets the pure waveform.
import wave
import pickle
import os
import numpy as np

import matplotlib.pyplot as plt

PATH_TO_AUDIO = 'copora/Vocalizations'


# for file_name in os.listdir(''):

def get_waveform(path_to_file):

    spf = wave.open(path_to_file,'r')

    #Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')


    #If Stereo
    if spf.getnchannels() == 2:
        print 'Just mono files'
        sys.exit(0)

    plt.title('Signal Wave...')
    plt.plot(signal)
    plt.show()

get_waveform('/Users/sam_coope/Documents/Programming/speech-ml/corpora/Vocalizations/42_fear.wav')
