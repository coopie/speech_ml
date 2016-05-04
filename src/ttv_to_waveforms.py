# Provides methods to get the raw waveforms from a TTV.yaml and normalise them
# TODO: test this

import numpy as np
import h5py
import os
from scipy.io.wavfile import read
from util import mkdir_p, get_emotion_number_from_filename

def read_wav_file(path_to_wav_file):
    return read(path_to_wav_file)[1]

def noop():
    return x

def ttv_to_waveforms(ttv_info, normaliser=lambda x: x, get_waveform_data=read_wav_file):
    """
    Returns each ttv_field as a dictionary {x:, y:} (explain more)
    """
    ttv_info = (ttv_info['test'], ttv_info['train'], ttv_info['validation'])
    wavfiles = map(lambda info_set: map(get_waveform_data, info_set), ttv_info)
    wavfiles = map(lambda info_set: map(normaliser, info_set), wavfiles)

    emotions = map(lambda info_set: map(get_emotion_number_from_filename, info_set), ttv_info)

    return (
    {'x': np.array(wavfiles[0]), 'y': np.array(emotions[0])},
    {'x': np.array(wavfiles[1]), 'y': np.array(emotions[1])},
    {'x': np.array(wavfiles[2]), 'y': np.array(emotions[2])},
    )
