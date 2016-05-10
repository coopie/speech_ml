# Provides methods to get the raw waveforms from a TTV.yaml and normalise them
# TODO: test this

import numpy as np
import h5py
import os
from scipy.io.wavfile import read
from util import mkdir_p, get_emotion_number_from_filename, EMOTIONS, cache_ttv_data, get_cached_ttv_data

from keras.utils.generic_utils import Progbar

def read_wav_file(path_to_wav_file):
    return read(path_to_wav_file)[1]

def filename_to_category_vector(filename):
    emotion_number = get_emotion_number_from_filename(filename)
    zeros = np.zeros(len(EMOTIONS), dtype='int16')
    zeros[emotion_number] = 1
    return zeros


def ttv_to_waveforms(ttv_info, normalise=None, get_waveform_data=read_wav_file, cache=None, verbosity=1):
    """
    Returns each ttv_field as a dictionary {x:, y:}
    """

    def log(msg, level):
        if level <= verbosity:
            print msg

    if cache is not None and os.path.exists(cache):
        log('FOUND CACHED TTV DATA', 1)
        return get_cached_ttv_data(cache)


    ttv_info = (ttv_info['test'], ttv_info['train'], ttv_info['validation'])


    NUM_RESOURCES = sum(map(lambda x: len(x), ttv_info))
    pb = Progbar(NUM_RESOURCES, verbose=verbosity)

    def get_data(path):
        wave_data = get_waveform_data(path)
        if normalise is not None:
            wave_data = normalise(wave_data)
        pb.add(1)
        return wave_data

    wavfiles = map(lambda info_set: map(get_data, info_set), ttv_info)

    emotions = map(lambda info_set: map(filename_to_category_vector, info_set), ttv_info)

    ttv_data = (
    {'x': np.array(wavfiles[0]), 'y': np.array(emotions[0])},
    {'x': np.array(wavfiles[1]), 'y': np.array(emotions[1])},
    {'x': np.array(wavfiles[2]), 'y': np.array(emotions[2])},
    )

    if cache is not None:
        cache_ttv_data(cache, ttv_data)
        log('CACHING TTV DATA FOR LATER USE', 1)

    return ttv_data
