# Provides methods to get the raw waveforms from a TTV.yaml and normalise them

import numpy as np
import h5py
import os
from scipy.io.wavfile import read
from util import cache_ttv_data, get_cached_ttv_data, filename_to_category_vector

from keras.utils.generic_utils import Progbar

CACHE_EXTENSION = '.waveforms.cache.hdf5'

def read_wav_file(path_to_wav_file):
    return read(path_to_wav_file)


def ttv_to_waveforms(ttv_info, normalise=None, get_waveform_data=read_wav_file, cache=None, verbosity=1):
    """
    Returns each ttv_field as a dictionary {x:, y:}
    cache: path to where cached data is, or where the data will be cached after retrieval. Note that the cahe filename will have ".waveforms.cache.hdf5 appended to the name"
    """
    def log(msg, level):
        if level <= verbosity:
            print(msg)

    if cache is not None and os.path.exists(cache + CACHE_EXTENSION):
        log('FOUND CACHED TTV WAVEFORM DATA', 1)
        return get_cached_ttv_data(cache + CACHE_EXTENSION)

    ttv_info = (ttv_info['test'], ttv_info['train'], ttv_info['validation'])


    NUM_RESOURCES = sum(list(map(lambda x: len(x), ttv_info)))
    pb = Progbar(NUM_RESOURCES, verbose=verbosity)

    def get_data(path):
        freq, wave_data = get_waveform_data(path)
        if normalise is not None:
            wave_data = normalise(wave_data, frequency=freq)
        pb.add(1)
        return freq, wave_data

    wavfiles = map_for_each_set(get_data, ttv_info)

    emotions = map_for_each_set(filename_to_category_vector, ttv_info)

    ttv_data = (
    {'x': np.array([x[1] for x in wavfiles[0]]), 'y': np.array(emotions[0]), 'frequencies': np.array([x[0] for x in wavfiles[0]])},
    {'x': np.array([x[1] for x in wavfiles[1]]), 'y': np.array(emotions[1]), 'frequencies': np.array([x[0] for x in wavfiles[1]])},
    {'x': np.array([x[1] for x in wavfiles[2]]), 'y': np.array(emotions[2]), 'frequencies': np.array([x[0] for x in wavfiles[2]])},
    )


    if cache is not None:
        cache_ttv_data(cache + CACHE_EXTENSION, ttv_data)
        log('CACHING WAVEFORM TTV DATA FOR LATER USE AT: ' +  cache + CACHE_EXTENSION, 1)

    return ttv_data

def map_for_each_set(func, ttv):
    return [list(map(func, s)) for s in ttv]
