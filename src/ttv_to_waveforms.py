# Provides methods to get the raw waveforms from a TTV.yaml and normalise them

import numpy as np
import h5py
import os
from scipy.io.wavfile import read
from util import cache_data, get_cached_data, filename_to_category_vector

from keras.utils.generic_utils import Progbar

CACHE_EXTENSION = '.waveforms.cache.hdf5'

def read_wav_file(path_to_wav_file):
    return read(path_to_wav_file)


def ttv_to_waveforms(ttv_info, normalise=None, get_waveform_data=read_wav_file, cache=None, verbosity=1):
    """
    TODO: explain this better 🐸
    cache: path to where cached data is, or where the data will be cached after retrieval. Note that the cahe filename will have ".waveforms.cache.hdf5 appended to the name"
    """
    def log(msg, level):
        if level <= verbosity:
            print(msg)

    if cache is not None and os.path.exists(cache + CACHE_EXTENSION):
        log('FOUND CACHED TTV WAVEFORM DATA', 1)
        return get_cached_data(cache + CACHE_EXTENSION)

    paths = np.array(ttv_info['test'] + ttv_info['train'] + ttv_info['validation'])
    sets = np.concatenate((
        np.repeat('test', len(ttv_info['test'])),
        np.repeat('train', len(ttv_info['train'])),
        np.repeat('validation', len(ttv_info['validation']))))

    NUM_RESOURCES = len(paths)

    pb = Progbar(NUM_RESOURCES, verbose=verbosity)

    def get_data(path):
        freq, wave_data = get_waveform_data(path)
        if normalise is not None:
            wave_data = normalise(wave_data, frequency=freq)
        pb.add(1)
        return (freq, wave_data)

    waveforms_and_frequencies = [get_data(path) for path in paths]

    waveforms   = np.array([x[1] for x in waveforms_and_frequencies])
    frequencies = np.array([x[0] for x in waveforms_and_frequencies])

    ids = np.array([strip_filename(path) for path in paths])

    ttv_data = ids, sets, waveforms, frequencies

    if cache is not None:
        cache_data(cache + CACHE_EXTENSION, ttv_data)
        log('CACHING WAVEFORM TTV DATA FOR LATER USE AT: ' +  cache + CACHE_EXTENSION, 1)

    return ttv_data


def strip_filename(path):
    return path.split('/')[-1].split('.')[0]
