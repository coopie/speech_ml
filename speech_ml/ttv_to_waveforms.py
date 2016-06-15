# Provides methods to get the raw waveforms from a TTV.yaml and normalise them

import numpy as np
import os
from scipy.io.wavfile import read
import posixpath
from .util import cache_data, get_cached_data, gen_data_from_cache

# from keras.utils.generic_utils import Progbar
from tqdm import tqdm

CACHE_EXTENSION = '.waveforms.cache.hdf5'


def read_wav_file(path_to_wav_file):
    return read(path_to_wav_file)


def ttv_to_waveforms(ttv_info, normalise=None, get_waveform_data=read_wav_file, cache=None, verbosity=1):
    u"""
    TODO: explain this better 🐸.

    cache: path to where cached data is, or where the data will be cached after retrieval. Note that the cache filename
    will have ".waveforms.cache.hdf5" appended to the name

    reading the waveforms is lazily evaluated
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

    if verbosity > 0:
        pb = tqdm(total=NUM_RESOURCES)


    def get_data(path):
        # to get over OS differnces
        path = os.path.join(*path.split(posixpath.sep))
        freq, wave_data = get_waveform_data(path)
        if normalise is not None:
            wave_data = normalise(wave_data, frequency=freq)
        if verbosity > 0:
            pb.update(1)
        return (freq, wave_data)

    # to get frequency
    frequency, _ = get_data(paths[0])
    waveforms = (get_data(path)[1] for path in paths)

    ids = np.array([strip_filename(path) for path in paths])

    ttv_data = ids, sets, waveforms, frequency

    if cache is not None:
        cache_data(cache + CACHE_EXTENSION, ttv_data)
        log('CACHING WAVEFORM TTV DATA FOR LATER USE AT: ' + cache + CACHE_EXTENSION, 1)
        # the waveforms generator has been used, so create a generator which pulls data from the file
        waveforms = gen_data_from_cache(cache + CACHE_EXTENSION)
        ttv_data = ids, sets, waveforms, frequency


    return ttv_data


def strip_filename(path):
    return os.path.split(path)[-1].split('.')[0]
