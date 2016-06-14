import numpy as np
from keras.utils.generic_utils import Progbar
from scipy.signal import spectrogram
import os

from .util import get_cached_data, cache_data
from .ttv_to_waveforms import ttv_to_waveforms
from .data_names import *


def default_make_spectrogram(waveform, **spec_args):
    return spectrogram(waveform, **spec_args)

CACHE_EXTENSION = '.spectrograms.cache.hdf5'

# from some paper:
# TODO
DEFAULT_SPECTROGRAM_ARGS = {
    # 'fs': 16000,
    # 'window': 16*40, #40ms,
    'nperseg': 16 * 40,
    'noverlap': 14 * 20
}


def ttv_to_spectrograms(ttv_info,
                        normalise_waveform=None,
                        normalise_spectrogram=None,
                        make_spectrogram=default_make_spectrogram,
                        spectrogram_args=DEFAULT_SPECTROGRAM_ARGS,
                        get_waveforms=ttv_to_waveforms,
                        cache=None,
                        verbosity=1):
    """
    TODO: explain caching.
    """
    def log(msg, v):
        if verbosity >= v:
            print(msg)


    if cache is not None and os.path.exists(cache + CACHE_EXTENSION):
        log('FOUND CACHED TTV SPECTORGRAM DATA', 1)
        return get_cached_data(cache + CACHE_EXTENSION)

    waveform_data = get_waveforms(
        ttv_info,
        normalise=normalise_waveform,
        cache=cache,
        verbosity=verbosity
    )

    NUM_RESOURCES = len(waveform_data[0])
    pb = Progbar(NUM_RESOURCES, verbose=verbosity)

    def make_spectrogram_with_progbar(waveform, frequency):
        fs, ts, s = make_spectrogram(waveform, fs=frequency, **spectrogram_args)
        if normalise_spectrogram is not None:
            s = normalise_spectrogram(s, frequencies=fs)
        pb.add(1)
        return (fs, ts, s)

    frequency = waveform_data[FREQUENCY]
    spectrogram_data = [make_spectrogram_with_progbar(waveform, frequency)
        for waveform in waveform_data[WAVEFORM]]

    spectrograms = []
    frequencies = []
    times = []
    for fs, ts, s in spectrogram_data:
        spectrograms.append(s)
        frequencies.append(fs)
        times.append(ts)

    spectrogram_data = (
        waveform_data[ID],
        waveform_data[SET],
        np.array(spectrograms),
        np.array(frequencies[0]),
        np.array(times[0])
    )

    if cache is not None:
        log('CACHING TTV SPECTORGRAM DATA FOR LATER USE AT: ' + cache + CACHE_EXTENSION, 1)
        cache_data(cache + CACHE_EXTENSION, spectrogram_data)

    return spectrogram_data


def map_for_each_set_x(func, ttv):
    return [list(map(func, s['x'])) for s in ttv]


def for_each_set(func, ttv):
    return [func(s) for s in ttv]
