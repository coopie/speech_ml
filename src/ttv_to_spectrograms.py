from ttv_to_waveforms import ttv_to_waveforms
import numpy as np
from keras.utils.generic_utils import Progbar
from scipy.signal import spectrogram
import os
from util import get_cached_ttv_data, cache_ttv_data


flag = True
def default_make_spectrogram(waveform, spec_args):
    global flag
    f, t, sxx = spectrogram(waveform, **spec_args)
    if flag:
        np.savetxt('frequencies.txt', f, fmt='%.8f')
        np.savetxt('times.txt', t, fmt='%.8f')
        flag = False

    return sxx

CACHE_EXTENSION = '.spectrograms.cache.hdf5'

# from some paper:
# TODO
DEFAULT_SPECTROGRAM_ARGS = {
    'fs': 48000
}

def ttv_to_spectrograms(ttv_info,
                        normalise_waveform=None,
                        normalise_spectrogram=None,
                        make_spectrogram=default_make_spectrogram,
                        spectrogram_args=DEFAULT_SPECTROGRAM_ARGS,
                        get_waveforms=ttv_to_waveforms,
                        cache=None,
                        verbosity=1):
    '''
    TODO: explain caching
    '''
    def log(msg, v):
        if verbosity >= v:
            print(msg)


    if cache is not None and os.path.exists(cache + CACHE_EXTENSION):
        log('FOUND CACHED TTV SPECTORGRAM DATA', 1)
        return get_cached_ttv_data(cache + CACHE_EXTENSION)

    test, train, validation = get_waveforms(
        ttv_info,
        normalise=normalise_waveform,
        cache=cache,
        verbosity=verbosity
    )

    NUM_RESOURCES = sum([len(test['x']), len(train['x']), len(validation['x'])])
    pb = Progbar(NUM_RESOURCES, verbose=verbosity)

    def make_spectrogram_with_progbar(waveform):
        s = make_spectrogram(waveform, spectrogram_args)
        if normalise_spectrogram is not None:
            s = normalise_spectrogram(s)
        pb.add(1)
        return s


    test['x'] = np.array([make_spectrogram_with_progbar(datum) for datum in test['x']])
    train['x'] = np.array([make_spectrogram_with_progbar(datum) for datum in train['x']])
    validation['x'] = np.array([make_spectrogram_with_progbar(datum) for datum in validation['x']])

    if cache is not None:
        log('CACHING TTV SPECTORGRAM DATA FOR LATER USE AT: ' + cache + CACHE_EXTENSION, 1)
        cache_ttv_data(cache + CACHE_EXTENSION, (test, train, validation))

    return test, train, validation


def map_for_each_set_x(func, ttv):
    return [list(map(func, s['x'])) for s in ttv]
