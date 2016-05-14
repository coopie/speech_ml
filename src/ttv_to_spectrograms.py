from ttv_to_waveforms import ttv_to_waveforms
import numpy as np
from keras.utils.generic_utils import Progbar


def default_make_spectrogram(waveform):
    pass


CACHE_EXTENSION = '.spectrograms.cache.hdf5'

def ttv_to_spectrograms(ttv_info,
                        normalise_waveform=None, # not suggested
                        normalise_spectrogram=None,
                        make_spectrogram=default_make_spectrogram,
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
        s = make_spectrogram(waveform)

        if normalise_spectrogram is not None:
            s = normalise_spectrogram(s)
        pb.add(1)
        return s


    test['x'] = np.array([make_spectrogram_with_progbar(datum) for datum in test['x']])
    train['x'] = np.array([make_spectrogram_with_progbar(datum) for datum in train['x']])
    validation['x'] = np.array([make_spectrogram_with_progbar(datum) for datum in validation['x']])

    if cache is not None and os.path.exists(cache + CACHE_EXTENSION):
        log('CACHING TTV SPECTORGRAM DATA FOR LATER USE AT: ' + cache + CACHE_EXTENSION, 1)
        cache_ttv_data(cache + CACHE_EXTENSION)

    return test, train, validation








def map_for_each_set_x(func, ttv):
    return [list(map(func, s['x'])) for s in ttv]


    if cache is not None:
        # save cache
        pass
