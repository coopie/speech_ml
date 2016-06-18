"""TODO.
"""

import os
from collections import Iterable
from abc import abstractmethod
import h5py
import numpy as np

from .util import yaml_to_dict, ttv_yaml_to_dict
from .lookup_tables import TTVLookupTable


class DataSource(object):
    """Base class for all datasources."""

    def __getitem__(self, key):
        if isinstance(key, Iterable) and type(key) != str:
            return (self._process(uri) for uri in key)
        else:
            return self._process(key)

    @abstractmethod
    def _process(self, ident):
        pass



class FileDataSource(DataSource):
    """Wrapper around file system"""

    def __init__(self, base_dir='', suffix=''):
        self.base_dir = base_dir
        self.suffix = suffix

    def _process(self, ident):
        path_to_file = os.path.join(self.base_dir, ident + self.suffix)
        assert os.path.exists(path_to_file)
        return path_to_file



class WaveformDataSource(DataSource):
    """Datasource which produces waveforms from a path to file.

    The file source must produce a path to a wav file when given a string id
    """

    def __init__(self, file_source, process_waveform):
        self.file_source = file_source
        self.process_waveform = process_waveform


    def _process(self, ident):
        return self.process_waveform(self.file_source[ident])



class SpectrogramDataSource(DataSource):
    """Datasource which produces spectrograms from a path to file.

    The file source must produce a waveform when given a string id
    """
    def __init__(self, waveform_source, process_spectrogram):
        self.waveform_source = waveform_source
        self.process_spectrogram = process_spectrogram


    def _process(self, key):
        return self.process_spectrogram(self.waveform_source[key])



class ExamplesDataSource(DataSource):
    """Base class for generating training examoples from processed data and metadata"""
    def __init__(self, data_source, make_target):
        self.data_source = data_source
        self.make_target = make_target



class TTVExamplesDataSource(ExamplesDataSource):
    """Uses ttv and subject information to build a lookuptable"""
    def __init__(self, data_source, make_target, ttv, subject_info_dir):
        super(TTVExamplesDataSource, self).__init__(data_source, make_target)
        self.ttv = ttv
        self.subject_info_data_source = FileDataSource(subject_info_dir, suffix='.yaml')

        all_users = merge_dicts(ttv['test'], ttv['train'], ttv['validation'])

        # invert the map
        uri_to_subjectID = {}
        for userID in all_users:
            resources = all_users[userID]
            for resourceID in resources:
                uri_to_subjectID[resourceID] = userID

        self.uri_to_subjectID = uri_to_subjectID


    def _process(self, key):
        X = self.data_source[key]

        subjectID = self.uri_to_subjectID[key]
        Y = self.make_target(X, key, subjectID, self.subject_info_data_source)
        return X, Y



class ArrayLikeDataSource(DataSource):

    """Base clas for ArrayLikeDataSources. These are used as a way of treating the data as if it were one big array."""
    @abstractmethod
    def __getitem___(self, key):
        pass



class TTVArrayLikeDataSource(ArrayLikeDataSource):
    """TODO."""
    def __init__(self, data_source, ttv):
        self.data_source = data_source
        self.lookup = TTVLookupTable(ttv, shuffle_in_set=True)

    def __getitem__(self, key):
        assert type(key) == slice or type(key) == int or (isinstance(key, Iterable) and type(key) != str)
        return self.data_source[self.lookup[key]]

    def __len__(self):
        return len(self.lookup)

    def get_set(self, set_name):
        return SubArrayLikeDataSource(self, *self.lookup.get_set_bounds(set_name))




class SubArrayLikeDataSource(ArrayLikeDataSource):
    """TODO."""
    def __init__(self, parent, lower, upper):
        assert len(parent) >= upper - lower

        self.parent = parent
        self.lower = lower
        self.upper = upper

    def __getitem__(self, key):
        assert type(key) == slice or type(key) == int or (isinstance(key, Iterable) and type(key) != str)
        if type(key) == int:
            key += self.lower
            assert key < self.upper
            return self.parent[key]
        elif type(key) == slice:
            s = key
            return (self.parent[i + self.lower] for i in slice_to_range(s, len(self)))
        else:
            keyarr = key
            return (self.parent[i + self.lower] for i in keyarr if i + self.lower < self.upper)

        def __len__(self):
            return self.upper - self.lower

    def __len__(self):
        return self.upper - self.lower



# TODO
class CacheNumericalDataSource(DataSource):
    def __init__(self, cache_name):
        self.cache = h5py.File(cache_name, 'a')
        # self.existence_cache = TODO

    def __getitem__(self, key):
        if type(key) == slice:
            s = key
            indices = np.arange(s.start, s.stop, s.step)
            indices_in_cache = np.fromiter((i, self._in_cache(i)) for i in range(s.start, s.stop, s.step))
            if len(indices) == indices_in_cache:
                return self.cache[s]
            else:
                return np.array([])
            # if all in cache, then use slice, else don't
        elif isinstance(key, Iterable) and type(key) != str:
            pass
        else:
            pass


def slice_to_range(s, max_value):
    start = s.start if s.start is not None else 0
    stop = s.stop if s.stop is not None else max_value
    step = s.step if s.step is not None else 1
    return range(start, stop, step)


def merge_dicts(*dicts):
    merged = {}
    for d in dicts:
        merged = {**merged, **d}
    return merged
