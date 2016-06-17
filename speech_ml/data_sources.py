"""TODO.
"""

import os
from collections import Iterable
from abc import abstractmethod
import h5py
import numpy as np

from .util import yaml_to_dict, ttv_yaml_to_dict


class DataSource(object):
    """Base class for all datasources."""

    def __getitem__(self, key):
        if isinstance(key, Iterable) and type(key) != str:
            return (self._process(key) for uri in key)
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



# TODO: this with lookup_table implemented
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

        all_users = {**ttv['test'], **ttv['train']}
        all_users = {**all_users, **ttv['validation']}


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
        return X,Y

        # def _create_lookup_table(self):
        #     """Create a table which rerturns (resource_name, subjectID)"""
        #     lookup_table = []
        #     for data_set in ['train', 'validate', 'test']:
        #         subjects = self.ttv[data_set]
        #         for subjectID in subjects:
        #             uris = subjects[subjectID]
        #             for uri in uris:
        #                 lookup_table.append(subjectID, uri)
        #
        #     return np.array(lookup_table)
        # assert type(key) == slice or type(key) == int or isinstance(key, Iterable)
        # if type(key) == slice or isinstance(key, Iterable):
        #
        #     lookup_data = lookup_table[key]
        #     subjectIDs (x[0] for x in lookup_data)
        #     uris = (x[1] for x in lookup_data)
        #
        #     X = self.data_source[uris]
        #     if type(key) == slice:
        #         s = key
        #         Y = (self.make_target(X[i], subjectID, self.subject_info_dir) for i, y in eumerate(range(s.start, s.stop, s.step)))
        #     elif isinstance(key, Iterable):
        #         Y = (self.make_target(X[i], subjectID, self.
        #         ) for i, y in enumerate(key))
        #
        #     return (X, Y)
        # else:
        #     X = self.data_source[uri]
        #     Y = self.make_target
        #
        # return self.data_source[]



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
