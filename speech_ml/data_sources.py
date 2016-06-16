"""TODO.
"""

import os
from abc import abstractmethod


class DataSource(object):
    """Base class for all datasources."""

    def __getitem__(self, key):
        """Default interpretation for string based getting."""
        if type(key) == str:
            return self._process(key)
        else:
            return [self._process(key) for uri in key]

    @abstractmethod
    def _process(self, ident):
        pass


class FileDataSource(DataSource):
    """Wrapper around file system"""

    def __init__(self, base_dir, suffix='.wav'):
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


    # TODO: caching
    def _process(self, key):
        return self.process_spectrogram(self.waveform_source[key])
