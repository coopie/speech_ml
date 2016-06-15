import unittest
import numpy as np
import os

from speech_ml.data_names import *
from speech_ml.ttv_to_spectrograms import ttv_to_spectrograms
from speech_ml.util import get_cached_data

DUMMY_DATA = 'test/dummy_data'

TEST_TTV_INFO = {
    "test": ['test/dummy_data/1_happy_kid_1.wav'],
    "train": ['test/dummy_data/2_happy_kid_1.wav'],
    "validation": ['test/dummy_data/3_happy_kid_1.wav']
}

EXPECTED = (
    np.array(['1_happy_kid_1', '2_happy_kid_1', '3_happy_kid_1']),
    np.array(['test', 'train', 'validation']),
    np.array([[[6, 12], [18, 24]], [[30, 36], [42, 48]], [[54, 0], [6, 12]]]),
    np.array([6, 6]),
    np.array([0, 1])
)


def dummy_get_waveforms(ttv_info, *unused, **also_unused):
    return (
        np.array(['1_happy_kid_1', '2_happy_kid_1', '3_happy_kid_1']),
        np.array(['test', 'train', 'validation']),
        np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]]),
        np.array(6)
    )


def dummy_make_spectrogram(waveform, fs, **unused):
    return np.array([fs, fs]), np.array([0, 1]), np.reshape(waveform * fs, (2, 2))


class TestTTVToSpectrogramMethods(unittest.TestCase):

    def test_get_spectrograms(self):

        spectrogram_data = ttv_to_spectrograms(
            TEST_TTV_INFO,
            make_spectrogram=dummy_make_spectrogram,
            get_waveforms=dummy_get_waveforms,
            verbosity=0
        )

        # to instantiate the data generator to a numpy array
        spectrogram_data = (
            spectrogram_data[0],
            spectrogram_data[1],
            np.array([x for x in spectrogram_data[2]]),
            spectrogram_data[3],
            spectrogram_data[4]
        )


        for expected, actual in zip(EXPECTED, spectrogram_data):
            self.assertTrue(
                np.all(expected == actual)
            )


    def test_caching(self):
        spectrogram_data = ttv_to_spectrograms(
            TEST_TTV_INFO,
            make_spectrogram=dummy_make_spectrogram,
            get_waveforms=dummy_get_waveforms,
            verbosity=0,
            cache='test'
        )

        # to instantiate the data generator to a numpy array
        spectrogram_data = (
            spectrogram_data[0],
            spectrogram_data[1],
            np.array([x for x in spectrogram_data[2]]),
            spectrogram_data[3],
            spectrogram_data[4]
        )

        self.assertTrue(os.path.exists('test.spectrograms.cache.hdf5'))

        cached = get_cached_data('test.spectrograms.cache.hdf5')
        cached = (
            cached[0],
            cached[1],
            np.array([x for x in cached[2]]),
            cached[3],
            cached[4]
        )

        for expected, actual in zip(EXPECTED, spectrogram_data):
            self.assertTrue(
                np.all(expected == actual)
            )

        for expected, actual in zip(cached, spectrogram_data):
            self.assertTrue(
                np.all(expected == actual)
            )




    @classmethod
    def tearDownClass(cls):
        if os.path.exists('test.spectrograms.cache.hdf5'):
            os.remove('test.spectrograms.cache.hdf5')

if __name__ == '__main__':
    unittest.main()
