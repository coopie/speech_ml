from ttv_to_spectrograms import ttv_to_spectrograms
import unittest
from util import filename_to_category_vector
import numpy as np
from data_names import *
import os
from util import get_cached_data

DUMMY_DATA = 'test/dummy_data'

TEST_TTV_INFO = {
"test" : ['test/dummy_data/1_happy_kid_1.wav'],
"train" : ['test/dummy_data/2_happy_kid_1.wav'],
"validation" : ['test/dummy_data/3_happy_kid_1.wav']
}

EXPECTED = (
    np.array(['1_happy_kid_1', '2_happy_kid_1', '3_happy_kid_1']),
    np.array(['test', 'train', 'validation']),
    np.array([[[1,2],[3,4]], [[10,12], [14,16]], [[27,0], [3,6]]]),
    np.array([[1,1], [2,2], [3,3]]),
    np.array([[0,1], [0,1], [0,1]])
)

def dummy_get_waveforms(ttv_info, *unused, **also_unused):
    return  (
        np.array(['1_happy_kid_1', '2_happy_kid_1', '3_happy_kid_1']),
        np.array(['test', 'train', 'validation']),
        np.array([[1,2,3,4], [5,6,7,8], [9,0,1,2]]),
        np.array([1,2,3])
    )


def dummy_make_spectrogram(waveform, fs, **unused):
    return np.array([fs, fs]), np.array([0,1]), np.reshape(waveform * fs, (2,2))


class TestTTVToSpectrogramMethods(unittest.TestCase):

    def test_get_spectrograms(self):

        spectrogram_data = ttv_to_spectrograms(
            TEST_TTV_INFO,
            make_spectrogram=dummy_make_spectrogram,
            get_waveforms=dummy_get_waveforms,
            verbosity=0
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
        self.assertTrue(os.path.exists('test.spectrograms.cache.hdf5'))

        cached = get_cached_data('test.spectrograms.cache.hdf5')

        for expected, actual in zip(EXPECTED, spectrogram_data):
            self.assertTrue(
                np.all(expected == actual)
            )


    @classmethod
    def tearDownClass(cls):
        if os.path.exists('test.spectrograms.cache.hdf5'):
            os.remove('test.spectrograms.cache.hdf5')

if __name__ == '__main__':
    unittest.main()
