from ttv_to_waveforms import ttv_to_waveforms
from data_names import *
import unittest

import numpy as np
import os

from util import get_cached_data

DUMMY_DATA = 'test/dummy_data'

TEST_TTV_INFO = {
"test" : ['test/dummy_data/1_happy_kid_1.wav', 'test/dummy_data/1_sad_kid_1.wav'],
"train" : ['test/dummy_data/2_happy_kid_1.wav', 'test/dummy_data/2_sad_kid_1.wav'],
"validation" : ['test/dummy_data/3_happy_kid_1.wav', 'test/dummy_data/3_sad_kid_1.wav']
}

def dummy_read_data(path):
    return (60, np.array([1,2,3,4]))

class TestTTVToWaveformMethods(unittest.TestCase):

    def test_get_dataset(self):
        ttv_data = ttv_to_waveforms(
            TEST_TTV_INFO,
            get_waveform_data=dummy_read_data,
            verbosity=0
        )

        ids = np.array([filename.split('.')[0]
            for filename in os.listdir(DUMMY_DATA)
            if filename.endswith('.wav') and int(filename[0]) <= 3])

        sets = np.concatenate((np.repeat('test', 2), np.repeat('train', 2), np.repeat('validation', 2)))

        waveforms = np.repeat(np.array([[1,2,3,4]]), 6, axis=0)
        frequencies = np.repeat(60, 6, axis=0)

        self.assertTrue(
            np.all(ttv_data[ID] == ids)
        )
        self.assertTrue(
            np.all(ttv_data[SET] == sets)
        )
        self.assertTrue(
            np.all(ttv_data[WAVEFORM] == waveforms)
        )
        self.assertTrue(
            np.all(ttv_data[FREQUENCY] == frequencies)
        )

    def test_caching(self):
        ttv_data = ttv_to_waveforms(
            TEST_TTV_INFO,
            get_waveform_data=dummy_read_data,
            verbosity=0,
            cache='test'
        )

        self.assertTrue(os.path.exists('test.waveforms.cache.hdf5'))

        ids, sets, waveforms, frequencies = get_cached_data('test.waveforms.cache.hdf5')

        self.assertTrue(
            np.all(ttv_data[ID] == ids)
        )
        self.assertTrue(
            np.all(ttv_data[SET] == sets)
        )
        self.assertTrue(
            np.all(ttv_data[WAVEFORM] == waveforms)
        )
        self.assertTrue(
            np.all(ttv_data[FREQUENCY] == frequencies)
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('test.waveforms.cache.hdf5'):
            os.remove('test.waveforms.cache.hdf5')

if __name__ == '__main__':
    unittest.main()
