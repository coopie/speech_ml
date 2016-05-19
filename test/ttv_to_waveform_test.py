from ttv_to_waveforms import ttv_to_waveforms
import unittest

import numpy as np

DUMMY_DATA = 'test/dummy_data'

TEST_TTV_INFO = {
"test" : ['test/dummy_data/1_happy_kid_1.wav', 'test/dummy_data/1_sad_kid_1.wav'],
"train" : ['test/dummy_data/2_happy_kid_1.wav', 'test/dummy_data/2_sad_kid_1.wav'],
"validation" : ['test/dummy_data/3_happy_kid_1.wav', 'test/dummy_data/3_sad_kid_1.wav']
}

def dummy_read_data(path):
    return np.array([5,6,7,8]), np.array([1,2,3,4])

class TestTTVToWaveformMethods(unittest.TestCase):

    def test_get_dataset(self):
        waveforms = ttv_to_waveforms(
            TEST_TTV_INFO,
            get_waveform_data=dummy_read_data,
            verbosity=0
        )
        expected_waveform = np.array([1,2,3,4])
        expected_dataset = {
            'x': [expected_waveform, expected_waveform],
            'y': np.array([np.array([0,0,1,0,0,0,0,0]), np.array([0,0,0,1,0,0,0,0])]),
            'frequencies': np.array([np.array([5,6,7,8]), np.array([5,6,7,8])])
        }

        self.assertTrue(
            np.all(waveforms[0]['y'] == expected_dataset['y'])
        )
        self.assertTrue(
            np.all(waveforms[1]['y'] == expected_dataset['y'])
        )
        self.assertTrue(
            np.all(waveforms[2]['y'] == expected_dataset['y'])
        )


        self.assertTrue(
            np.all(waveforms[0]['x'] == expected_dataset['x'])
        )
        self.assertTrue(
            np.all(waveforms[1]['x'] == expected_dataset['x'])
        )
        self.assertTrue(
            np.all(waveforms[2]['x'] == expected_dataset['x'])
        )

        self.assertTrue(
            np.all(waveforms[0]['frequencies'] == expected_dataset['frequencies'])
        )
        self.assertTrue(
            np.all(waveforms[1]['frequencies'] == expected_dataset['frequencies'])
        )
        self.assertTrue(
            np.all(waveforms[2]['frequencies'] == expected_dataset['frequencies'])
        )



if __name__ == '__main__':
    unittest.main()
