from ttv_to_spectrograms import ttv_to_spectrograms
import unittest
from util import filename_to_category_vector
import numpy as np

DUMMY_DATA = 'test/dummy_data'

TEST_TTV_INFO = {
"test" : ['test/dummy_data/1_happy_kid_1.wav', 'test/dummy_data/1_sad_kid_1.wav'],
"train" : ['test/dummy_data/2_happy_kid_1.wav', 'test/dummy_data/2_sad_kid_1.wav'],
"validation" : ['test/dummy_data/3_happy_kid_1.wav', 'test/dummy_data/3_sad_kid_1.wav']
}

def dummy_get_waveforms(ttv_info,*unused, **also_unused):
    test_y =       np.array([filename_to_category_vector(path) for path in ttv_info['test']])
    train_y =      np.array([filename_to_category_vector(path) for path in ttv_info['train']])
    validation_y = np.array([filename_to_category_vector(path) for path in ttv_info['validation']])
    return (
        {'x':np.array([322, 1337]) ,'y':test_y},
        {'x':np.array([322, 1337]) ,'y':train_y},
        {'x':np.array([322, 1337]) ,'y':validation_y},
    )


def dummy_make_spectrogram(waveform):
    return np.array([[1,2],[4,6]])


class TestTTVToWaveformMethods(unittest.TestCase):


    def test_get_spectrograms(self):

        expected_dataset = {
                'x': np.array([np.array([[1,2],[4,6]]), np.array([[1,2],[4,6]])]),
                'y': np.array([np.array([0,0,1,0,0,0,0,0]), np.array([0,0,0,1,0,0,0,0])])
            }

        spectrograms = ttv_to_spectrograms(
            TEST_TTV_INFO,
            make_spectrogram=dummy_make_spectrogram,
            get_waveforms=dummy_get_waveforms,
            verbosity=0
        )

        self.assertTrue(
            np.all(spectrograms[0]['y'] == expected_dataset['y'])
        )
        self.assertTrue(
            np.all(spectrograms[1]['y'] == expected_dataset['y'])
        )
        self.assertTrue(
            np.all(spectrograms[2]['y'] == expected_dataset['y'])
        )


        self.assertTrue(
            np.all(spectrograms[0]['x'] == expected_dataset['x'])
        )
        self.assertTrue(
            np.all(spectrograms[1]['x'] == expected_dataset['x'])
        )
        self.assertTrue(
            np.all(spectrograms[2]['x'] == expected_dataset['x'])
        )





if __name__ == '__main__':
    unittest.main()
0
