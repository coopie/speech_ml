import unittest
import os
import numpy as np
import numpy.testing as npt

from speech_ml.data_sources import *


DUMMY_DATA_PATH = os.path.join('test', 'dummy_data')


class DataSourcesTests(unittest.TestCase):


    def test_file_data_source(self):
        ds = FileDataSource(DUMMY_DATA_PATH)

        self.assertTrue(
            os.path.isfile(ds['1_sad_kid_1'])
        )
        self.assertEqual(
            ds['1_sad_kid_1'],
            os.path.join(DUMMY_DATA_PATH, '1_sad_kid_1.wav')
        )

        filenames = (x.split('.')[0] for x in os.listdir(DUMMY_DATA_PATH) if x.endswith('wav'))
        self.assertTrue(
            all(
                [os.path.exists(ds[f]) for f in filenames]
            )
        )




    def test_waveform_data_source(self):
        ds = WaveformDataSource(FileDataSource(DUMMY_DATA_PATH), process_waveform=dummy_process_waveforms)

        self.assertTrue(
            np.all(
                ds['1_sad_kid_1'] == np.array(2)
            )
        )

        paths = [os.path.join(DUMMY_DATA_PATH, f) for f in os.listdir(DUMMY_DATA_PATH) if f.endswith('.wav')]
        filenames = [x.split(os.sep)[-1].split('.')[0] for x in paths]

        npt.assert_array_equal(
            np.array([ds[f] for f in filenames]),
            np.array([dummy_process_waveforms(p) for p in paths])
        )

    def test_spectrogram_data_source(self):
        ds = \
            SpectrogramDataSource(
                WaveformDataSource(
                    FileDataSource(DUMMY_DATA_PATH),
                    process_waveform=dummy_process_waveforms),
                dummy_process_spectrograms
            )

        print(ds['1_sad_kid_1'])

        self.assertTrue(
            np.all(
                ds['1_sad_kid_1'] == np.eye(2) * 2
            )
        )

        paths = [os.path.join(DUMMY_DATA_PATH, f) for f in os.listdir(DUMMY_DATA_PATH) if f.endswith('.wav')]
        filenames = [x.split(os.sep)[-1].split('.')[0] for x in paths]

        npt.assert_array_equal(
            np.array([ds[f] for f in filenames]),
            np.array([dummy_process_spectrograms(dummy_process_waveforms(p)) for p in paths])
        )




def dummy_process_waveforms(path):
    """A way of identifying which file the dummy waveform comes from."""
    filename_split = path.split(os.sep)[-1].split('.')[0].split('_')
    ident = filename_split[0]
    is_happy = filename_split[1] == 'happy'
    return np.array((int(ident) * 2) + int(is_happy))


def dummy_process_spectrograms(waveform):
    return np.eye(2) * waveform

if __name__ == '__main__':
    unittest.main()
