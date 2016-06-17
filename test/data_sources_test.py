import unittest
import os
import numpy as np
import numpy.testing as npt

from speech_ml.data_sources import *
from speech_ml.util import yaml_to_dict

DUMMY_DATA_PATH = os.path.join('test', 'dummy_data')


class DataSourcesTests(unittest.TestCase):


    def test_file_data_source(self):
        ds = FileDataSource(DUMMY_DATA_PATH, suffix='.wav')

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
        ds = WaveformDataSource(FileDataSource(DUMMY_DATA_PATH, suffix='.wav'), process_waveform=dummy_process_waveforms)

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
                    FileDataSource(DUMMY_DATA_PATH, suffix='.wav'),
                    process_waveform=dummy_process_waveforms),
                dummy_process_spectrograms
            )

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




    def test_ttv_examples_generator(self):
        data_source = DummyDataSource()

        def make_target(X, key, subjectID, subject_info_data_source):
            metadata = yaml_to_dict(subject_info_data_source[subjectID])
            return metadata['legs']


        subject_info_dir = os.path.join('test', 'dummy_data', 'metadata')
        ttv = yaml_to_dict(os.path.join(subject_info_dir, 'dummy_ttv.yaml'))

        examples_ds = TTVExamplesDataSource(data_source, make_target, ttv, subject_info_dir)

        self.assertTrue(
            examples_ds['blorp_2'],
            (data_source.data['blorp_2'], 1)
        )
        self.assertTrue(
            examples_ds['blerp_1'],
            (data_source.data['blerp_1'], 2)
        )
        self.assertTrue(
            examples_ds['shlerp_322'],
            (data_source.data['shlerp_322'], 3)
        )



def dummy_process_waveforms(path):
    """A way of identifying which file the dummy waveform comes from."""
    filename_split = path.split(os.sep)[-1].split('.')[0].split('_')
    ident = filename_split[0]
    is_happy = filename_split[1] == 'happy'
    return np.array((int(ident) * 2) + int(is_happy))


def dummy_process_spectrograms(waveform):
    return np.eye(2) * waveform


class DummyDataSource(DataSource):
    def __init__(self):
        self.data = {
            'blorp_2': np.eye(2) * 1,
            'blerp_1': np.eye(2) * 2,
            'shlerp_322': np.eye(2) * 3
        }

    def _process(self, key):
        return self.data[key]




if __name__ == '__main__':
    unittest.main()
