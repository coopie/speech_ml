import unittest
import os
import numpy as np
import numpy.testing as npt
import h5py

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
            np.array([dummy_process_waveforms(p)[1] for p in paths])
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
            np.array([dummy_process_spectrograms(dummy_process_waveforms(p)[1])[-1] for p in paths])
        )


    def test_ttv_examples_generator(self):
        data_source = DummyDataSource()

        def make_target(X, key, subjectID, subject_info_data_source):
            metadata = yaml_to_dict(subject_info_data_source[subjectID])
            return metadata['legs']


        subject_info_dir = os.path.join('test', 'dummy_data', 'metadata')
        ttv = yaml_to_dict(os.path.join(subject_info_dir, 'dummy_ttv.yaml'))

        examples_ds = TTVExamplesDataSource(data_source, make_target, ttv, subject_info_dir)

        self.assertEqual(
            examples_ds['blorp_2'],
            (data_source.data['blorp_2'], 1)
        )
        self.assertEqual(
            examples_ds['blerp_1'],
            (data_source.data['blerp_1'], 2)
        )
        self.assertEqual(
            examples_ds['shlerp_322'],
            (data_source.data['shlerp_322'], 3)
        )
        self.assertEqual(
            examples_ds[['shlerp_322', 'blerp_1', 'blorp_2']],
            ([data_source.data[x] for x in ['shlerp_322', 'blerp_1', 'blorp_2']], [3, 2, 1])
        )


    def test_lambda_data_source(self):
        data_source = DummyDataSource()

        lam_ds = LambdaDataSource(lambda x: x + 1, data_source)

        for key in ['blorp_2', 'blerp_1', 'shlerp_322']:
            self.assertEqual(
                lam_ds[key],
                data_source.data[key] + 1
            )


    def test_ttv_array_like_data_source(self):
        dummy_data_source = DummyDataSource()
        subject_info_dir = os.path.join('test', 'dummy_data', 'metadata')
        ttv = yaml_to_dict(os.path.join(subject_info_dir, 'dummy_ttv.yaml'))

        array_ds = TTVArrayLikeDataSource(dummy_data_source, ttv)

        self.assertEqual(len(array_ds), 3)

        all_values = np.fromiter((x for x in array_ds[:]), dtype='int16')

        self.assertTrue(
            np.all(
                np.in1d(
                    all_values,
                    np.array([1, 2, 3])
                )
            )
        )


    def test_subarray_like_data_source(self):
        dummy_data_source = DummyDataSource()
        subject_info_dir = os.path.join('test', 'dummy_data', 'metadata')
        ttv = yaml_to_dict(os.path.join(subject_info_dir, 'dummy_ttv.yaml'))

        array_ds = TTVArrayLikeDataSource(dummy_data_source, ttv)


        def get_all_values_set(ttv, set_name):
            data_set = ttv[set_name]
            uris = []
            for subjectID in data_set:
                uris += data_set[subjectID]
            return uris

        for set_name in ['test', 'train', 'validation']:
            set_ds = array_ds.get_set(set_name)

            self.assertTrue(len(set_ds), 1)
            self.assertEqual(
                [x for x in set_ds[:]],
                [dummy_data_source[x] for x in get_all_values_set(ttv, set_name)]
            )


    def test_cached_ttv_array_like_data_source(self):
        dummy_data_source = DummyDataSource()
        subject_info_dir = os.path.join('test', 'dummy_data', 'metadata')
        ttv = yaml_to_dict(os.path.join(subject_info_dir, 'dummy_ttv.yaml'))

        array_ds = CachedTTVArrayLikeDataSource(dummy_data_source, ttv, data_name='dummy', cache_name='test')

        self.assertEqual(len(array_ds), 3)

        all_values = array_ds[:]

        self.assertTrue(
            np.all(
                np.in1d(
                    all_values,
                    np.array([1, 2, 3])
                )
            )
        )

        f = h5py.File('test.cache.hdf5', 'a')
        self.assertEqual(len(f['dummy']), len(array_ds))

        for in_cache, in_data_source in zip(f['dummy'], array_ds):
            self.assertTrue(
                np.all(
                    in_cache == in_data_source
                )
            )

        # changing a value in the cache now should alter the results returned by the dataset.
        f['dummy'][0] = 322
        all_values = all_values = array_ds[:]
        self.assertTrue(
            np.all(
                np.in1d(
                    all_values,
                    np.array([322, 2, 3])
                )
            )
        )

        # now resetting the cache, we shoud get the original results
        del f['dummy']
        f['dummy'] = np.repeat(CachedTTVArrayLikeDataSource.CACHE_MAGIC, 3)
        array_ds._CachedTTVArrayLikeDataSource__init_existence_cache()

        all_values = array_ds[:]
        self.assertTrue(
            np.all(
                np.in1d(
                    all_values,
                    np.array([1, 2, 3])
                )
            )
        )


    @classmethod
    def tearDownClass(cls):
        if os.path.exists('test.cache.hdf5'):
            os.remove('test.cache.hdf5')



def dummy_process_waveforms(path):
    """A way of identifying which file the dummy waveform comes from."""
    filename_split = path.split(os.sep)[-1].split('.')[0].split('_')
    ident = filename_split[0]
    is_happy = filename_split[1] == 'happy'

    frequency = 123

    return (frequency, np.array([int(ident) * 2]) + int(is_happy))


def dummy_process_spectrograms(waveform, *unused):
    times = np.array([1, 2])
    frequencies = np.array([3, 4])
    return (frequencies, times, np.eye(2) * waveform)


class dummyExampleDataSource():
    def __init__(arr):
        self.arr = arr

    def get_set(set_name):
        set_division = {
            'test': [1],
            'train': [2, 3],
            'validation': [4]
        }
        return dummyExampleDataSource(self.arr[set_division[set_name]])

    def __getitem__(self, key):
        return self.arr[key, 0], self.arr[key, 1]


class DummyDataSource(DataSource):
    def __init__(self):
        self.data = {
            'blorp_2': np.array(1) * 1,
            'blerp_1': np.array(1) * 2,
            'shlerp_322': np.array(1) * 3
        }

    def _process(self, key):
        return self.data[key]


if __name__ == '__main__':
    unittest.main()
