import unittest

import numpy as np
import h5py
import os

from speech_ml.keras_generators import KerasGenerator


TEST_FILE = 'keras_generators.data.hdf5'


class TestKerasGenerators(unittest.TestCase):

    def test_dataset_exists(self):
        with h5py.File(TEST_FILE, 'r') as f:
            self.assertTrue('data' in f)


    def test_fitting_batch_size(self):
        f = h5py.File(TEST_FILE, 'r')
        gen = KerasGenerator(f['data'], batch_size=1)
        a = []
        for i in range(4):
            a.append(gen.__next__())

        self.assertEqual(
            len(a),
            4
        )

        expected_result = [(np.repeat((i % 3) + 1, 3), np.repeat((i % 3) + 1, 3)) for i in range(3)]
        for actual, expected in zip(a, expected_result):
            self.assertTrue(
                np.all(expected[1] == actual[0])
            )
            self.assertTrue(
                np.all(expected[1] == actual[1])
            )

    def test_not_well_fitting_batch_size(self):
        f = h5py.File(TEST_FILE, 'r')
        gen = KerasGenerator(f['data'], batch_size=2)
        a = []
        for i in range(2):
            a.append(gen.__next__())

        self.assertEqual(
            len(a),
            2
        )

        x = np.array([np.repeat(1, 3), np.repeat(2, 3)])
        one_next_call = (x, x)
        expected_result = [one_next_call, one_next_call]

        for actual, expected in zip(a, expected_result):
            for i in range(2):
                for j in range(2):
                    self.assertTrue(
                        np.all(expected[i][j] == actual[i][j])
                    )
                    self.assertTrue(
                        np.all(expected[i][j] == actual[i][j])
                    )



    @classmethod
    def setUpClass(cls):
        f = h5py.File(TEST_FILE, 'w')
        data = np.array([np.repeat(x, 3) for x in range(1, 4)])
        f['data'] = data



    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_FILE):
            os.remove(TEST_FILE)

if __name__ == '__main__':
    unittest.main()
