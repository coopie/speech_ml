import unittest
import os
import numpy as np

from speech_ml.lookup_tables import *
from speech_ml.util import yaml_to_dict

DUMMY_DATA_PATH = os.path.join('test', 'dummy_data', 'metadata')


class LookupTablesTests(unittest.TestCase):

    def test_ttv_lookup_table(self):
        ttv = yaml_to_dict(os.path.join(DUMMY_DATA_PATH, 'dummy_ttv.yaml'))
        lt = TTVLookupTable(ttv)

        self.assertEqual(
            len(lt),
            3
        )

        for set_name in ['test', 'train', 'validation']:
            start, end = lt.get_set_bounds(set_name)

            uris_in_set = sum((x for x in ttv[set_name].values()), [])

            self.assertEqual(
                set(lt[start:end]),
                set(uris_in_set)
            )


    def test_ttv_lookup_table_shuffled(self):
        ttv = yaml_to_dict(os.path.join(DUMMY_DATA_PATH, 'dummy_ttv.yaml'))
        ttv['train'] = dict((str(i), [str(i)]) for i in range(100))

        lt = TTVLookupTable(ttv, shuffle_in_set=True)

        start, end = lt.get_set_bounds('train')

        uris_in_set = sum((x for x in ttv['train'].values()), [])

        self.assertEqual(
            set(lt[start:end]),
            set(uris_in_set)
        )


        self.assertFalse(
            lt[start:end] ==
            uris_in_set
        )


    def test_shuffle_deterministic(self):
        """Test that the sufffle made in several lookup tables are the same."""

        ttv = yaml_to_dict(os.path.join(DUMMY_DATA_PATH, 'dummy_large_ttv.yaml'))

        lt_unshuffled = TTVLookupTable(ttv, shuffle_in_set=False)
        lt1 = TTVLookupTable(ttv, shuffle_in_set=True)
        lt2 = TTVLookupTable(ttv, shuffle_in_set=True)

        for set_name in ['test', 'train', 'validation']:
            start_unshuf, end_unshuf = lt_unshuffled.get_set_bounds('test')
            uris_unshuf = lt_unshuffled[start_unshuf:end_unshuf]

            start_shuf, end_shuf = lt1.get_set_bounds('test')
            uris_shuf = lt1[start_shuf:end_shuf]

            self.assertEqual(set(uris_shuf), set(uris_unshuf))
            self.assertFalse(
                np.all(
                    uris_shuf == uris_unshuf
                )
            )




        np.testing.assert_equal(lt1[:], lt2[:])





if __name__ == '__main__':
    unittest.main()
