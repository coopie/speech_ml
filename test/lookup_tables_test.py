import unittest
import os

from speech_ml.lookup_tables import *
from speech_ml.util import yaml_to_dict

# so that shuffling is deterministic
import random
random.seed(322)


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


if __name__ == '__main__':
    unittest.main()
