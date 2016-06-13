import ttv
import unittest
from os.path import split as split_path, join as join_path


def make_path(fielname):
    return join_path('test', 'dummy_data', fielname)

DUMMY_DATASET = {
"1": [make_path('1_happy_kid_1.wav'), make_path('1_sad_kid_1.wav')],
"2": [make_path('2_happy_kid_1.wav'), make_path('2_sad_kid_1.wav')],
"3": [make_path('3_happy_kid_1.wav'), make_path('3_sad_kid_1.wav')],
"4": [make_path('4_happy_kid_1.wav'), make_path('4_sad_kid_1.wav')],
"5": [make_path('5_happy_kid_1.wav'), make_path('5_sad_kid_1.wav')]
}


BIG_DUMMY_DATASET = {
"1": [make_path('1_happy_kid_1.wav'), make_path('1_sad_kid_1.wav')],
"2": [make_path('2_happy_kid_1.wav'), make_path('2_sad_kid_1.wav')],
"3": [make_path('3_happy_kid_1.wav'), make_path('3_sad_kid_1.wav')],
"4": [make_path('4_happy_kid_1.wav'), make_path('4_sad_kid_1.wav')],
"5": [make_path('5_happy_kid_1.wav'), make_path('5_sad_kid_1.wav')],
"6": [make_path('6_happy_kid_1.wav'), make_path('6_sad_kid_1.wav')],
"7": [make_path('7_happy_kid_1.wav'), make_path('7_sad_kid_1.wav'), make_path('7_sad_BOB_1.wav')],
"8": [make_path('8_happy_kid_1.wav'), make_path('8_sad_kid_1.wav'), make_path('8_sad_A_1.wav'), make_path('8_sad_B_1.wav'), make_path('8_sad_C_1.wav'), make_path('8_sad_D_1.wav')]
}


def get_for_ttv(key, data_sets):
    return (
        data_sets['test'][key],
        data_sets['train'][key],
        data_sets['validation'][key]
    )


class TestTTVMethods(unittest.TestCase):

    def test_get_dataset(self):
        self.assertEqual(
            ttv.get_dataset([join_path('test', 'dummy_data')]),
            DUMMY_DATASET
        )

    def test_make_ttv_size(self):
        data_sets = ttv.make_ttv(DUMMY_DATASET, ttv_ratio=(1, 3, 1))
        test, train, validation = get_for_ttv('paths', data_sets)


        self.assertEqual(
            len(train) / len(test),
            3
        )
        self.assertEqual(
            len(test),
            len(validation)
        )


    def test_subject_independent(self):
        data_sets = ttv.make_ttv(BIG_DUMMY_DATASET, ttv_ratio=(1, 3, 1))
        (test, train, validation) = get_for_ttv('paths', data_sets)

        test, train, validation = [set([split_path(f)[-1].split('.')[0].split('_')[0] for f in data_set])
                                    for data_set in (test, train, validation)]

        for ident in test:
            self.assertNotIn(ident, train)
            self.assertNotIn(ident, validation)

        for ident in train:
            self.assertNotIn(ident, test)
            self.assertNotIn(ident, validation)

        for ident in validation:
            self.assertNotIn(ident, train)
            self.assertNotIn(ident, test)


    def test_split_list_simple(self):
        self.assertEqual(
            ttv.split_list([1, 2], 0.5),
            ([1], [2])
        )
        self.assertEqual(
            ttv.split_list([1, 2], 0.6),
            ([1], [2])
        )



if __name__ == '__main__':
    unittest.main()
