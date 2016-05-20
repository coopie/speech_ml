import ttv
import unittest


DUMMY_DATASET = {
"1" : ['test/dummy_data/1_happy_kid_1.wav', 'test/dummy_data/1_sad_kid_1.wav'],
"2" : ['test/dummy_data/2_happy_kid_1.wav', 'test/dummy_data/2_sad_kid_1.wav'],
"3" : ['test/dummy_data/3_happy_kid_1.wav', 'test/dummy_data/3_sad_kid_1.wav'],
"4" : ['test/dummy_data/4_happy_kid_1.wav', 'test/dummy_data/4_sad_kid_1.wav'],
"5" : ['test/dummy_data/5_happy_kid_1.wav', 'test/dummy_data/5_sad_kid_1.wav']
}


BIG_DUMMY_DATASET = {
"1" : ['test/dummy_data/1_happy_kid_1.wav', 'test/dummy_data/1_sad_kid_1.wav'],
"2" : ['test/dummy_data/2_happy_kid_1.wav', 'test/dummy_data/2_sad_kid_1.wav'],
"3" : ['test/dummy_data/3_happy_kid_1.wav', 'test/dummy_data/3_sad_kid_1.wav'],
"4" : ['test/dummy_data/4_happy_kid_1.wav', 'test/dummy_data/4_sad_kid_1.wav'],
"5" : ['test/dummy_data/5_happy_kid_1.wav', 'test/dummy_data/5_sad_kid_1.wav'],
"6" : ['test/dummy_data/6_happy_kid_1.wav', 'test/dummy_data/6_sad_kid_1.wav'],
"7" : ['test/dummy_data/7_happy_kid_1.wav', 'test/dummy_data/7_sad_kid_1.wav', 'test/dummy_data/7_sad_BOB_1.wav'],
"8" : ['test/dummy_data/8_happy_kid_1.wav', 'test/dummy_data/8_sad_kid_1.wav', 'test/dummy_data/8_sad_A_1.wav', 'test/dummy_data/8_sad_B_1.wav', 'test/dummy_data/8_sad_C_1.wav', 'test/dummy_data/8_sad_D_1.wav']
}


class TestTTVMethods(unittest.TestCase):

    def test_get_dataset(self):
        self.assertEqual(
            ttv.get_dataset(['test/dummy_data']),
            DUMMY_DATASET
        )

    def test_make_ttv_size(self):
        test, train, validation = ttv.make_ttv(DUMMY_DATASET, ttv_ratio=(1, 3, 1))
        self.assertEqual(
            len(train) / len(test),
            3
        )
        self.assertEqual(
            len(test),
            len(validation)
        )


    def test_subject_independent(self):
        sets = (test, train, validation) = ttv.make_ttv(BIG_DUMMY_DATASET, ttv_ratio=(1, 3, 1))

        test, train, validation = [set([f.split('/')[-1].split('.')[0].split('_')[0] for f in data_set])
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
            ttv.split_list([1,2], 0.5),
            ([1], [2])
        )
        self.assertEqual(
            ttv.split_list([1,2], 0.6),
            ([1], [2])
        )



if __name__ == '__main__':
    unittest.main()
