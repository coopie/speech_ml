from waveform_tools import *
import unittest

def find_subarray(arr, subarr):
    return [x for x in range(len(arr)) if np.all(arr[x:x+len(subarr)] == subarr)]


class TestWaveformToolsMethods(unittest.TestCase):

    def test_middle_slice(self):
        self.assertEqual(
            middle_slice([1,2,3,4,5], 3),
            [2,3,4]
        )
        self.assertEqual(
            middle_slice([1,2,3,4,5], 5),
            [1,2,3,4,5]
        )
        self.assertEqual(
            middle_slice([1,2,3,4,5], 1),
            [3]
        )


    def test_pad_or_slice(self):
        self.assertEqual(
            pad_or_slice([1,2,3,4,5], 3),
            [2,3,4]
        )
        self.assertEqual(
            len(pad_or_slice([1,2,3,4,5], 7)),
            7
        )
        self.assertTrue(
            len(find_subarray(
                pad_or_slice([1,2,3], 7), [1,2,3])) == 1
        )



if __name__ == '__main__':
    unittest.main()
