import src_context

from toronto import renaming

import unittest

# corpora\toronto\OAF_back_angry.wav
# corpora\RAVDESS\RAVDESS1_angry_dog_1.wav

class TestTorontoRenaming(unittest.TestCase):
    def test_renaming(self):
        self.assertEqual(
            renaming.get_new_name('OAF_back_angry.wav'),
            'OAF_angry_back_1.wav'
        )

if __name__ == '__main__':
    unittest.main()
