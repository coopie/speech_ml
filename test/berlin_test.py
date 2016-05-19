import src_context

from berlin import renaming

import unittest


class TestBerlinRenaming(unittest.TestCase):
    def test_renaming(self):
        self.assertEqual(
            renaming.rename_file('berlin/03a01Fa.wav'),
            'berlin03_happy_a01_1.wav'
        )

if __name__ == '__main__':
    unittest.main()
