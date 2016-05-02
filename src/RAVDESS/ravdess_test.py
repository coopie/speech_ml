import unittest
import renaming

class TestRenaming(unittest.TestCase):

    def test_renaming(self):
        self.assertEqual(
            renaming.get_new_name_RAVDESS("03-02-03-02-02-322"),
            "RAVDESS322_happy_dog_2"
        )

if __name__ == '__main__':
    unittest.main()
