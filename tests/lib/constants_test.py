import unittest

from src.lib.constants import OPTION_LETTERS


class TestOptionLetters(unittest.TestCase):
    def test_exact_value(self):
        self.assertEqual(OPTION_LETTERS, ["A", "B", "C", "D"])

    def test_is_list_of_unique_uppercase(self):
        self.assertIsInstance(OPTION_LETTERS, list)
        self.assertEqual(len(set(OPTION_LETTERS)), 4)
        for letter in OPTION_LETTERS:
            self.assertTrue(letter.isupper())
            self.assertEqual(len(letter), 1)


if __name__ == "__main__":
    unittest.main()
