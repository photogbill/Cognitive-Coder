import unittest

from wordfreq import top_words


class TestTopWords(unittest.TestCase):
    def test_counts_and_order(self):
        text = "the cat and the hat. The bat!"
        self.assertEqual(top_words(text, 2), [("the", 3), ("and", 1)])

    def test_ties_are_alphabetical(self):
        self.assertEqual(top_words("b a c", 3), [("a", 1), ("b", 1), ("c", 1)])

    def test_punctuation_and_apostrophes(self):
        self.assertEqual(top_words("Hello, hello! don't DON'T", 5),
                         [("don't", 2), ("hello", 2)])

    def test_fewer_words_than_n(self):
        self.assertEqual(top_words("one two", 10), [("one", 1), ("two", 1)])

    def test_zero_and_negative(self):
        self.assertEqual(top_words("a b", 0), [])
        with self.assertRaises(ValueError):
            top_words("a b", -1)
