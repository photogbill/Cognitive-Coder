import unittest

from roman import from_roman, to_roman


class TestRoman(unittest.TestCase):
    def test_known_values(self):
        cases = {1: "I", 4: "IV", 9: "IX", 14: "XIV", 40: "XL", 90: "XC",
                 400: "CD", 900: "CM", 1994: "MCMXCIV", 2024: "MMXXIV",
                 3999: "MMMCMXCIX"}
        for n, s in cases.items():
            self.assertEqual(to_roman(n), s)
            self.assertEqual(from_roman(s), n)

    def test_round_trip(self):
        for n in range(1, 4000):
            self.assertEqual(from_roman(to_roman(n)), n)

    def test_lower_case(self):
        self.assertEqual(from_roman("mcmxciv"), 1994)

    def test_out_of_range(self):
        for n in (0, 4000, -5):
            with self.assertRaises(ValueError):
                to_roman(n)

    def test_invalid_numerals(self):
        for s in ("", "IIII", "VV", "IC", "ABC", "MMMM"):
            with self.assertRaises(ValueError):
                from_roman(s)
