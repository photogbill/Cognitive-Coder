import unittest

import textutils as t


class TestDuration(unittest.TestCase):
    def test_minutes_and_hours(self):
        self.assertEqual(t.parse_duration("90m"), 90)
        self.assertEqual(t.parse_duration("2h"), 120)
        self.assertEqual(t.parse_duration("1h30m"), 90)
        self.assertEqual(t.parse_duration(" 1H05M "), 65)

    def test_bad(self):
        for bad in ("", "h", "m", "90", "1h30", "abc", "1m30h"):
            with self.assertRaises(ValueError):
                t.parse_duration(bad)


class TestTheRestIsUntouched(unittest.TestCase):
    def test_text(self):
        self.assertEqual(t.slugify("Hello, World!"), "hello-world")
        self.assertEqual(t.title_case("the lord of the rings"),
                         "The Lord of the Rings")
        self.assertEqual(t.truncate("abcdefghij", 6), "abc...")
        self.assertTrue(t.is_palindrome("A man, a plan, a canal: Panama"))
        self.assertEqual(t.wrap("a bb ccc dddd", 6), ["a bb", "ccc", "dddd"])
        self.assertEqual(t.rot13("abc"), "nop")

    def test_numbers(self):
        self.assertEqual(t.format_bytes(1536), "1.5 KB")
        self.assertEqual(t.format_seconds(125), "2:05")
        self.assertEqual(t.format_seconds(3725), "1:02:05")
        self.assertEqual(t.median([3, 1, 2, 4]), 2.5)
        self.assertEqual(t.chunks([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])
        self.assertEqual(t.flatten([1, [2, [3]], (4,)]), [1, 2, 3, 4])
        self.assertTrue(t.parse_bool(" Yes "))
