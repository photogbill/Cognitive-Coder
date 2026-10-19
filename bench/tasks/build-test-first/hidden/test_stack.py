import unittest

from stack import Stack


class TestStack(unittest.TestCase):
    def test_lifo(self):
        s = Stack()
        self.assertTrue(s.is_empty())
        s.push(1)
        s.push(2)
        self.assertEqual(len(s), 2)
        self.assertEqual(s.peek(), 2)
        self.assertEqual(s.pop(), 2)
        self.assertEqual(s.pop(), 1)
        self.assertTrue(s.is_empty())
        self.assertEqual(len(s), 0)

    def test_empty(self):
        s = Stack()
        with self.assertRaises(IndexError):
            s.pop()
        with self.assertRaises(IndexError):
            s.peek()
