import unittest

from todo import TodoList, demo


class TestTodo(unittest.TestCase):
    def test_existing_behaviour_survives(self):
        t = TodoList()
        self.assertEqual(t.add(" a "), 0)
        t.add("b")
        t.complete(0)
        self.assertEqual(t.pending(), ["b"])
        self.assertEqual(t.summary(), "1 of 2 done")
        with self.assertRaises(ValueError):
            t.add("   ")
        demo()

    def test_remove(self):
        t = TodoList()
        t.add("a")
        t.add("b")
        self.assertEqual(t.remove(0), "a")
        self.assertEqual(t.pending(), ["b"])
        for bad in (5, -1):
            with self.assertRaises(IndexError):
                t.remove(bad)
