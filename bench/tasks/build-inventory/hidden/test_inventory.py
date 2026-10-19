import unittest

from inventory import Inventory, Item
from report import format_report


class TestInventory(unittest.TestCase):
    def test_add_and_get(self):
        inv = Inventory()
        inv.add("bolt", 10, 0.25)
        inv.add("bolt", 5, 9.99)
        item = inv.get("bolt")
        self.assertIsInstance(item, Item)
        self.assertEqual(item.qty, 15)
        self.assertAlmostEqual(item.price, 0.25)
        self.assertIsNone(inv.get("nut"))

    def test_remove(self):
        inv = Inventory()
        inv.add("nut", 3, 0.1)
        inv.remove("nut", 2)
        self.assertEqual(inv.get("nut").qty, 1)
        with self.assertRaises(ValueError):
            inv.remove("nut", 5)
        inv.remove("nut", 1)
        self.assertIsNone(inv.get("nut"))
        with self.assertRaises(KeyError):
            inv.remove("washer", 1)

    def test_total_value(self):
        inv = Inventory()
        inv.add("a", 2, 1.5)
        inv.add("b", 4, 0.25)
        self.assertAlmostEqual(inv.total_value(), 4.0)

    def test_report(self):
        inv = Inventory()
        inv.add("zeta", 1, 2)
        inv.add("alpha", 3, 0.5)
        lines = format_report(inv).strip().splitlines()
        self.assertEqual(lines[0], "alpha: 3 @ 0.50")
        self.assertEqual(lines[1], "zeta: 1 @ 2.00")
        self.assertEqual(lines[-1], "TOTAL: 3.50")
