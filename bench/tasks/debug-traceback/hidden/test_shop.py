import unittest

from shop import apply_discount, receipt, subtotal


class TestShop(unittest.TestCase):
    def test_discount_forms(self):
        self.assertEqual(apply_discount(200, 10), 180)
        self.assertEqual(apply_discount(200, "10%"), 180)
        self.assertEqual(apply_discount(200, " 25 % "), 150)
        self.assertEqual(apply_discount(200, 0), 200)

    def test_receipt(self):
        order = [("tea", 2, 3.50), ("cake", 1, 4.00)]
        self.assertEqual(subtotal(order), 11.0)
        self.assertEqual(receipt(order, "10%"), "total 9.90")
        self.assertEqual(receipt(order), "total 11.00")
