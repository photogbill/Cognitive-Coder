import math
import unittest

import geometry as g

SQUARE = [(0, 0), (4, 0), (4, 4), (0, 4)]


class TestGeometry(unittest.TestCase):
    def test_centroid(self):
        self.assertEqual(g.centroid(SQUARE), (2.0, 2.0))
        with self.assertRaises(ValueError):
            g.centroid([])

    def test_describe(self):
        self.assertEqual(g.describe(SQUARE),
                         "4 points, area 16.00, centred at (2.00, 2.00)")

    def test_the_rest_is_untouched(self):
        self.assertEqual(g.polygon_area(list(reversed(SQUARE))), 16.0)
        self.assertEqual(g.perimeter(SQUARE), 16.0)
        self.assertTrue(g.is_convex(SQUARE))
        self.assertTrue(g.point_in_polygon((1, 1), SQUARE))
        self.assertFalse(g.point_in_polygon((5, 1), SQUARE))
        self.assertEqual(g.bounding_box(SQUARE), (0, 0, 4, 4))
        x, y = g.rotate((1, 0), 90)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.0)
        hexagon = g.regular_polygon(6, 1.0)
        self.assertEqual(len(hexagon), 6)
        self.assertAlmostEqual(g.perimeter(hexagon), 6.0)
        self.assertAlmostEqual(g.distance((0, 0), (3, 4)), 5.0)
        self.assertTrue(math.isclose(g.polygon_area(hexagon),
                                     3 * math.sqrt(3) / 2))
