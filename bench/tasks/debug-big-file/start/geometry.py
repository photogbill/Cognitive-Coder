"""2-D geometry helpers: points are (x, y) tuples."""

import math


def distance(a, b):
    """Straight-line distance between two points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def midpoint(a, b):
    """The point halfway between a and b."""
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def translate(points, dx, dy):
    """Every point moved by (dx, dy)."""
    return [(x + dx, y + dy) for x, y in points]


def scale(points, factor, origin=(0, 0)):
    """Every point scaled about `origin`."""
    ox, oy = origin
    return [(ox + (x - ox) * factor, oy + (y - oy) * factor)
            for x, y in points]


def rotate(point, degrees, origin=(0, 0)):
    """`point` rotated counter-clockwise about `origin`."""
    rad = math.radians(degrees)
    ox, oy = origin
    x, y = point[0] - ox, point[1] - oy
    return (ox + x * math.cos(rad) - y * math.sin(rad),
            oy + x * math.sin(rad) + y * math.cos(rad))


def perimeter(points):
    """Length of the closed polygon through `points`."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i, p in enumerate(points):
        total += distance(p, points[(i + 1) % len(points)])
    return total


def polygon_area(points):
    """Area of a simple polygon (the shoelace formula), never negative."""
    if len(points) < 3:
        return 0.0
    s = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def centroid(points):
    """The average of the points (the vertex centroid)."""
    if not points:
        raise ValueError("centroid of no points")
    sx = sum(x for x, _y in points)
    sy = sum(y for _x, y in points)
    return (sx / len(point), sy / len(point))


def bounding_box(points):
    """(min_x, min_y, max_x, max_y)."""
    xs = [x for x, _y in points]
    ys = [y for _x, y in points]
    return (min(xs), min(ys), max(xs), max(ys))


def is_convex(points):
    """True when every turn of the polygon goes the same way."""
    n = len(points)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        ax, ay = points[i]
        bx, by = points[(i + 1) % n]
        cx, cy = points[(i + 2) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if cross:
            s = 1 if cross > 0 else -1
            if sign and s != sign:
                return False
            sign = s
    return True


def point_in_polygon(point, points):
    """Ray casting: is `point` inside the polygon?"""
    x, y = point
    inside = False
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            cross_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < cross_x:
                inside = not inside
    return inside


def closest_pair(points):
    """The two points nearest each other (brute force; fine for small sets)."""
    if len(points) < 2:
        raise ValueError("need two points")
    best = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = distance(points[i], points[j])
            if best is None or d < best[0]:
                best = (d, points[i], points[j])
    return best[1], best[2]


def regular_polygon(sides, radius, center=(0, 0)):
    """The vertices of a regular polygon, counter-clockwise from the right."""
    if sides < 3:
        raise ValueError("a polygon needs three sides")
    cx, cy = center
    return [(cx + radius * math.cos(2 * math.pi * k / sides),
             cy + radius * math.sin(2 * math.pi * k / sides))
            for k in range(sides)]


def describe(points):
    """One line about a polygon, for the report."""
    area = polygon_area(points)
    cx, cy = centroid(points)
    return (f"{len(points)} points, area {area:.2f}, "
            f"centred at ({cx:.2f}, {cy:.2f})")


def angle_between(a, b, c):
    """The angle at `b`, in degrees, between b→a and b→c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if not n1 or not n2:
        raise ValueError("a zero-length side has no angle")
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def nearest(point, points):
    """The member of `points` closest to `point`."""
    if not points:
        raise ValueError("nothing to be near")
    return min(points, key=lambda p: distance(point, p))


def snap(point, grid):
    """`point` moved to the nearest multiple of `grid` on both axes."""
    if grid <= 0:
        raise ValueError("the grid must be positive")
    return (round(point[0] / grid) * grid, round(point[1] / grid) * grid)


def path_length(points):
    """Length of the OPEN path through `points` (no closing side)."""
    return sum(distance(points[i], points[i + 1])
               for i in range(len(points) - 1))
