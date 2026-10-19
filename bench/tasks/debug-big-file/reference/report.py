from geometry import describe, regular_polygon

SQUARE = [(0, 0), (4, 0), (4, 4), (0, 4)]

if __name__ == "__main__":
    print(describe(SQUARE))
    print(describe(regular_polygon(6, 2.0)))
