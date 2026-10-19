from shop import receipt

ORDER = [("tea", 2, 3.50), ("cake", 1, 4.00)]

if __name__ == "__main__":
    print(receipt(ORDER, "10%"))
