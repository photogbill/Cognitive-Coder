from inventory import Inventory
from report import format_report


def main():
    inv = Inventory()
    inv.add("bolt", 10, 0.25)
    inv.add("nut", 20, 0.1)
    print(format_report(inv))


if __name__ == "__main__":
    main()
