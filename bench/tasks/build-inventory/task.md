A small stock-keeping program in Python, in three files:

- `inventory.py`: a dataclass `Item` with `name: str`, `qty: int` and `price: float`; and a class `Inventory` with `add(name, qty, price)` (adding to an item that exists increases its quantity and keeps its price), `remove(name, qty)` (raises `ValueError` if there is not enough, `KeyError` if the item is unknown; an item whose quantity reaches 0 is removed), `get(name)` returning the `Item` or `None`, and `total_value()` returning the sum of qty × price.
- `report.py`: `format_report(inventory)` returning one line per item, sorted by name, as `name: qty @ price` with the price to two decimals, then a last line `TOTAL: value` to two decimals.
- `main.py`: builds an inventory with a few items and prints the report.
