from dataclasses import dataclass


@dataclass
class Item:
    name: str
    qty: int
    price: float


class Inventory:
    def __init__(self):
        self._items = {}

    def add(self, name, qty, price):
        if name in self._items:
            self._items[name].qty += qty
        else:
            self._items[name] = Item(name, qty, price)

    def remove(self, name, qty):
        if name not in self._items:
            raise KeyError(name)
        item = self._items[name]
        if qty > item.qty:
            raise ValueError(f"only {item.qty} {name} in stock")
        item.qty -= qty
        if item.qty == 0:
            del self._items[name]

    def get(self, name):
        return self._items.get(name)

    def items(self):
        return list(self._items.values())

    def total_value(self):
        return sum(i.qty * i.price for i in self._items.values())
