"""A to-do list kept in memory."""


class TodoList:
    def __init__(self):
        self.items = []

    def add(self, text):
        """Add an item; returns its index."""
        if not text.strip():
            raise ValueError("an item needs some text")
        self.items.append({"text": text.strip(), "done": False})
        return len(self.items) - 1

    def complete(self, index):
        """Mark the item at `index` as done."""
        self.items[index]["done"] = True

    def pending(self):
        """The texts of the items not done yet, in order."""
        return [i["text"] for i in self.items if not i["done"]]

    def summary(self):
        done = sum(1 for i in self.items if i["done"])
        return f"{done} of {len(self.items)} done"


def demo():
    todo = TodoList()
    todo.add("buy milk")
    todo.add("fix the fence")
    todo.complete(0)
    print(todo.summary())


if __name__ == "__main__":
    demo()
