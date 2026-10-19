def format_report(inventory):
    lines = [f"{i.name}: {i.qty} @ {i.price:.2f}"
             for i in sorted(inventory.items(), key=lambda i: i.name)]
    lines.append(f"TOTAL: {inventory.total_value():.2f}")
    return "\n".join(lines)
