Write a Python module `roman.py` with two functions:

- `to_roman(n)` turns an integer from 1 to 3999 into a Roman numeral string, using subtractive notation (4 is "IV", 9 is "IX", 40 is "XL", 90 is "XC", 400 is "CD", 900 is "CM").
- `from_roman(s)` turns a Roman numeral string back into an integer. It accepts upper or lower case.

Both MUST raise `ValueError` for anything outside that range or that is not a valid numeral: 0, 4000, a negative number, an empty string, "IIII", "VV", "IC", or a string with other letters.
