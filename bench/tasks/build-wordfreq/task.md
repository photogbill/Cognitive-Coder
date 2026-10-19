Write `wordfreq.py`, a word-frequency tool.

- `top_words(text, n)` returns the `n` most frequent words in `text` as a list of `(word, count)` tuples, most frequent first. Words are compared in lower case; punctuation is not part of a word (so "Hello," and "hello" are the same word); an apostrophe inside a word is kept ("don't" is one word). Words with the same count are in alphabetical order. If there are fewer than `n` different words, return them all. `n` of 0 returns an empty list; a negative `n` raises `ValueError`.
- Run as a program, `python wordfreq.py FILE [N]` prints the top N (default 10) words of the file, one per line as `word count`.
