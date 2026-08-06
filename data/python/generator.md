# Understanding Python Generators

Generators are functions that return an iterable set of items, one at a time, using the `yield` keyword instead of returning a full list.

## Benefits

- **Memory Efficiency**: Items are calculated on-the-fly rather than loaded into RAM all at once.
- **Lazy Evaluation**: Useful for processing massive datasets, log files, or streaming API responses in batches.
