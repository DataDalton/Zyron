# range_upper

Returns the 8-byte order key the range holds for its upper bound, not a number. The encoding is the same one range_lower returns, so the two compare against each other directly.

## Syntax

```sql
range_upper(range)
```

## Returns

BYTEA of 8 bytes. NULL when the range is NULL, empty, or unbounded above.

## Examples

```sql
SELECT range_upper(range_create(1, 10, true, false)) = range_lower(range_create(10, 20, true, false))
```

true, because one range ends where the next begins.

## Refused

- The argument is not a binary or range column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [range_lower](range-lower.md)
- [range_upper_inclusive](range-upper-inclusive.md)
- [range_adjacent](range-adjacent.md)
