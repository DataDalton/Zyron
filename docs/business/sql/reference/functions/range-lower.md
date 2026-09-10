# range_lower

Returns the 8-byte order key the range holds for its lower bound, not a number. Compare two bounds as bytes, because the encoding is sign-flipped big-endian and so orders the same way the values do.

## Syntax

```sql
range_lower(range)
```

## Returns

BYTEA of 8 bytes. NULL when the range is NULL, empty, or unbounded below.

## Examples

```sql
SELECT range_lower(range_create(1, 10, true, false)) = range_lower(range_create(1, 20, true, false))
```

true, because both ranges start at the same bound.

## Refused

- The argument is not a binary or range column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [range_upper](range-upper.md)
- [range_lower_inclusive](range-lower-inclusive.md)
- [range_create](range-create.md)
