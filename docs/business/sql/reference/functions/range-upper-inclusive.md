# range_upper_inclusive

Reads the upper inclusivity flag. A range unbounded above reports true, because no value is excluded at that end.

## Syntax

```sql
range_upper_inclusive(range)
```

## Returns

BOOLEAN. NULL when the range is NULL.

## Examples

```sql
SELECT range_upper_inclusive(range_create(1, 10, true, false))
```

false.

## Refused

- The argument is not a binary or range column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [range_lower_inclusive](range-lower-inclusive.md)
- [range_upper](range-upper.md)
