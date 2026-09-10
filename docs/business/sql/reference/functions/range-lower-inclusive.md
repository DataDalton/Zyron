# range_lower_inclusive

Reads the lower inclusivity flag. A range unbounded below reports true, because no value is excluded at that end.

## Syntax

```sql
range_lower_inclusive(range)
```

## Returns

BOOLEAN. NULL when the range is NULL.

## Examples

```sql
SELECT range_lower_inclusive(range_create(1, 10, true, false))
```

true.

## Refused

- The argument is not a binary or range column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [range_upper_inclusive](range-upper-inclusive.md)
- [range_lower](range-lower.md)
