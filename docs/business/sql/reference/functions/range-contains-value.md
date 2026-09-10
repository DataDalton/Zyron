# range_contains_value

Encodes the value the same way the bounds are encoded and compares it against both, respecting each side's inclusivity. An unbounded side holds every value beyond it. The empty range holds nothing.

## Syntax

```sql
range_contains_value(range, value)
```

## Returns

BOOLEAN. NULL when the range or the value is NULL.

## Examples

```sql
SELECT range_contains_value(range_create(1, 10, true, false), 5)
```

true.

## Refused

- The value argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [range_contains_range](range-contains-range.md)
- [range_overlaps](range-overlaps.md)
- [range_create](range-create.md)
