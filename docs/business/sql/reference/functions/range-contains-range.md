# range_contains_range

Checks that the inner range starts no earlier and ends no later than the outer one, treating a shared bound as held only where the outer side is at least as inclusive. The outer range comes first. An empty inner range is held by any range.

## Syntax

```sql
range_contains_range(outer, inner)
```

## Returns

BOOLEAN. NULL when either range is NULL.

## Examples

```sql
SELECT range_contains_range(range_create(1, 10, true, false), range_create(2, 5, true, false))
```

true.

## Refused

- Either argument is not a binary or range column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [range_contains_value](range-contains-value.md)
- [range_overlaps](range-overlaps.md)
