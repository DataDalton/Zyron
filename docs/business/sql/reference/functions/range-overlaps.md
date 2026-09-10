# range_overlaps

True when at least one value falls in both ranges. Two ranges meeting at a bound only overlap when both sides include it, so a pair that is adjacent does not overlap. An empty range overlaps nothing.

## Syntax

```sql
range_overlaps(a, b)
```

## Returns

BOOLEAN. NULL when either range is NULL.

## Examples

```sql
SELECT range_overlaps(range_create(1, 10, true, false), range_create(5, 15, true, false))
```

true.

## Refused

- Either argument is not a binary or range column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [range_adjacent](range-adjacent.md)
- [range_intersection](range-intersection.md)
- [range_contains_range](range-contains-range.md)
