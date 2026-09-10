# range_union

Takes the lower of the two lower bounds and the higher of the two upper bounds. The two ranges must overlap or be adjacent, because one range cannot hold a gap. A union with the empty range returns the other range unchanged.

## Syntax

```sql
range_union(a, b)
```

## Returns

RANGE. NULL when either range is NULL.

## Examples

```sql
SELECT range_contains_value(range_union(range_create(1, 5, true, false), range_create(5, 10, true, false)), 7)
```

true, because the union spans 1 up to 10.

## Refused

- The two ranges neither overlap nor meet.
- Either argument is not a binary or range column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [range_intersection](range-intersection.md)
- [range_adjacent](range-adjacent.md)
- [range_overlaps](range-overlaps.md)
