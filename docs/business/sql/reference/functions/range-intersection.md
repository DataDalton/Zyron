# range_intersection

Takes the higher of the two lower bounds and the lower of the two upper bounds, carrying each kept bound's own inclusivity. Two ranges that do not overlap give the empty range rather than an error, where range_union refuses the same pair.

## Syntax

```sql
range_intersection(a, b)
```

## Returns

RANGE, empty when the two do not overlap. NULL when either range is NULL.

## Examples

```sql
SELECT range_contains_value(range_intersection(range_create(1, 10, true, false), range_create(5, 15, true, false)), 5)
```

true, because the intersection starts at 5.

## Refused

- Either argument is not a binary or range column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [range_union](range-union.md)
- [range_overlaps](range-overlaps.md)
- [range_is_empty](range-is-empty.md)
