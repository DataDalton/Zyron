# range_adjacent

True when one range's upper bound equals the other's lower bound and exactly one of those two sides includes it. Two ranges both including the shared bound overlap instead, and two both excluding it leave the bound in neither. An empty range is adjacent to nothing. range_union accepts an adjacent pair, so adjacency decides whether two ranges that do not overlap can still be joined.

## Syntax

```sql
range_adjacent(a, b)
```

## Returns

BOOLEAN. NULL when either range is NULL.

## Examples

```sql
SELECT range_adjacent(range_create(1, 5, true, false), range_create(5, 10, true, false))
```

true.

## Refused

- Either argument is not a binary or range column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [range_union](range-union.md)
- [range_overlaps](range-overlaps.md)
