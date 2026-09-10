# range_is_empty

Reads the empty flag the encoding carries. A range becomes empty at creation, when its lower bound is above its upper or the two are equal with either side excluded, and as the result of an intersection of two ranges that do not overlap.

## Syntax

```sql
range_is_empty(range)
```

## Returns

BOOLEAN. NULL when the range is NULL.

## Examples

```sql
SELECT range_is_empty(range_create(5, 5, true, false))
```

true, because an excluded upper bound leaves nothing between.

## Refused

- The argument is not a binary or range column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [range_create](range-create.md)
- [range_intersection](range-intersection.md)
