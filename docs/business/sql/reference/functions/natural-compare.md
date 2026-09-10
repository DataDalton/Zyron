# natural_compare

Returns -1, 0 or 1. A run of digits compares by value rather than character by character, so item2 sorts before item10 where a plain comparison puts item10 first. Leading zeros do not change a run's value.

## Syntax

```sql
natural_compare(a, b)
```

## Returns

INTEGER, one of -1, 0 or 1. NULL when either string is NULL.

## Examples

```sql
SELECT natural_compare('item2', 'item10')
```

-1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [natural_sort_key](natural-sort-key.md)
- [path_compare](path-compare.md)
- [version_compare](version-compare.md)
