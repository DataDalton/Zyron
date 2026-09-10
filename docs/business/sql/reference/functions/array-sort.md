# array_sort

Sorts an array ascending by the values its elements hold, not by their bytes. A null sorts after every value, which is where an ascending order puts an absent one.

## Syntax

```sql
array_sort(arr)
```

## Returns

An array of the same element type.

## Examples

```sql
SELECT array_sort(ARRAY[3, 1, 2])
```

An array holding 1, 2, 3.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [array_distinct](array-distinct.md)
