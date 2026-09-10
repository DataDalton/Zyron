# array_length

Counts an array's elements, including NULL elements, each of which occupies a position. A NULL array returns NULL, not zero.

## Syntax

```sql
array_length(arr)
```

## Returns

BIGINT, or NULL for a null array.

## Examples

```sql
SELECT array_length(ARRAY[1, 2, 3])
```

3.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [UNNEST](../queries/unnest.md)
