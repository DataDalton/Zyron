# array_concat

Joins arrays end to end in the order they are written. A null array contributes nothing, and a call where every argument is null yields null.

## Syntax

```sql
array_concat(arr, arr [, arr ...])
```

## Returns

An array of the element type the inputs agree on.

## Examples

```sql
SELECT array_concat(ARRAY[1], ARRAY[2, 3])
```

An array holding 1, 2, 3.

## Refused

- Fewer than two arrays are given.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [array_slice](array-slice.md)
