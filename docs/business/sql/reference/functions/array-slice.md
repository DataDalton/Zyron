# array_slice

Takes a run of elements out of an array. The start counts from 1, a start below 1 reads as 1, and a length reaching past the end stops at the end rather than padding. A length of zero yields an empty array.

## Syntax

```sql
array_slice(arr, start, length)
```

## Returns

An array of the same element type, empty when the run is empty.

## Examples

```sql
SELECT array_slice(ARRAY[1, 2, 3, 4], 2, 2)
```

An array holding 2 and 3.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [array_concat](array-concat.md)
