# array_distinct

Removes duplicate elements, keeping the first appearance of each so the result reads in the order the input did rather than in an order nothing asked for. A null element is an element like any other and appears once.

## Syntax

```sql
array_distinct(arr)
```

## Returns

An array of the same element type.

## Examples

```sql
SELECT array_distinct(ARRAY[2, 1, 2])
```

An array holding 2 then 1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [array_sort](array-sort.md)
