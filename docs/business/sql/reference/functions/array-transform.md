# array_transform

Replaces each element with what a lambda yields for it, keeping the array's order and length. The element type of the result is the lambda body's type, which need not be the input's. Like array_filter, the body runs once over every element of the batch rather than once per element.

## Syntax

```sql
array_transform(arr, x -> expr)
```

## Returns

An array whose element type is the lambda body's type.

## Examples

```sql
SELECT array_transform(ARRAY[1, 2], x -> x * 10)
```

An array holding 10 and 20.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [array_filter](array-filter.md)
