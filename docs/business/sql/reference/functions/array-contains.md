# array_contains

Answers whether an array holds a value, comparing elements as values rather than as bytes. A null is never found, so array_contains over a null value is false rather than null.

## Syntax

```sql
array_contains(arr, value)
```

## Returns

BOOLEAN, or NULL for a null array.

## Examples

```sql
SELECT array_contains(ARRAY[10, 20], 30)
```

false.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [array_position](array-position.md)
