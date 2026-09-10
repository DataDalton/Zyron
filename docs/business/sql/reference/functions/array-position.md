# array_position

Finds a value in an array and answers where it is, counting from 1. Elements are compared as values rather than as bytes, so a whole number literal finds a match in an INT[] whose elements are four bytes wide even though the literal binds to eight. A null is never found, because a null equals nothing.

## Syntax

```sql
array_position(arr, value)
```

## Returns

BIGINT, the 1-based position, or NULL when the value is absent.

## Examples

```sql
SELECT array_position(ARRAY[10, 20], 20)
```

2.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [array_contains](array-contains.md)
