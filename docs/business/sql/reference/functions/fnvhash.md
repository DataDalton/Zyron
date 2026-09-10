# fnvhash

Names the same function as fnv1a_64.

## Syntax

```sql
fnvhash(bytes)
```

## Returns

BIGINT holding the 64-bit hash. NULL when the input is NULL.

## Examples

```sql
SELECT fnvhash('abc') = fnv1a_64('abc')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [fnv1a_64](fnv1a-64.md)
