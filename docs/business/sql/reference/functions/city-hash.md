# city_hash

Names the same function as cityhash64.

## Syntax

```sql
city_hash(bytes)
```

## Returns

BIGINT holding the 64-bit hash. NULL when the input is NULL.

## Examples

```sql
SELECT city_hash('abc') = cityhash64('abc')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cityhash64](cityhash64.md)
