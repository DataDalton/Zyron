# hash_combine

Mixes the second hash into the first so the result depends on both and on their order, which makes a single hash for a composite key. Combining is not commutative, so the two arguments must be passed in a fixed order for the result to be stable.

## Syntax

```sql
hash_combine(a, b)
```

## Returns

BIGINT. NULL when either argument is NULL.

## Examples

```sql
SELECT hash_combine(fnv1a_64('a'), fnv1a_64('b')) = hash_combine(fnv1a_64('b'), fnv1a_64('a'))
```

false, because the order of the two hashes matters.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [fnv1a_64](fnv1a-64.md)
- [xxhash64](xxhash64.md)
- [consistent_hash](consistent-hash.md)
