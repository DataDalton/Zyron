# xxhash64

Computes a 64-bit xxHash. It distributes well enough for hash tables, bucketing and partition assignment, and runs faster than any cryptographic hash. It is not a cryptographic hash and offers no resistance to a chosen collision.

## Syntax

```sql
xxhash64(bytes)
```

## Returns

BIGINT. NULL when the argument is NULL.

## Examples

```sql
SELECT xxhash64('hello'::BYTEA)
```

A BIGINT hash.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [xxhash32](xxhash32.md)
- [cityhash64](cityhash64.md)
- [fnv1a_64](fnv1a-64.md)
- [siphash](siphash.md)
