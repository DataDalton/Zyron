# cityhash64

Computes a 64-bit CityHash. It is a non-cryptographic hash with distribution comparable to xxhash64, and is the hash several other systems partition by, so it is the one to use when a partition assignment has to agree with theirs.

## Syntax

```sql
cityhash64(bytes)
```

## Returns

BIGINT. NULL when the argument is NULL.

## Examples

```sql
SELECT cityhash64('hello'::BYTEA)
```

A BIGINT hash.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [xxhash64](xxhash64.md)
- [fnv1a_64](fnv1a-64.md)
