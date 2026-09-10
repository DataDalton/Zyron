# fnv1a_64

Computes the 64-bit FNV-1a hash. Its loop is a multiply and an exclusive-or per byte, which makes it quick on short keys and slower than xxhash64 on long ones. It is not a cryptographic hash.

## Syntax

```sql
fnv1a_64(bytes)
```

## Returns

BIGINT. NULL when the argument is NULL.

## Examples

```sql
SELECT fnv1a_64('hello'::BYTEA)
```

A BIGINT hash.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [xxhash64](xxhash64.md)
- [cityhash64](cityhash64.md)
