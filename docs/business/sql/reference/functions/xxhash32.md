# xxhash32

Computes a 32-bit xxHash. It costs less than xxhash64 and collides far sooner: with 32 bits, a collision becomes likely around 65,000 distinct values. Prefer xxhash64 where the key count is not small and known.

## Syntax

```sql
xxhash32(bytes)
```

## Returns

INTEGER. NULL when the argument is NULL.

## Examples

```sql
SELECT xxhash32('hello'::BYTEA)
```

An INTEGER hash.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [xxhash64](xxhash64.md)
