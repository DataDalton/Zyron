# murmur3_32

A fast non-cryptographic hash suited to hash tables and partitioning. Two inputs can be made to collide deliberately, so it must not stand in for a digest. A different seed gives an unrelated hash for the same input, which separates independent hash tables.

## Syntax

```sql
murmur3_32(bytes [, seed])
```

## Returns

INTEGER holding the 32-bit hash. NULL when the input is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `seed` | Starting value, which shifts the whole hash space. | 0. |

## Examples

```sql
SELECT murmur3_32('abc') = murmur3_32('abc', 0)
```

true, because the default seed is 0.

## Refused

- Called with no arguments or more than two.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [murmur3_128](murmur3-128.md)
- [xxhash32](xxhash32.md)
- [consistent_hash](consistent-hash.md)
