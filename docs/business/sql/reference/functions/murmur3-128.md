# murmur3_128

A fast non-cryptographic hash with a wide output, so accidental collisions are vanishingly unlikely even across very many values. The digest is carried as a 128-bit integer with the bit pattern preserved, so it may read as negative.

## Syntax

```sql
murmur3_128(bytes [, seed])
```

## Returns

The 128-bit hash as a signed integer. NULL when the input is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `seed` | Starting value, which shifts the whole hash space. | 0. |

## Examples

```sql
SELECT murmur3_128('abc') = murmur3_128('abc', 0)
```

true.

## Refused

- Called with no arguments or more than two.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [murmur3_32](murmur3-32.md)
- [xxhash128](xxhash128.md)
