# xxhash128

A fast non-cryptographic hash with a wide output, so accidental collisions are vanishingly unlikely. It must not stand in for a digest, because a collision can be produced deliberately. The value is carried as a 128-bit integer with the bit pattern preserved, so it may read as negative.

## Syntax

```sql
xxhash128(bytes)
```

## Returns

The 128-bit hash as a signed integer. NULL when the input is NULL.

## Examples

```sql
SELECT xxhash128('abc') = xxhash128('abc')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [xxhash64](xxhash64.md)
- [xxhash32](xxhash32.md)
- [murmur3_128](murmur3-128.md)
- [blake3](blake3.md)
