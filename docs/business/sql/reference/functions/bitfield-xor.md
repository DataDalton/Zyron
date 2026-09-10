# bitfield_xor

Returns the bitwise exclusive or, which keeps a bit where exactly one field holds it. The result names the bits that differ between the two fields, so bitfield_count over it is the Hamming distance.

## Syntax

```sql
bitfield_xor(a, b)
```

## Returns

BITFIELD. NULL when either argument is NULL.

## Examples

```sql
SELECT bitfield_count(bitfield_xor(3, 1))
```

1, the one bit the two fields differ in.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_and](bitfield-and.md)
- [bitfield_or](bitfield-or.md)
- [simhash_distance](simhash-distance.md)
