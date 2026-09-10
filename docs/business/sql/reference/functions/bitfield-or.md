# bitfield_or

Returns the bitwise OR, which keeps a bit where either field holds it. Use it to merge two sets of flags.

## Syntax

```sql
bitfield_or(a, b)
```

## Returns

BITFIELD. NULL when either argument is NULL.

## Examples

```sql
SELECT bitfield_count(bitfield_or(1, 2))
```

2.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_and](bitfield-and.md)
- [bitfield_xor](bitfield-xor.md)
- [bitfield_any](bitfield-any.md)
