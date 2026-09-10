# bitfield_and

Returns the bitwise AND, which keeps a bit only where both fields hold it. Use it to mask a field down to a subset of flags.

## Syntax

```sql
bitfield_and(a, b)
```

## Returns

BITFIELD. NULL when either argument is NULL.

## Examples

```sql
SELECT bitfield_count(bitfield_and(3, 1))
```

1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_or](bitfield-or.md)
- [bitfield_xor](bitfield-xor.md)
- [bitfield_all](bitfield-all.md)
