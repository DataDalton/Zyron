# bitfield_not

Returns the bitwise complement over all 64 bits, so positions never used come back set. Mask the result with bitfield_and where only some positions carry meaning.

## Syntax

```sql
bitfield_not(bitfield)
```

## Returns

BITFIELD. NULL when the field is NULL.

## Examples

```sql
SELECT bitfield_count(bitfield_not(0))
```

64.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_and](bitfield-and.md)
- [bitfield_toggle](bitfield-toggle.md)
