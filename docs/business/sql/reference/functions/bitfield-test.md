# bitfield_test

Reads the bit at the given position. Use bitfield_all or bitfield_any to test several bits at once rather than calling this per bit.

## Syntax

```sql
bitfield_test(bitfield, position)
```

## Returns

BOOLEAN. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `position` | Bit to read, from 0 to 63. | Not applicable. |

## Examples

```sql
SELECT bitfield_test(bitfield_set(0, 7), 7)
```

true.

## Refused

- The position is 64 or above.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_all](bitfield-all.md)
- [bitfield_any](bitfield-any.md)
- [bitfield_set](bitfield-set.md)
