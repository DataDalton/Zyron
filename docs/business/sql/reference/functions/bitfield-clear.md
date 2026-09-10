# bitfield_clear

Returns the field with the bit at the given position cleared, leaving every other bit alone. Clearing a bit already clear changes nothing.

## Syntax

```sql
bitfield_clear(bitfield, position)
```

## Returns

BITFIELD. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `position` | Bit to clear, from 0 to 63. | Not applicable. |

## Examples

```sql
SELECT bitfield_count(bitfield_clear(bitfield_set(0, 3), 3))
```

0.

## Refused

- The position is 64 or above.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_set](bitfield-set.md)
- [bitfield_toggle](bitfield-toggle.md)
