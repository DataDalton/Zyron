# bitfield_set

Returns the field with the bit at the given position set, leaving every other bit alone. Setting a bit already set changes nothing. Positions run from 0 for the lowest bit to 63 for the highest, so a bitfield holds 64 flags in the space of one integer.

## Syntax

```sql
bitfield_set(bitfield, position)
```

## Returns

BITFIELD. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `position` | Bit to set, from 0 to 63. | Not applicable. |

## Examples

```sql
SELECT bitfield_count(bitfield_set(0, 3))
```

1.

## Refused

- The position is 64 or above.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_clear](bitfield-clear.md)
- [bitfield_toggle](bitfield-toggle.md)
- [bitfield_test](bitfield-test.md)
