# bitfield_toggle

Returns the field with the bit at the given position inverted. Toggling the same position twice returns the original field.

## Syntax

```sql
bitfield_toggle(bitfield, position)
```

## Returns

BITFIELD. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `position` | Bit to flip, from 0 to 63. | Not applicable. |

## Examples

```sql
SELECT bitfield_test(bitfield_toggle(0, 5), 5)
```

true.

## Refused

- The position is 64 or above.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_set](bitfield-set.md)
- [bitfield_clear](bitfield-clear.md)
- [bitfield_xor](bitfield-xor.md)
