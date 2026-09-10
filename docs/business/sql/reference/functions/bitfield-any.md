# bitfield_any

True when the field and the mask share a bit. An empty mask shares nothing, so a mask of 0 gives false, the opposite of bitfield_all.

## Syntax

```sql
bitfield_any(bitfield, mask)
```

## Returns

BOOLEAN. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `mask` | Bits any one of which is enough. | Not applicable. |

## Examples

```sql
SELECT bitfield_any(bitfield_set(0, 0), 3)
```

true, because bit 0 is one of the mask's bits.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_all](bitfield-all.md)
- [bitfield_or](bitfield-or.md)
