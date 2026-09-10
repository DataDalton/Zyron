# bitfield_all

True when the field holds all of the mask's bits. An empty mask is held by every field, so a mask of 0 gives true.

## Syntax

```sql
bitfield_all(bitfield, mask)
```

## Returns

BOOLEAN. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `mask` | Bits that must all be present. | Not applicable. |

## Examples

```sql
SELECT bitfield_all(bitfield_set(bitfield_set(0, 0), 1), 3)
```

true, because bits 0 and 1 together are the mask 3.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_any](bitfield-any.md)
- [bitfield_and](bitfield-and.md)
- [bitfield_test](bitfield-test.md)
