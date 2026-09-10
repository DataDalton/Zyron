# bitfield_from_positions

Sets one bit per position in the array, inverting bitfield_to_positions. A position of 64 or above is skipped rather than refused, unlike bitfield_set, so a list holding one is built from the rest.

## Syntax

```sql
bitfield_from_positions(positions)
```

## Returns

BITFIELD. NULL when the array is NULL or does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `positions` | JSON array of bit positions. | Not applicable. |

## Examples

```sql
SELECT bitfield_to_positions(bitfield_from_positions('[1,4]'))
```

[1,4].

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_to_positions](bitfield-to-positions.md)
- [bitfield_set](bitfield-set.md)
