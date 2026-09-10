# bitfield_to_positions

Lists the set bit positions in ascending order, which makes a field readable and joinable against a table of flag names. An empty field gives an empty array.

## Syntax

```sql
bitfield_to_positions(bitfield)
```

## Returns

ARRAY of positions as JSON text. NULL when the field is NULL.

## Examples

```sql
SELECT bitfield_to_positions(bitfield_set(bitfield_set(0, 1), 4))
```

[1,4].

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_from_positions](bitfield-from-positions.md)
- [bitfield_count](bitfield-count.md)
