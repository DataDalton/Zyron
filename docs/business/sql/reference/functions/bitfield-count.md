# bitfield_count

Counts the set bits, which is a single machine instruction rather than a loop over 64 positions.

## Syntax

```sql
bitfield_count(bitfield)
```

## Returns

INTEGER between 0 and 64. NULL when the field is NULL.

## Examples

```sql
SELECT bitfield_count(bitfield_set(bitfield_set(0, 1), 4))
```

2.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bitfield_to_positions](bitfield-to-positions.md)
- [bitfield_test](bitfield-test.md)
