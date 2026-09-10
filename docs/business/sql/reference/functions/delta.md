# delta

Subtracts the previous row's value from this one, in the window's order. The first row of each partition has no predecessor and gives NULL. The figure is a plain difference, so use rate to divide it by elapsed time.

## Syntax

```sql
delta(value) OVER (...)
```

## Returns

DOUBLE PRECISION. NULL on the first row of a partition.

## Examples

```sql
SELECT delta(reading) OVER (ORDER BY taken_at) FROM zyron_test.readings
```

The change at each reading, NULL on the first.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [rate](rate.md)
- [derivative](derivative.md)
- [lag](lag.md)
