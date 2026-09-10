# rate

Divides the change in value by the seconds between the two rows, taking the time from the window's ORDER BY column. An ORDER BY is required, because without it there is no time to divide by. Two rows at the same instant, or out of order, give NULL rather than a division by zero.

## Syntax

```sql
rate(value) OVER (ORDER BY time ...)
```

## Returns

DOUBLE PRECISION per second. NULL on the first row of a partition and where no time passed.

## Examples

```sql
SELECT rate(counter) OVER (ORDER BY taken_at) FROM zyron_test.readings
```

The counter's change per second, NULL on the first row.

## Refused

- The window states no ORDER BY.
- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [delta](delta.md)
- [derivative](derivative.md)
