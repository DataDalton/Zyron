# derivative

Computes the same figure rate does, taking the time from a second argument where one is given and from the window's ORDER BY otherwise. Pass the time explicitly to measure against a column other than the one the window orders by.

## Syntax

```sql
derivative(value [, time]) OVER (...)
```

## Returns

DOUBLE PRECISION per second. NULL on the first row of a partition and where no time passed.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `time` | Timestamp column to measure against. | The window's ORDER BY column. |

## Examples

```sql
SELECT derivative(reading, taken_at) OVER (ORDER BY id) FROM zyron_test.readings
```

The change per second measured against taken_at.

## Refused

- Neither a time argument nor an ORDER BY is given.
- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [rate](rate.md)
- [delta](delta.md)
