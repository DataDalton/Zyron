# ema

Emits the first row's value unchanged, then weights each row against the running average. Weight on older rows decays geometrically rather than dropping off at a frame edge. Alpha is clamped to between 0.0 and 1.0.

## Syntax

```sql
ema(value [, alpha]) OVER (...)
```

## Returns

DOUBLE PRECISION. Never NULL for a non-NULL input.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `alpha` | Weight on the current row. Lower values smooth harder. | 0.5. |

## Examples

```sql
SELECT ema(reading, 0.3) OVER (ORDER BY taken_at) FROM zyron_test.readings
```

A smoothed series starting at the first reading.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [moving_avg](moving-avg.md)
- [exponential_smoothing](exponential-smoothing.md)
