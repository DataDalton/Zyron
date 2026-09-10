# moving_avg

Averages the rows the frame covers. A ROWS frame counts rows, a RANGE frame covers a span of ordering values or of time, and with no frame stated the second argument fixes a row count. Use a RANGE frame where readings arrive unevenly, because a row count then covers a varying span of time.

## Syntax

```sql
moving_avg(value [, window]) OVER (...)
```

## Returns

DOUBLE PRECISION. NULL when the frame holds no non-NULL value.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `window` | Fixed row count, used only when the window states no frame. | Not applicable. |

## Examples

```sql
SELECT moving_avg(reading) OVER (ORDER BY taken_at ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) FROM zyron_test.readings
```

The mean of each reading and the four before it.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ema](ema.md)
- [moving_average](moving-average.md)
