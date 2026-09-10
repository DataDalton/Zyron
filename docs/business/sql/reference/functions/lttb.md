# lttb

Picks the points whose triangles with their neighbours have the largest area, which keeps peaks and troughs a plain every-nth sample would drop. The first and last points are always kept. A threshold at or above the point count, or below 3, returns every point. The result holds the chosen positions rather than the values.

## Syntax

```sql
lttb(timestamps, values, threshold)
```

## Returns

ARRAY of positions as JSON text. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `threshold` | How many points to keep. | Not applicable. |

## Examples

```sql
SELECT lttb('[1,2,3,4,5]', '[10,50,20,80,30]', 3)
```

Three positions, always including the first and the last.

## Refused

- An argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [moving_average](moving-average.md)
- [time_bucket_calendar](time-bucket-calendar.md)
