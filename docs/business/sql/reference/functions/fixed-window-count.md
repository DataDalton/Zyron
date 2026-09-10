# fixed_window_count

Divides the timestamp by the window width, rounding toward negative infinity so instants before the epoch land in the window below rather than sharing window 0. Grouping by this value counts events per window, at the cost of allowing a burst across a boundary that a sliding window refuses. A window width of 0 or below gives 0.

## Syntax

```sql
fixed_window_count(timestamp, window)
```

## Returns

BIGINT naming the window. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `window` | Window width in microseconds. | Not applicable. |

## Examples

```sql
SELECT fixed_window_count(1500000, 1000000)
```

1, the second one-second window.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sliding_window_count](sliding-window-count.md)
- [time_bucket_calendar](time-bucket-calendar.md)
