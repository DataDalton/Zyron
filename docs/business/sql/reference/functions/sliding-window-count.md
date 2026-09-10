# sliding_window_count

Counts the timestamps at or after the cutoff, which is now less the window. The timestamps are sorted first and the count found by binary search, so the cost grows with the logarithm of the event count rather than with the count itself.

## Syntax

```sql
sliding_window_count(timestamps, window, now)
```

## Returns

BIGINT. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `window` | Window width in microseconds. | Not applicable. |
| `now` | Instant the window ends at, as epoch microseconds. | Not applicable. |

## Examples

```sql
SELECT sliding_window_count('[0,500000,900000]', 1000000, 1000000)
```

3, because all three fall inside the last second.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sliding_window_check](sliding-window-check.md)
- [fixed_window_count](fixed-window-count.md)
