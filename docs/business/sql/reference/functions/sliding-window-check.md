# sliding_window_check

True when sliding_window_count is strictly below the limit, so a window already holding max_count events answers false and the request that would be the max_count-th is refused. A sliding window admits no burst at a boundary, which a fixed window does.

## Syntax

```sql
sliding_window_check(timestamps, window, max_count, now)
```

## Returns

BOOLEAN. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `max_count` | Event count at which the window is full. | Not applicable. |

## Examples

```sql
SELECT sliding_window_check('[0,500000,900000]', 1000000, 3, 1000000)
```

false, because the window already holds three events.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sliding_window_count](sliding-window-count.md)
- [token_bucket_consume](token-bucket-consume.md)
