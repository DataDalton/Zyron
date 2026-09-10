# cron_list

Walks forward from the given instant collecting firing times until the count is reached. The count is clamped to between 0 and 1000, so a larger request returns 1000 rather than failing. Use this where the number of firings matters and cron_between where the window does.

## Syntax

```sql
cron_list(cron, after, count)
```

## Returns

ARRAY of epoch microseconds as JSON text. NULL when any argument is NULL or the expression does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `count` | How many firings to return, clamped to at most 1000. | Not applicable. |

## Examples

```sql
SELECT cron_list('@hourly', 0, 2)
```

[3600000000,7200000000].

## Refused

- An instant or count argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cron_between](cron-between.md)
- [cron_next](cron-next.md)
