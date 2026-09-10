# cron_prev

Steps backward a minute at a time from the minute before the given instant, with seconds zeroed, so the given instant itself is never returned. The search covers four years backward and gives NULL when nothing matches in that span.

## Syntax

```sql
cron_prev(cron, before)
```

## Returns

TIMESTAMPTZ as epoch microseconds. NULL when either argument is NULL, the expression does not parse, or nothing matches within four years.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `before` | Ending instant as epoch microseconds. | Not applicable. |

## Examples

```sql
SELECT cron_prev('@hourly', 0)
```

-3600000000, one hour before the epoch.

## Refused

- The instant argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cron_next](cron-next.md)
- [cron_between](cron-between.md)
