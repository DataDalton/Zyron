# cron_next

Steps forward a minute at a time from the minute after the given instant, with seconds zeroed, so the given instant itself is never returned even when it matches. The search covers four years, which bounds the cost of an expression such as a 29 February that only falls in a leap year. An expression that matches nothing in four years gives NULL.

## Syntax

```sql
cron_next(cron, after)
```

## Returns

TIMESTAMPTZ as epoch microseconds. NULL when either argument is NULL, the expression does not parse, or nothing matches within four years.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `after` | Starting instant as epoch microseconds. | Not applicable. |

## Examples

```sql
SELECT cron_next('@hourly', 0)
```

3600000000, one hour after the epoch.

## Refused

- The instant argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cron_prev](cron-prev.md)
- [cron_list](cron-list.md)
- [cron_between](cron-between.md)
- [cron_matches](cron-matches.md)
