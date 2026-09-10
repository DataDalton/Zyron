# cron_between

Returns the firing times from the start instant up to but not including the end instant, so a window whose end lands exactly on a firing excludes it. A window holding more than a million firings gives NULL rather than a large array.

## Syntax

```sql
cron_between(cron, start, end)
```

## Returns

ARRAY of epoch microseconds as JSON text. NULL when any argument is NULL, the expression does not parse, or the window holds more than a million firings.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `start` | First instant in the window, included when it matches. | Not applicable. |
| `end` | Instant the window stops before. | Not applicable. |

## Examples

```sql
SELECT cron_between('@hourly', 0, 10800000000)
```

[0,3600000000,7200000000], three firings in a three hour window.

## Refused

- An instant argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cron_list](cron-list.md)
- [cron_next](cron-next.md)
- [cron_matches](cron-matches.md)
