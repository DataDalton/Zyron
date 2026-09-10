# cron_matches

Tests the instant's minute, hour, day of month, month and day of week against the expression. Seconds are not tested, so every instant within a matching minute matches. When both the day of month and the day of week fields are restricted, a day matching either one matches, following cron's own rule rather than requiring both.

## Syntax

```sql
cron_matches(cron, timestamp)
```

## Returns

BOOLEAN. NULL when either argument is NULL or the expression does not parse.

## Examples

```sql
SELECT cron_matches('@daily', 0)
```

true, because the epoch is midnight.

## Refused

- The instant argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cron_next](cron-next.md)
- [cron_between](cron-between.md)
- [cron_parse](cron-parse.md)
