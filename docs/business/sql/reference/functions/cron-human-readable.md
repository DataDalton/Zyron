# cron_human_readable

Writes the five fields as Minute, Hour, Day, Month and Weekday, each listing the values it allows or a star where it allows all of them. A range or a step is expanded to the values it covers, so */15 reads as 0,15,30,45. The output names the fields rather than forming a sentence.

## Syntax

```sql
cron_human_readable(cron)
```

## Returns

TEXT. NULL when the expression is NULL or does not parse.

## Examples

```sql
SELECT cron_human_readable('*/15 * * * *')
```

Minute: 0,15,30,45, Hour: *, Day: *, Month: *, Weekday: *.

## Refused

- The argument is neither text nor the parsed form.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cron_parse](cron-parse.md)
- [cron_matches](cron-matches.md)
