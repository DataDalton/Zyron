# cron_parse

Reads a five-field cron expression and returns a 19-byte parsed form holding one bit per allowed minute, hour, day of month, month and day of week. Every other cron function accepts either the text or this parsed form, so parsing once and carrying the result avoids reparsing the same text per row. Text that does not parse gives NULL for that row rather than failing the statement.

## Syntax

```sql
cron_parse(text)
```

## Returns

BYTEA, 19 bytes. NULL when the text is NULL or does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `text` | Five fields separated by spaces, or one of the shortcuts @yearly, @annually, @monthly, @weekly, @daily, @midnight and @hourly. A field accepts a star, a single value, a range such as 1-5, a step such as */5 and a comma list. Month accepts JAN to DEC and day of week accepts SUN to SAT, matched without regard to case. Day of week accepts 7 for Sunday as well as 0. | Not applicable. |

## Examples

```sql
SELECT cron_human_readable(cron_parse('0 9 * * 1-5'))
```

Minute: 0, Hour: 9, Day: *, Month: *, Weekday: 1,2,3,4,5.

## Refused

- The argument is neither text nor the parsed form.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cron_next](cron-next.md)
- [cron_matches](cron-matches.md)
- [cron_human_readable](cron-human-readable.md)
