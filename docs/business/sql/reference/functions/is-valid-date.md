# is_valid_date

Reads the text against the layout and answers whether it parsed and named a real date, so 2026-02-30 fails on the day rather than only on the shape.

## Syntax

```sql
is_valid_date(text, format)
```

## Returns

BOOLEAN. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `format` | Layout built from YYYY for the year, MM for the month, DD for the day, HH for the hour, mm for the minute and SS for the second, with any other characters matched literally. | Not applicable. |

## Examples

```sql
SELECT is_valid_date('2026-02-30', 'YYYY-MM-DD')
```

false, because February has no 30th.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [parse_natural_date](parse-natural-date.md)
- [day_of_week](day-of-week.md)
