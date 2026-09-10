# parse_natural_date

Accepts today, now, yesterday and tomorrow, the phrases beginning of month, start of month, end of month, beginning of year, start of year and end of year, next or last followed by a weekday name, and a count with a unit of days, weeks, months or years written as N units ago, N units from now, or in N units. Phrases are matched without regard to case. Anything else gives NULL for that row.

## Syntax

```sql
parse_natural_date(text, reference_date)
```

## Returns

DATE as days since 1970-01-01. NULL when either argument is NULL or the phrase is not recognised.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `reference_date` | Date the phrase is relative to, as days since 1970-01-01. | Not applicable. |

## Examples

```sql
SELECT parse_natural_date('tomorrow', 0)
```

1, the day after the reference date.

## Refused

- The phrase argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [parse_natural_duration](parse-natural-duration.md)
- [add_business_days](add-business-days.md)
