# is_business_day

False on Saturday and Sunday and on any date the holiday list names, true otherwise. The weekend is fixed at Saturday and Sunday, so a calendar resting on other days is expressed by listing those days as holidays.

## Syntax

```sql
is_business_day(date [, holidays])
```

## Returns

BOOLEAN. NULL when the date is NULL or the holiday list does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `holidays` | JSON array of dates as days since 1970-01-01. | No holidays, so only weekends are excluded. |

## Examples

```sql
SELECT is_business_day(0)
```

true, because 1970-01-01 was a Thursday.

## Refused

- The date argument is not an integer or date column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [next_business_day](next-business-day.md)
- [add_business_days](add-business-days.md)
- [business_days_between](business-days-between.md)
