# next_business_day

Starts at the day after the given date and steps forward until it reaches a day that is neither a weekend nor a holiday. A date that is itself a business day is never returned, so the result is always later than the argument.

## Syntax

```sql
next_business_day(date [, holidays])
```

## Returns

DATE as days since 1970-01-01. NULL when the date is NULL or the holiday list does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `holidays` | JSON array of dates as days since 1970-01-01. | No holidays, so only weekends are skipped. |

## Examples

```sql
SELECT next_business_day(0)
```

1, the Friday after the Thursday epoch.

## Refused

- The date argument is not an integer or date column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [is_business_day](is-business-day.md)
- [add_business_days](add-business-days.md)
