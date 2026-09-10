# business_days_between

Counts every day from the start to the end that is neither a weekend nor a holiday, including both endpoints. A start later than the end returns a negative count of the same size. The cost grows with the span, because every day is tested.

## Syntax

```sql
business_days_between(start, end [, holidays])
```

## Returns

INTEGER, negative when the start is later than the end. NULL when either date is NULL or the holiday list does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `holidays` | JSON array of dates as days since 1970-01-01. | No holidays, so only weekends are excluded. |

## Examples

```sql
SELECT business_days_between(0, 4)
```

3, counting Thursday, Friday and Monday.

## Refused

- A date argument is not an integer or date column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [add_business_days](add-business-days.md)
- [is_business_day](is-business-day.md)
