# add_business_days

Steps one day at a time, counting down only on days that are neither a weekend nor a holiday. A negative count steps backward. A count of zero returns the date unchanged even when it is a weekend, so the result is only guaranteed to be a business day for a non-zero count.

## Syntax

```sql
add_business_days(date, n [, holidays])
```

## Returns

DATE as days since 1970-01-01. NULL when the date or the count is NULL, or the holiday list does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `n` | Working days to add. Negative counts move backward. | Not applicable. |
| `holidays` | JSON array of dates as days since 1970-01-01. | No holidays, so only weekends are skipped. |

## Examples

```sql
SELECT add_business_days(0, 1)
```

1, the Friday after the Thursday epoch.

## Refused

- The date or count argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [business_days_between](business-days-between.md)
- [next_business_day](next-business-day.md)
- [is_business_day](is-business-day.md)
