# day_of_week

Returns 0 for Sunday through 6 for Saturday. The date is days since 1970-01-01, which was a Thursday. Dates before the epoch count backward and still land on the right weekday.

## Syntax

```sql
day_of_week(date)
```

## Returns

INTEGER between 0 and 6. NULL when the date is NULL.

## Examples

```sql
SELECT day_of_week(0)
```

4, because 1970-01-01 was a Thursday.

## Refused

- The argument is not an integer or date column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [is_business_day](is-business-day.md)
- [next_business_day](next-business-day.md)
