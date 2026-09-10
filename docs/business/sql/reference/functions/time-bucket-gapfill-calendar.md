# time_bucket_gapfill_calendar

Lists the bucket starts from the bucket holding the start instant up to the end instant, so a report can show a row for a bucket the data has nothing in. Month and year intervals step by calendar boundaries rather than by a fixed number of microseconds.

## Syntax

```sql
time_bucket_gapfill_calendar(interval, start, end)
```

## Returns

ARRAY of epoch microseconds as JSON text. NULL when any argument is NULL.

## Examples

```sql
SELECT time_bucket_gapfill_calendar(INTERVAL '1 month', 0, 5184000000000)
```

The starts of the months the range covers.

## Refused

- A range argument is not a timestamp.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [time_bucket_calendar](time-bucket-calendar.md)
