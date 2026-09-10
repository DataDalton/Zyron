# time_bucket_calendar

Aligns to a calendar boundary for a month-based interval, so a monthly bucket starts on the first of the month whatever the epoch offset would give. An interval shorter than a day is bucketed by plain division, matching time_bucket. Use this over time_bucket wherever months or years are involved, because those have no fixed length.

## Syntax

```sql
time_bucket_calendar(interval, timestamp)
```

## Returns

TIMESTAMPTZ as epoch microseconds. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `interval` | Bucket width as an interval. | Not applicable. |

## Examples

```sql
SELECT time_bucket_calendar(INTERVAL '1 month', 0)
```

0, the start of the month the epoch falls in.

## Refused

- The second argument is not a timestamp.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [time_bucket_gapfill_calendar](time-bucket-gapfill-calendar.md)
- [fiscal_quarter](fiscal-quarter.md)
