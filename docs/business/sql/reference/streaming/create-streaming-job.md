# CREATE STREAMING JOB

Runs a query continuously, writing its result into the target as rows arrive at the sources it reads. The write mode determines whether result rows are appended or applied by key. The run mode determines whether the job watches for new rows or runs on a schedule. A watermark bounds how late a row may arrive and still be counted, without which a window over event time never closes.

## Syntax

```sql
CREATE STREAMING JOB [IF NOT EXISTS] name AS SELECT ... INTO target [WRITE MODE APPEND | UPSERT] [MODE WATCH | SCHEDULED EVERY 'interval' | SCHEDULED CRON 'expr'] [WATERMARK FOR col AS expr] [WITH (...)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WRITE MODE UPSERT` | Applies each result row by key, replacing the row already there. | Rows are appended. |
| `MODE WATCH` | Runs as rows arrive rather than on a schedule. | Not applicable. |
| `MODE SCHEDULED EVERY 'interval'` | Runs on that interval instead of watching. | Not applicable. |
| `WATERMARK FOR col AS expr` | Bounds how late a row may arrive and still be counted. A window over event time closes once the watermark passes its end. | A window over event time never closes on lateness alone. |

## Examples

```sql
CREATE STREAMING JOB sj AS SELECT id, v FROM src INTO dst
```

A job that keeps the target current as rows arrive at the source.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER STREAMING JOB](alter-streaming-job.md)
- [DROP STREAMING JOB](drop-streaming-job.md)
