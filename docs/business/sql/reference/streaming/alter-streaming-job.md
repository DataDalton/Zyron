# ALTER STREAMING JOB

Pauses a job without losing its position, resumes one that was paused, or changes its options. A paused job stops reading and holds where it had reached, so resuming continues rather than restarting.

## Syntax

```sql
ALTER STREAMING JOB name PAUSE | RESUME | SET (option = value, ...)
```

## Examples

```sql
ALTER STREAMING JOB sj PAUSE
```

The job stops reading and holds its position.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE STREAMING JOB](create-streaming-job.md)
