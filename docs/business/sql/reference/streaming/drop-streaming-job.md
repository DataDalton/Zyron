# DROP STREAMING JOB

Stops a job and discards its position. Rows it already wrote to the target remain. A new job over the same query starts from the beginning.

## Syntax

```sql
DROP STREAMING JOB [IF EXISTS] name
```

## Examples

```sql
DROP STREAMING JOB sj
```

The job stops and its position is forgotten.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE STREAMING JOB](create-streaming-job.md)
