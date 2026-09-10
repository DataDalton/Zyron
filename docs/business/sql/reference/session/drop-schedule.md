# DROP SCHEDULE

Removes a schedule and the statement it ran. A firing already in progress finishes.

## Syntax

```sql
DROP SCHEDULE name
```

## Examples

```sql
DROP SCHEDULE auto_vacuum
```

The schedule is gone and a firing in progress finishes.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE SCHEDULE](create-schedule.md)
