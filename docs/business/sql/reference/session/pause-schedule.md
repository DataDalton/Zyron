# PAUSE SCHEDULE

Stops a schedule firing without removing it. Firings missed while it is paused are not made up when it resumes, because a schedule says when to run rather than how many times.

## Syntax

```sql
PAUSE SCHEDULE name
```

## Examples

```sql
PAUSE SCHEDULE auto_vacuum
```

The schedule stops firing and keeps its definition.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [RESUME SCHEDULE](resume-schedule.md)
- [CREATE SCHEDULE](create-schedule.md)
