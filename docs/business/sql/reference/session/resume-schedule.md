# RESUME SCHEDULE

Lets a paused schedule fire again, from the next time it is due. Nothing missed is replayed.

## Syntax

```sql
RESUME SCHEDULE name
```

## Examples

```sql
RESUME SCHEDULE auto_vacuum
```

The schedule fires again from its next due time.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [PAUSE SCHEDULE](pause-schedule.md)
