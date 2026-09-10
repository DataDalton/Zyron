# CREATE SCHEDULE

Runs a statement repeatedly without anything outside the database driving it. On a cluster the schedule runs on one member rather than on all of them, so a scheduled statement happens once per firing rather than once per node.

## Syntax

```sql
CREATE SCHEDULE name EVERY n MINUTES | HOURS | DAYS | CRON 'expr' DO statement
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `CRON 'expr'` | Fires on a cron expression rather than on a fixed interval. | The schedule fires on the interval written after EVERY. |

## Examples

```sql
CREATE SCHEDULE auto_vacuum EVERY 5 MINUTES DO VACUUM
```

That statement runs every five minutes, on one member.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP SCHEDULE](drop-schedule.md)
- [PAUSE SCHEDULE](pause-schedule.md)
- [RESUME SCHEDULE](resume-schedule.md)
