# RUN RETENTION JOB

Runs retention now instead of waiting for its schedule, acting on the rows whose TTL has passed. DRY RUN reports what would be acted on without touching a row, which is how a TTL is checked before it removes anything.

## Syntax

```sql
RUN RETENTION JOB [FOR TABLE name] [DRY RUN]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `DRY RUN` | Reports what would be acted on without acting. | The expired rows are acted on. |

## Examples

```sql
RUN RETENTION JOB
```

Rows whose TTL has passed are acted on now.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [ALTER TABLE SET TTL](alter-table-set-ttl.md)
- [ARCHIVE TABLE](archive-table.md)
