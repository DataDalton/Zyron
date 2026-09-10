# RESTORE SOFT DELETE

Clears the soft-delete mark on rows, so they read as live again. Succeeds while the marked rows remain, which is until a retention run removes them.

## Syntax

```sql
RESTORE FROM table [WHERE predicate]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WHERE predicate` | Brings back only the marked rows the predicate holds for. | Every marked row is brought back. |

## Examples

```sql
RESTORE FROM orders
```

Rows a soft delete had marked read as live again.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [DELETE](../dml/delete.md)
- [RUN RETENTION JOB](run-retention-job.md)
