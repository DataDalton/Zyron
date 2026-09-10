# MERGE

Matches each source row against the target by a predicate and applies the clause for the branch it falls in. A matched row may be updated or deleted. An unmatched row may be inserted. All branches apply in one statement, so the target is never observable part way through.

## Syntax

```sql
MERGE INTO target USING source ON predicate WHEN MATCHED [AND ...] THEN UPDATE SET ... | DELETE WHEN NOT MATCHED [AND ...] THEN INSERT ...
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WHEN MATCHED [AND predicate] THEN UPDATE SET ...` | Changes a target row that the source matched. | Not applicable. |
| `WHEN MATCHED [AND predicate] THEN DELETE` | Removes a target row that the source matched. | Not applicable. |
| `WHEN NOT MATCHED [AND predicate] THEN INSERT ...` | Adds a row for a source row the target did not have. | Not applicable. |

## Examples

```sql
MERGE INTO orders USING staging ON orders.id = staging.id WHEN MATCHED THEN UPDATE SET total = staging.total WHEN NOT MATCHED THEN INSERT (id, total) VALUES (staging.id, staging.total)
```

Rows already present are updated and new rows are added, in one pass.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [INSERT](insert.md)
- [UPDATE](update.md)
- [DELETE](delete.md)
