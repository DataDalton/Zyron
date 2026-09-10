# ANALYZE

Samples a table and records what it found: how many rows, how they are distributed, how wide a column's values are. The planner estimates with these, so a table whose statistics are stale gets plans chosen for the shape it used to have rather than the shape it has.

## Syntax

```sql
ANALYZE [table]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `table` | Samples that table alone. | Every table is sampled. |

## Examples

```sql
ANALYZE orders
```

The planner's statistics for that table are current.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [EXPLAIN](explain.md)
- [VACUUM](vacuum.md)
