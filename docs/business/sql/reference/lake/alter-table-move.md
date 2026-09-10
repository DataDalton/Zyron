# ALTER TABLE MOVE

Moves rows to another storage tier. The rows remain readable. Read latency and cost follow the tier they were moved to.

## Syntax

```sql
ALTER TABLE name MOVE PARTITION 'spec' | WHERE predicate TO TIER 'tier'
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `PARTITION 'spec'` | Moves one named partition. | Not applicable. |
| `WHERE predicate` | Moves the rows the predicate selects. | Not applicable. |

## Examples

```sql
ALTER TABLE logs MOVE WHERE id < 100 TO TIER 'cold'
```

Those rows are held on the named tier and stay readable.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER TABLE](../ddl/alter-table.md)
