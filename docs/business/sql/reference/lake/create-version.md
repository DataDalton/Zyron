# CREATE VERSION

Names a table version so a query can address it by name rather than by number. The name is immutable and continues to resolve to the same version regardless of later writes.

## Syntax

```sql
CREATE VERSION name ON table [AT VERSION n]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `AT VERSION n` | Names that version rather than the table's current one. | The table's current version is named. |

## Examples

```sql
CREATE VERSION v1 ON orders
```

A name pointing at the table's current version, permanently.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP VERSION](drop-version.md)
- [RESTORE TABLE VERSION](restore-table-version.md)
