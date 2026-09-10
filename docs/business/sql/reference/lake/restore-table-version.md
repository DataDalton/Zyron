# RESTORE TABLE VERSION

Returns a table to the contents it held at a version or at a timestamp. Writes made since are no longer current. Requires that the versions it needs are still retained.

## Syntax

```sql
RESTORE TABLE name TO VERSION n | TO TIMESTAMP 'text'
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `TO TIMESTAMP 'text'` | Restores to the version current at that moment, rather than to a numbered one. | Not applicable. |

## Examples

```sql
RESTORE TABLE orders TO VERSION 1
```

The table holds what it held at that version.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE VERSION](create-version.md)
- [RESTORE TABLE](restore-table.md)
- [UNDROP TABLE](../ddl/undrop-table.md)
