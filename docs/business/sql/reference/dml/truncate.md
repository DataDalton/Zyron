# TRUNCATE

Empties a table without reading its rows, so the cost does not scale with what the table holds. The table's definition, indexes and grants are unaffected. Unlike DELETE, it emits no per-row changes.

## Syntax

```sql
TRUNCATE TABLE name
```

## Examples

```sql
TRUNCATE TABLE staging
```

The table holds no rows and its definition is unchanged.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DELETE](delete.md)
