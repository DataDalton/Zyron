# UNDROP TABLE

Restores a dropped table with the rows it held. Succeeds while the dropped table's versions remain within the retention window. Fails once retention has removed them.

## Syntax

```sql
UNDROP TABLE name
```

## Examples

```sql
UNDROP TABLE orders
```

The table is readable again, with the rows it held when it was dropped.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP TABLE](drop-table.md)
- [RESTORE TABLE](../lake/restore-table.md)
