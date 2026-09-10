# ALTER INDEX

Renames an index or changes the options it was built with. The tree itself is not rebuilt, so this costs a catalog write rather than a pass over the table.

## Syntax

```sql
ALTER INDEX name RENAME TO name | SET (option = value, ...)
```

## Examples

```sql
ALTER INDEX orders_id RENAME TO orders_pk
```

The index answers to a new name.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE INDEX](create-index.md)
- [REINDEX](reindex.md)
