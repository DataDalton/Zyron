# DROP INDEX

Removes an index and the file behind it. The table and its rows are untouched, and a query that was using the index falls back to reading the table.

## Syntax

```sql
DROP INDEX [IF EXISTS] name
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `IF EXISTS` | Does nothing when no index of that name is there. | A missing index fails the statement. |

## Examples

```sql
DROP INDEX orders_id
```

The index is gone and the table is unchanged.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE INDEX](create-index.md)
