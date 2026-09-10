# DROP TABLE

Removes a table, its rows, its indexes and its grants. On a table that keeps versions the drop is recoverable by UNDROP TABLE until the retention window passes. CASCADE removes the objects that depend on it rather than refusing.

## Syntax

```sql
DROP TABLE [IF EXISTS] name [CASCADE | RESTRICT]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `IF EXISTS` | Does nothing when no table of that name is there, rather than failing. | A missing table fails the statement. |
| `CASCADE` | Removes the views and constraints that depend on the table too. | A dependent object refuses the drop. |

## Examples

```sql
DROP TABLE IF EXISTS staging
```

The table is gone, and nothing happens if it was not there.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE TABLE](create-table.md)
- [UNDROP TABLE](undrop-table.md)
- [TRUNCATE](../dml/truncate.md)
