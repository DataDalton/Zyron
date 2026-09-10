# CREATE TABLE

Creates a table. The columns are written out with their types and constraints, or taken from a query's output when AS SELECT is written, in which case the query also fills the table in the same statement. Writing USING ZYRONLAKE makes it a lake table, which stores columns rather than rows and accepts the branching and time-travel operations a heap table does not.

## Syntax

```sql
CREATE [OR REPLACE] [TEMPORARY | TEMP] TABLE [IF NOT EXISTS] name (col type [constraint ...], ...) [USING ZYRONLAKE] [CLUSTER BY (...)] [AS SELECT ...]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `IF NOT EXISTS` | Does nothing when a table of that name is already there, rather than failing. | A name already taken fails the statement. |
| `OR REPLACE` | Drops a table of that name and creates the new one in its place. | A name already taken fails the statement. |
| `TEMPORARY | TEMP` | Makes the table belong to this session alone, on this node alone. | The table is permanent and reaches every member of the group. |
| `AS SELECT ...` | Takes the columns from the query's output and fills the table in one statement. | The columns are written out and the table starts empty. |
| `ON COMMIT ...` | On a temporary table, what a commit does to its rows. | The rows outlive the transaction. |

## Examples

```sql
CREATE TABLE orders (id INT, total INT)
```

An empty table of two columns.

## Refused

- A temporary table is given a schema-qualified name.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE TEMPORARY TABLE](../session/create-temporary-table.md)
- [DROP TABLE](drop-table.md)
- [ALTER TABLE](alter-table.md)
