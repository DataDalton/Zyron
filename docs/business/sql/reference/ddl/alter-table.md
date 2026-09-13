# ALTER TABLE

Changes a table's definition. A column added or dropped and most type changes complete without touching a row, because the tuple carries the schema epoch it was written under and a reader resolves it against the current definition. A type change the stored bytes cannot satisfy runs a rewrite instead, which the progress view reports while it runs.

## Syntax

```sql
ALTER TABLE name ADD COLUMN ... | DROP COLUMN ... | ALTER COLUMN ... | RENAME ... | ADD CONSTRAINT ... | DROP CONSTRAINT ...
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ADD COLUMN name type [constraint ...]` | Adds a column. Existing rows read it as its default without being rewritten. | Not applicable. |
| `DROP COLUMN name` | Removes a column. The bytes stay in the stored rows until they are next written. | Not applicable. |
| `ALTER COLUMN name TYPE type` | Changes a column's type, rewriting the table only when the stored bytes cannot be read as the new type. | Not applicable. |
| `ALTER COLUMN name TYPE type ACKNOWLEDGE STREAM BREAK` | Narrows a column's type on a table with change streams. Each stream over the table is marked as needing attention, keeps its position, and yields again once its definition is corrected. | A narrowing on a table with change streams is refused. |
| `ADD CONSTRAINT name ...` | Adds a constraint, which is checked against the rows already there before it takes effect. | Not applicable. |

## Examples

```sql
ALTER TABLE orders ADD COLUMN note TEXT
```

The table has a new column and no row was rewritten.

```sql
ALTER TABLE orders ALTER COLUMN total TYPE INT ACKNOWLEDGE STREAM BREAK
```

The column narrows, and every change stream over the table needs attention until its column list is corrected.

## Refused

- A column's type is narrowed on a table with change streams and the break is not acknowledged.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE TABLE](create-table.md)
- [ALTER TABLE SET TTL](../lake/alter-table-set-ttl.md)
- [ALTER CHANGE STREAM](../streaming/alter-change-stream.md)
