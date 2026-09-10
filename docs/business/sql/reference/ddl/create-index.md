# CREATE INDEX

Builds a B-tree index so a lookup on the indexed columns reads a tree rather than the whole table. The build does not block writes: the index is published first, writers maintain it while a scan fills it from the rows already there, and it is flipped to usable once the scan completes. A partial index carries a predicate and holds only the rows it selects.

## Syntax

```sql
CREATE [UNIQUE] INDEX [IF NOT EXISTS] name ON table (col [ASC | DESC], ...) [WHERE predicate]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `UNIQUE` | Refuses two live rows sharing a key, which the build checks against the rows already there. | Duplicate keys are allowed. |
| `WHERE predicate` | Indexes only the rows the predicate holds for, which keeps the tree small. | Every row is indexed. |
| `col DESC` | Orders that column descending in the tree. | The column is ordered ascending. |

## Examples

```sql
CREATE UNIQUE INDEX orders_id ON orders (id)
```

An index that also refuses two rows sharing an id.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP INDEX](drop-index.md)
- [ALTER INDEX](alter-index.md)
- [REINDEX](reindex.md)
