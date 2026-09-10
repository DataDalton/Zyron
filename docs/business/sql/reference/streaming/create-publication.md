# CREATE PUBLICATION

Names what a subscriber receives. A publication is a set of tables, and a table may be narrowed to some of its columns and to the rows a predicate selects, so one database can publish different shapes of the same table to different consumers.

## Syntax

```sql
CREATE PUBLICATION [IF NOT EXISTS] name FOR TABLE name [(col, ...)] [WHERE predicate] [, ...] [WITH (option = value, ...)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `(col, ...)` | Publishes those columns of the table alone. | Every column is published. |
| `WHERE predicate` | Publishes only the rows the predicate holds for. | Every row is published. |

## Examples

```sql
CREATE PUBLICATION pub FOR TABLE orders
```

A named set that a subscriber receives the table's changes through.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER PUBLICATION](alter-publication.md)
- [DROP PUBLICATION](drop-publication.md)
- [TAG PUBLICATION](tag-publication.md)
