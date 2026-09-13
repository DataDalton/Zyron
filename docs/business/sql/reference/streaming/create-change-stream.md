# CREATE CHANGE STREAM

Creates a change stream, a durable position over the changes of a table, of several tables, or of a view over one table. A read of the stream in a FROM clause yields the changes past its position, with the same columns table_changes yields, and the position moves when the reading transaction commits. A read that rolls back leaves the position where it was. Two transactions reading one stream wait on each other, so no change is handed out twice. A stream over several tables yields the union of their columns by name, an absent column reading NULL, plus _source_table naming the table each change came from, and every read ends at one boundary across the tables so a transaction that wrote to two of them is wholly inside the read or wholly after it. Written after the stream in FROM, WITH (peek => true) reads without moving the position, WITH (schema => 'as_of_change') renders each change through the columns it was written under, and WITH (max_rows => n) ends the read after about n changes at a boundary no transaction writes across.

## Syntax

```sql
CREATE CHANGE STREAM [IF NOT EXISTS] name ON TABLE table | ON TABLES (table, ...) | ON VIEW view [AT VERSION n | AT TIMESTAMP 'ts' | SHOW INITIAL ROWS] [APPEND_ONLY] [WHERE predicate] [COLUMNS (col, ...)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ON TABLE table` | The stream yields that table's changes. | Not applicable. |
| `ON TABLES (table, ...)` | The stream yields the changes of every table named, each row naming its table in _source_table, with one position moved for all of them. | Not applicable. |
| `ON VIEW view` | The stream yields the changes of the view's one base table, projected through the view's columns and narrowed by its predicate. | Not applicable. |
| `AT VERSION n | AT TIMESTAMP 'ts'` | Where the position starts, so the first read yields the changes past that version or instant. | The source's current version, so the first read yields nothing until the next change. |
| `SHOW INITIAL ROWS` | The first read yields every existing row as an insert, then the position continues from the feed, so one definition seeds a target and keeps it current. | Not applicable. |
| `APPEND_ONLY` | The stream yields inserts alone, and a purge of updates and deletes beneath its position does not make it stale. | Every change is yielded. |
| `WHERE predicate` | Narrows the changes yielded. The predicate reads the row after an insert or an update and the row before a delete, and reads the source's columns whether or not COLUMNS exposes them. | Every change is yielded. |
| `COLUMNS (col, ...)` | The stream yields these columns and the metadata columns alone. | The source's own columns are yielded. |

## Examples

```sql
CREATE CHANGE STREAM order_changes ON TABLE orders
```

A stream positioned at the table's current version, whose reads yield the changes from here on.

```sql
CREATE CHANGE STREAM eu_orders ON TABLE orders SHOW INITIAL ROWS WHERE region = 'eu' COLUMNS (id, total)
```

A stream whose first read yields the existing eu rows as inserts and later reads yield the eu changes, each with id, total and the metadata columns.

## Refused

- The table has no change data feed.
- Two of the tables hold a column of one name at different types.
- The table already carries as many change streams as the node's change_streams_per_table setting allows.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER CHANGE STREAM](alter-change-stream.md)
- [DROP CHANGE STREAM](drop-change-stream.md)
- [SHOW CHANGE STREAMS](show-change-streams.md)
- [APPLY CHANGES](apply-changes.md)
- [table_changes](table-changes.md)
- [GRANT](../ddl/grant.md)
