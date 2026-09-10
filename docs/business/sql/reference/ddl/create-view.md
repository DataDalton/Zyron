# CREATE VIEW

Names a query. Reading the view runs the query, so a view holds no rows of its own and is always as current as the tables under it. The stored definition is qualified when it is written, so a later change to the session's search path cannot make it read different tables.

## Syntax

```sql
CREATE [OR REPLACE] VIEW [IF NOT EXISTS] name [(col, ...)] AS SELECT ...
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `OR REPLACE` | Replaces an existing view's query, keeping its name and its grants. | A name already taken fails the statement. |
| `(col, ...)` | Names the view's columns, rather than taking the names the query produced. | The columns take the query's own output names. |

## Examples

```sql
CREATE VIEW paid AS SELECT id FROM orders WHERE total > 0
```

A name that reads as the query's result whenever it is selected from.

## Refused

- The query reads a temporary table.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP VIEW](drop-view.md)
- [CREATE MATERIALIZED VIEW](create-materialized-view.md)
