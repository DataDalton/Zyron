# CREATE MATERIALIZED VIEW

Runs a query and stores its result, so reading it reads stored rows rather than running the query again. What it holds is as current as its last refresh, which is the trade it makes against a plain view: reads are cheap and the answer can be behind.

## Syntax

```sql
CREATE MATERIALIZED VIEW [IF NOT EXISTS] name AS SELECT ...
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `IF NOT EXISTS` | Does nothing when a view of that name is already there. | A name already taken fails the statement. |

## Examples

```sql
CREATE MATERIALIZED VIEW totals AS SELECT region, SUM(amount) FROM sales GROUP BY region
```

A stored result that is read directly and refreshed on demand.

## Refused

- The query reads a temporary table.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [REFRESH MATERIALIZED VIEW](refresh-materialized-view.md)
- [DROP MATERIALIZED VIEW](drop-materialized-view.md)
- [CREATE VIEW](create-view.md)
