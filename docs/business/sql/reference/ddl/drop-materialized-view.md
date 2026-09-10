# DROP MATERIALIZED VIEW

Removes the view's definition and the rows it had stored. The tables its query read are untouched.

## Syntax

```sql
DROP MATERIALIZED VIEW [IF EXISTS] name
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `IF EXISTS` | Does nothing when no view of that name is there. | A missing view fails the statement. |

## Examples

```sql
DROP MATERIALIZED VIEW totals
```

The view and its stored rows are gone.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE MATERIALIZED VIEW](create-materialized-view.md)
