# DROP VIEW

Removes a view's definition. The tables it read are untouched, because a view never held rows of its own.

## Syntax

```sql
DROP VIEW [IF EXISTS] name [CASCADE | RESTRICT]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `IF EXISTS` | Does nothing when no view of that name is there. | A missing view fails the statement. |

## Examples

```sql
DROP VIEW paid
```

The view is gone and its tables are unchanged.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE VIEW](create-view.md)
