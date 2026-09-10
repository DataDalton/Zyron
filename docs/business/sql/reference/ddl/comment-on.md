# COMMENT ON

Records a description against an object, which the catalog views report and tooling shows. Writing NULL instead of text removes the description.

## Syntax

```sql
COMMENT ON TABLE | COLUMN | INDEX | VIEW | SCHEMA name IS 'text' | NULL
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `IS NULL` | Removes the description rather than setting one. | Not applicable. |

## Examples

```sql
COMMENT ON TABLE orders IS 'one row per placed order'
```

The description is recorded and reported by the catalog views.

## Refused

- The object is a temporary table or one of its columns.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE TABLE](create-table.md)
