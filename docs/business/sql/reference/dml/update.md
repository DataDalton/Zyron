# UPDATE

Assigns new values to columns of the rows a predicate selects. Without a predicate, every row in the table is changed. The row count is unaffected: UPDATE never adds or removes rows.

## Syntax

```sql
UPDATE name SET col = expr [, ...] [WHERE predicate] [RETURNING ...]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WHERE predicate` | Changes only the rows the predicate holds for. | Every row in the table is changed. |
| `RETURNING items` | Returns the rows as they are after the change. | Only a count is returned. |

## Examples

```sql
UPDATE orders SET total = 0 WHERE id = 1
```

The one row with that id has its total set to zero.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [INSERT](insert.md)
- [DELETE](delete.md)
- [MERGE](merge.md)
