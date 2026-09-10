# DELETE

Removes the rows a predicate selects, and every row when no predicate is written. On a table with soft delete enabled, rows are marked rather than erased and remain readable through the soft-delete select modes until a retention run removes them. HARD erases them immediately.

## Syntax

```sql
DELETE FROM name [WHERE predicate] [RETURNING ...] [HARD]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WHERE predicate` | Removes only the rows the predicate holds for. | Every row in the table is removed. |
| `HARD` | Erases the rows rather than marking them, on a table that soft-deletes. | A soft-deleting table marks the rows and keeps them readable. |
| `RETURNING items` | Returns the rows as they were before removal. | Only a count is returned. |

## Examples

```sql
DELETE FROM orders WHERE total = 0
```

Every row whose total is zero is removed.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [TRUNCATE](truncate.md)
- [UPDATE](update.md)
