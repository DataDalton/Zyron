# INSERT

Adds rows to a table. The rows come from values written out or from a query, and a column list says which columns they fill, leaving the rest to their defaults. ON CONFLICT says what to do when a row collides with a unique constraint, and RETURNING hands back the rows as they were stored, which is how a generated key is read.

## Syntax

```sql
INSERT INTO name [(col, ...)] VALUES (...) | SELECT ... [ON CONFLICT ...] [RETURNING ...]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `(col, ...)` | Names the columns the values fill, in the order they are written. | The values fill every column, in the table's own order. |
| `ON CONFLICT ... DO NOTHING | DO UPDATE SET ...` | Says what happens to a row that collides with a unique constraint, rather than failing. | A collision fails the statement. |
| `RETURNING items` | Returns the rows as stored, with defaults and generated values filled in. | Only a count is returned. |

## Examples

```sql
INSERT INTO orders (id, total) VALUES (1, 100) RETURNING id
```

One row added, and its id handed back.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [MERGE](merge.md)
- [UPDATE](update.md)
- [COPY](copy.md)
