# UNNEST

Produces one row per array element, in array order. With several arrays, produces one column per array, zipped to the length of the longest, padding shorter arrays with NULL. Each column takes its array's element type, so a STRUCT[] produces STRUCT rows whose fields are addressable and a nested ARRAY[] produces arrays that can be unnested again. Written bare in FROM, the argument must be a literal or a subquery. Written under LATERAL, the argument may read columns of relations earlier in the FROM clause, and the item runs once per row of them.

## Syntax

```sql
UNNEST(array_expr [, array_expr ...]) [WITH ORDINALITY] [AS alias (col [, col ...])]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WITH ORDINALITY` | Appends a BIGINT column numbering the rows one array produced, starting at 1. | No position column is produced. |
| `AS alias (col [, col ...])` | Names the relation and its columns, which is how a query addresses them. | The columns take the names the arrays were written as. |

## Examples

```sql
SELECT * FROM UNNEST(ARRAY[10, 20, 30]) AS t (n)
```

Three rows, n = 10, 20 and 30.

```sql
SELECT o.id, u.item FROM orders AS o, LATERAL UNNEST(o.items) AS u (item)
```

One row per item of each order, carrying the order's id beside it.

## Refused

- The argument is not an array and does not declare an element type.
- It reads a column of a relation written earlier in FROM without LATERAL.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [WITH ORDINALITY](with-ordinality.md)
- [LATERAL](lateral.md)
- [FLATTEN](flatten.md)
