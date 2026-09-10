# LATERAL

Says that a FROM item reads the rows of the relations written before it. Without it, every item in a FROM clause is independent and a reference to an earlier relation's column has nowhere to resolve. With it, the item runs once per row of what came before, with that row's columns in scope. A row-generating item that produces nothing for a row drops that row, so keeping the row means writing the join out as a LEFT JOIN LATERAL with ON TRUE.

## Syntax

```sql
LATERAL (subquery) | LATERAL UNNEST(...) | LATERAL FLATTEN(...)
```

## Examples

```sql
SELECT o.id, u.item FROM orders AS o LEFT JOIN LATERAL UNNEST(o.items) AS u (item) ON TRUE
```

One row per item, and one null-item row for an order whose array is empty.

## Refused

- A FROM item reads an earlier relation's column and LATERAL was not written.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [UNNEST](unnest.md)
- [FLATTEN](flatten.md)
