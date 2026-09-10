# VALUES

Yields rows written out in the statement, as a relation. Every row must have the same number of expressions, and a column takes the type the rows agree on. It stands as a statement of its own and as a FROM item, which is how a small fixed table is written inline rather than created.

## Syntax

```sql
VALUES (expr, ...) [, (expr, ...) ...]
```

## Examples

```sql
VALUES (1, 'a'), (2, 'b')
```

Two rows of two columns.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [SELECT](select.md)
- [INSERT](insert.md)
