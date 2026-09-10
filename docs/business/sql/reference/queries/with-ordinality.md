# WITH ORDINALITY

Appends a BIGINT column holding each row's position within its array. Numbering starts at 1 for each input row and does not run across the whole result. The column is appended after the element columns.

## Syntax

```sql
UNNEST(...) WITH ORDINALITY
```

## Examples

```sql
SELECT * FROM UNNEST(ARRAY['a', 'b']) WITH ORDINALITY AS t (letter, at)
```

Two rows: ('a', 1) and ('b', 2).

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [UNNEST](unnest.md)
