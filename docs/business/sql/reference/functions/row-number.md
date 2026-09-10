# row_number

Numbers the rows of each partition from 1 in the window's order. Two rows that tie on the ordering still get different numbers, so the result depends on how the tie is broken. Called without OVER it raises an error rather than returning a value.

## Syntax

```sql
row_number() OVER (...)
```

## Returns

BIGINT from 1. Never NULL.

## Examples

```sql
SELECT row_number() OVER (PARTITION BY team ORDER BY score DESC) FROM zyron_test.players
```

1 for the top scorer in each team, then 2, and so on.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [rank](rank.md)
- [dense_rank](dense-rank.md)
- [ntile](ntile.md)
