# rank

Gives tied rows the same rank and then skips as many ranks as there were ties, so two rows at rank 1 are followed by rank 3. Use dense_rank where the numbering must stay consecutive.

## Syntax

```sql
rank() OVER (...)
```

## Returns

BIGINT from 1. Never NULL.

## Examples

```sql
SELECT rank() OVER (ORDER BY score DESC) FROM zyron_test.players
```

1, 1, 3 where the top two scores tie.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [dense_rank](dense-rank.md)
- [row_number](row-number.md)
- [percent_rank](percent-rank.md)
