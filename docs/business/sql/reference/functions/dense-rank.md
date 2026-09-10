# dense_rank

Gives tied rows the same rank and continues with the next number, so two rows at rank 1 are followed by rank 2. The highest rank equals the number of distinct ordering values.

## Syntax

```sql
dense_rank() OVER (...)
```

## Returns

BIGINT from 1. Never NULL.

## Examples

```sql
SELECT dense_rank() OVER (ORDER BY score DESC) FROM zyron_test.players
```

1, 1, 2 where the top two scores tie.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [rank](rank.md)
- [row_number](row-number.md)
