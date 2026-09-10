# percent_rank

Returns the rank less one over the partition size less one, so the first row is always 0.0 and the last is 1.0. A partition holding one row gives 0.0. cume_dist differs in counting the rows at or below the current one instead.

## Syntax

```sql
percent_rank() OVER (...)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. Never NULL.

## Examples

```sql
SELECT percent_rank() OVER (ORDER BY score) FROM zyron_test.players
```

0.0 for the lowest score and 1.0 for the highest.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cume_dist](cume-dist.md)
- [rank](rank.md)
- [ntile](ntile.md)
