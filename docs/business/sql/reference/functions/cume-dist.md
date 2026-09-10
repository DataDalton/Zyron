# cume_dist

Returns the count of rows ordering at or below this one over the partition size, so the last row is always 1.0 and no row is 0.0. percent_rank differs in starting at 0.0 for the first row.

## Syntax

```sql
cume_dist() OVER (...)
```

## Returns

DOUBLE PRECISION above 0.0 and at most 1.0. Never NULL.

## Examples

```sql
SELECT cume_dist() OVER (ORDER BY score) FROM zyron_test.players
```

1.0 for the highest score.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [percent_rank](percent-rank.md)
- [ntile](ntile.md)
