# ntile

Splits each partition into the requested number of buckets in the window's order and numbers them from 1. A partition that does not divide evenly puts the extra rows in the earlier buckets, so bucket sizes differ by at most one.

## Syntax

```sql
ntile(buckets) OVER (...)
```

## Returns

BIGINT from 1 to the bucket count. Never NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `buckets` | How many buckets to split the partition into. | Not applicable. |

## Examples

```sql
SELECT ntile(4) OVER (ORDER BY score) FROM zyron_test.players
```

1 for the lowest quarter of scores and 4 for the highest.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [percent_rank](percent-rank.md)
- [cume_dist](cume-dist.md)
- [rank](rank.md)
