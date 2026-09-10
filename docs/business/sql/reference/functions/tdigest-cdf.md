# tdigest_cdf

Accumulates centroid weight while the centroid mean stays below the value and divides by the total weight, which inverts tdigest_quantile. A value above every centroid gives 1.0. A digest with no centroids gives 0.0.

## Syntax

```sql
tdigest_cdf(digest, value)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when the digest or the value is NULL, or the digest bytes do not parse.

## Examples

```sql
SELECT tdigest_cdf(tdigest_add(tdigest_add(tdigest_create(100), 1), tdigest_add(tdigest_create(100), 9)), 5)
```

0.5.

## Refused

- The digest argument is not text or binary.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [tdigest_quantile](tdigest-quantile.md)
- [tdigest_add](tdigest-add.md)
