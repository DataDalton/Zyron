# tdigest_add

Inserts the value as a centroid of weight one in mean order, then merges neighbouring centroids back inside the compression budget. The digest grows until that budget binds and stays bounded after it, so its size depends on the compression rather than on how many values were added.

## Syntax

```sql
tdigest_add(tdigest, value)
```

## Returns

TDIGEST. NULL when the digest or the value is NULL, or the digest bytes do not parse.

## Examples

```sql
SELECT tdigest_quantile(tdigest_add(tdigest_add(tdigest_create(100), 10), 20), 0.5)
```

20, the mean of the centroid holding the weight at the median.

## Refused

- The digest argument is not text or binary.
- Called with any count of arguments other than two.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [tdigest_create](tdigest-create.md)
- [tdigest_quantile](tdigest-quantile.md)
- [tdigest_merge](tdigest-merge.md)
