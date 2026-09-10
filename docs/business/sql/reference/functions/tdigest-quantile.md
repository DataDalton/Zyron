# tdigest_quantile

Walks the centroids in mean order until the cumulative weight reaches the requested fraction of the total, and returns that centroid's mean. The quantile is clamped to between 0.0 and 1.0. A digest with no centroids gives NULL.

## Syntax

```sql
tdigest_quantile(digest, q)
```

## Returns

DOUBLE PRECISION. NULL when the digest or the quantile is NULL, the digest holds no centroids, or its bytes do not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `q` | Quantile as a fraction, where 0.5 is the median and 0.99 the 99th percentile. | Not applicable. |

## Examples

```sql
SELECT tdigest_quantile(tdigest_add(tdigest_create(100), 42), 0.99)
```

42.

## Refused

- The digest argument is not text or binary.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [tdigest_cdf](tdigest-cdf.md)
- [tdigest_add](tdigest-add.md)
- [percentile](percentile.md)
