# tdigest_create

Returns a nine-byte header with no centroids. Compression bounds how many centroids survive a merge, so a higher setting holds more of them, estimates quantiles closer to exactly, and takes more space. Accuracy is highest at the tails and lowest near the median, which is the opposite of an equal-width histogram.

## Syntax

```sql
tdigest_create(compression)
```

## Returns

TDIGEST. NULL when the compression is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `compression` | Centroid budget, between 1 and 10000. Truncated to a whole number. | Not applicable. |

## Examples

```sql
SELECT tdigest_create(100)
```

An empty digest of 9 bytes.

## Refused

- The compression is below 1 or above 10000.
- Called with any count of arguments other than one.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [tdigest_add](tdigest-add.md)
- [tdigest_quantile](tdigest-quantile.md)
- [tdigest_cdf](tdigest-cdf.md)
- [tdigest_merge](tdigest-merge.md)
