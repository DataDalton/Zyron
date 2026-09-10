# percentile

Sorts the values and reads the position p of the way along them, interpolating between the two neighbours when the position falls between them. The fraction is clamped to between 0.0 and 1.0. A single value is returned as it stands, and an empty array gives 0.0. The whole series is sorted on every call, so a T-Digest is the cheaper form over a large or growing series.

## Syntax

```sql
percentile(values, p)
```

## Returns

DOUBLE PRECISION. NULL when the array or the fraction is NULL, or the array holds a non-number.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `p` | Position as a fraction, where 0.5 is the median and 0.95 the 95th percentile. | Not applicable. |

## Examples

```sql
SELECT percentile('[1,2,3,4]', 0.5)
```

2.5, interpolated between 2 and 3.

## Refused

- Called with any count of arguments other than two.
- The values argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [tdigest_quantile](tdigest-quantile.md)
- [outlier_detect_iqr](outlier-detect-iqr.md)
