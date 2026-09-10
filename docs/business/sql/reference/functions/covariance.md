# covariance

Averages the product of each pair's deviations from their series means, over a denominator of one less than the pair count. The result carries the units of both inputs multiplied together, so comparing covariances across different pairs of series means little. Use correlation for a scale-free measure.

## Syntax

```sql
covariance(x, y)
```

## Returns

DOUBLE PRECISION. NULL when either array is NULL, holds a non-number, has a different length from the other, or holds fewer than two values.

## Examples

```sql
SELECT covariance('[1,2,3]', '[2,4,6]')
```

2.

## Refused

- Either argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [correlation](correlation.md)
- [variance_sample](variance-sample.md)
