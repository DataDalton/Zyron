# correlation

Divides the covariance of the two series by the product of their standard deviations, giving 1.0 for a perfect increasing straight line, -1.0 for a perfect decreasing one and 0.0 for no linear relationship. A series with no variance gives 0.0. Both arguments are JSON arrays of numbers held as text or binary.

## Syntax

```sql
correlation(x, y)
```

## Returns

DOUBLE PRECISION between -1.0 and 1.0. NULL when either array is NULL, holds a non-number, has a different length from the other, or holds fewer than two values.

## Examples

```sql
SELECT correlation('[1,2,3,4]', '[2,4,6,8]')
```

1.

## Refused

- Either argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [covariance](covariance.md)
- [linear_regression](linear-regression.md)
- [stddev_sample](stddev-sample.md)
