# linear_regression

Fits y against x by ordinary least squares and returns the slope, the intercept and r squared, in that order. The r squared term is the fraction of the variance in y the line accounts for, and is 0.0 when y has no variance at all.

## Syntax

```sql
linear_regression(x, y)
```

## Returns

ARRAY of three numbers as JSON text. NULL when either array is NULL, holds a non-number, has a different length from the other, holds fewer than two values, or every x is the same.

## Examples

```sql
SELECT linear_regression('[1,2,3]', '[2,4,6]')
```

[2.0,0.0,1.0], a slope of 2 through the origin accounting for all of the variance.

## Refused

- Either argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [forecast_linear](forecast-linear.md)
- [correlation](correlation.md)
