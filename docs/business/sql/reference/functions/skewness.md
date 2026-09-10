# skewness

Divides the mean cubed deviation by the cube of the sample standard deviation. A positive result means the long tail runs high, a negative one means it runs low, and a symmetric series gives 0.0. The figure carries no small-sample adjustment, so it sits below the adjusted Fisher-Pearson form a spreadsheet reports. Fewer than two values, or no variance, gives 0.0.

## Syntax

```sql
skewness(values)
```

## Returns

DOUBLE PRECISION. NULL when the array is NULL or holds a non-number.

## Examples

```sql
SELECT skewness('[1,2,3,4,5]')
```

0, because the series is symmetric.

## Refused

- The argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [kurtosis](kurtosis.md)
- [stddev_sample](stddev-sample.md)
