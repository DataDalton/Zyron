# kurtosis

Divides the mean fourth deviation by the square of the sample variance and subtracts 3, so a normal distribution sits at 0.0. A positive result means heavier tails and a sharper peak than normal, a negative one means lighter tails and a flatter peak. Fewer than two values, or no variance, gives 0.0.

## Syntax

```sql
kurtosis(values)
```

## Returns

DOUBLE PRECISION. NULL when the array is NULL or holds a non-number.

## Examples

```sql
SELECT kurtosis('[1,2,3,4,5]')
```

-1.912, lighter tails than a normal distribution.

## Refused

- The argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [skewness](skewness.md)
- [variance_sample](variance-sample.md)
